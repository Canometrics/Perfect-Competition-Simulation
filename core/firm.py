from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, TYPE_CHECKING
from enum import Enum
import math

import pandas as pd
import numpy as np

import config.config as cfg
import core.goods as gds

if TYPE_CHECKING:
    from province import Province

_HISTORY_COLS = [
    "tick",
    "quantity",
    "price",
    "revenue",
    "cost",
    "profit",
    "active",
    "treasury",
    "employees",
    "output_inventory",
]


class FirmType(Enum):
    RGO = "rgo"
    Manu = "manu"
    # Office = "office"
    # Retail = "retail"

@dataclass
class Firm:
    id: int
    good: gds.GoodID
    province: "Province"
    FC: float
    MC: float
    capacity: int
    q: float

    firm_type: FirmType = field(init=False)

    base_MC: Optional[float] = None
    base_capacity: Optional[float] = None
    resource_rights: Optional[float] = None

    # what inputs per unit of this firm's output?
    input_requirements: Dict[gds.GoodID, float] = field(init=False)

    # stock of inputs (not yet fully used)
    input_inventory: Dict[gds.GoodID, int] = field(init=False)
    output_inventory: int = 0

    employees: int = 0

    active: bool = True

    start_capital: float = 0.0
    treasury: float = 0.0
    neg_treasury_streak: int = 0

    _rows: List[Dict] = field(default_factory=list, repr=False)

    _cached_df: Optional[pd.DataFrame] = field(default=None, repr=False)
    _last_quantity: Optional[float] = None

    @property
    def last_quantity(self) -> Optional[float]:
        return self._last_quantity

    @property
    def history(self) -> pd.DataFrame:
        """Materialize if needed, but read-only during simulation."""
        if self._cached_df is None:
            if self._rows:
                self._cached_df = pd.DataFrame.from_records(self._rows, columns=_HISTORY_COLS)
            else:
                self._cached_df = pd.DataFrame(columns=_HISTORY_COLS)
        return self._cached_df

    def __post_init__(self):
        # initialize firm type from good type
        if gds.is_raw(self.good):
            self.firm_type = FirmType.RGO
        else:
            self.firm_type = FirmType.Manu

        if self.firm_type is FirmType.RGO:
            self.capacity = self.resource_rights * self.province.resources.get(self.good, 0)

        # production recipe: inputs per unit of this firm's output good
        self.input_requirements = gds.PRODUCTION_RECIPES.get(self.good, {}).get('inputs', {}).copy()
        self.input_inventory = {g: 0 for g in self.input_requirements.keys()}

        if self.treasury == 0.0 and self.start_capital != 0.0: # <- this exists as a safeguard for good reason
            self.treasury = float(self.start_capital)

        # baseline MC before adding input costs (labor / tech)
        self.base_MC = float(self.MC)

        self.base_capacity = float(self.capacity)

    def update_input_cost(self, prices: Dict[gds.GoodID, float], wage: float) -> None:
        """
        Update this firm's MC to include:
          - the cost of its input bundle at current prices (for Manu firms)
          - the cost of labor per unit for all firms

        MC = base_MC + input_cost_per_unit + wage * labor_intensity
        """
        # Make sure we have a baseline MC (non input, non wage part)
        if self.base_MC is None: # for some reason this check is important
            self.base_MC = float(self.MC)

        # 1) Input bundle cost per unit of output (only if firm uses inputs)
        input_cost_per_unit = 0.0
        if self.input_requirements:
            for g_in, units in self.input_requirements.items():
                p_in = prices.get(g_in, 0.0)
                input_cost_per_unit += float(units) * float(p_in)

        # 2) Labor cost per unit of output
        recipe = gds.PRODUCTION_RECIPES.get(self.good, {})
        labor_intensity = float(recipe.get("labor_intensity", 0.0))  # workers per unit
        labor_cost_per_unit = wage * labor_intensity

        # 3) Effective marginal cost
        self.MC = float(self.base_MC + input_cost_per_unit + labor_cost_per_unit)

    def _effective_capacity(self) -> int:
        """
        Capacity used for production / planning:
        - RGO firms: limited by their share of the province's resource pool
        - Manu firms: limited by their normal factory capacity
        """
        # Manufacturing: normal capacity
        if self.firm_type is not FirmType.RGO:
            return int(self.capacity)

        # RGO firms: compute resource-limited capacity
        pool = self.province.resources.get(self.good, 0)

        rights = self.resource_rights or 0.0
        self.capacity = int(max(0, pool * rights))

        return self.capacity

    def plan_quantity(self, price: float) -> int:
        if cfg.PLANNING_RULE == "guess":
            return self.plan_quantity_guessing(price)
        return self.plan_quantity_optimize(price)


    def plan_quantity_guessing(self, price: float) -> int:
        if not self.active:
            return 0

        c = self._effective_capacity()

        # First production tick -> start moderately
        if self.last_quantity is None:
            return int(c * 0.05)

        # --- ±5 percent MC band ---
        lower = 0.97 * self.MC
        upper = 1.03 * self.MC

        # If price in band, keep quantity stable
        if lower <= price <= upper:
            return int(self.last_quantity)

        # Price too low -> scale down
        if price < lower:
            return int(max(0.0, self.last_quantity - c * 0.05))

        # Price too high -> scale up
        return int(min(self.last_quantity + c * 0.05, c))

    def plan_quantity_optimize(self, price: float) -> int: # this is a better option!
        if not self.active:
            return 0

        c = self._effective_capacity()

        if self.MC <= 0:
            return c

        # profit margin as fraction of price: 0 when price=MC, 1 when price→∞
        margin = max(0.0, (price - self.MC) / price)
        target = int(c * margin)

        if self.last_quantity is None:
            return int(c * 0.05)

        # smooth adjustment toward target — max 10% capacity step per tick
        step = c * 0.10
        if target > self.last_quantity:
            return int(min(target, self.last_quantity + step))
        elif target < self.last_quantity:
            return int(max(target, self.last_quantity - step))
        return int(self.last_quantity)

    def update_quantity(self, price: float, tick: int) -> None:
        """
        Apply the planning rule (plan_quantity), then apply the labor constraint
        via hire_and_fire, and commit the final quantity.
        """
        if not self.active:
            self.q = 0
            return
        # self.output_inventory = 0 # just to disable inventory for now

        # Planned quantity given price and capacity / resources
        q_desired = self.plan_quantity(price)

        # Treat q_desired as TOTAL we want available (prod + inventory).
        # Only produce what we are missing, never less than 0.
        inv = int(max(0, self.output_inventory))
        q_to_produce = max(0, q_desired - inv)

        # Labor-constrained feasible quantity of NEW production
        q_final = self.hire_and_fire(q_to_produce, hypothetical=False) + inv

        self.q = int(max(0, q_final))
        self.q = q_to_produce # to disable labor market, this is not logically correct but it works
        self.output_inventory = 0

        self._log_tick(tick, price, self.q)
        self._last_quantity = self.q

    def hire_and_fire(self, desired_q: int, hypothetical: bool) -> int:
        """
        Adjust this firm's employment toward the level needed for desired_q,
        subject to the available labor in the province.

        Returns the *feasible* output quantity given actual employment.
        """
        pop = self.province.population

        # Labor needed per unit of output for this good
        recipe = gds.PRODUCTION_RECIPES.get(self.good, {})
        intensity = float(recipe.get("labor_intensity", 0.0))

        desired_q = max(0, int(desired_q))

        # Desired headcount from planned output
        desired_headcount = int(max(0, math.ceil(desired_q * intensity)))

        # Max headcount we could possibly have = current employees + unemployed pool
        max_headcount = self.employees + pop.number_unemployed
        target_headcount = min(desired_headcount, max_headcount)

        delta = target_headcount - self.employees

        if not hypothetical:
            if delta > 0:
                # Try to hire up to delta workers from the province population
                hired = pop.hired(delta)
                self.employees += hired

            elif delta < 0:
                # Fire up to -delta workers and return them to the unemployment pool
                to_fire = -delta
                fired = pop.fired(to_fire)
                self.employees -= fired

            # guard
            if self.employees < 0:
                self.employees = 0

        # Given actual employees, how much can we produce?
        q_from_labor = int(self.employees / intensity)

        # Final output cannot exceed planned quantity
        return int(min(desired_q, q_from_labor))


    def book_finance(self, price: float, sales: float) -> Tuple[float, float, float]:
        TR = price * sales
        VC = self.MC * sales
        TC = self.FC + VC
        profit = TR - TC

        self.treasury += profit

        if self.treasury < 0:
            self.neg_treasury_streak += 1
        else:
            self.neg_treasury_streak = 0

        # --- handle death + freeing resource rights ---
        if self.neg_treasury_streak >= cfg.TREASURY_GRACE_TICKS and self.active:
            # firm just died this tick
            self.active = False

            # Only RGOs have meaningful resource rights to free
            if getattr(self, "firm_type", None) is FirmType.RGO and hasattr(self, "province"):
                prov = self.province
                good = self.good

                # make sure province has a rights_given dict
                if not hasattr(prov, "rights_given"):
                    prov.rights_given = {}

                # current endowed rights for this good in this province
                current = float(prov.rights_given.get(good, 0.0))
                # treat None as 0.0
                rr = float(self.resource_rights or 0.0)

                prov.rights_given[good] = max(0.0, current - rr)
                # firm no longer holds any rights
                self.resource_rights = 0.0

            # release workers back to the province labor pool
            if self.employees > 0 and getattr(self.province, "population", None) is not None:
                self.province.population.fired(self.employees)
                self.employees = 0

        # --- logging ---
        row = self._rows[-1]
        row["revenue"] = float(TR)
        row["cost"] = float(TC)
        row["profit"] = float(profit)
        row["active"] = bool(self.active)
        row["treasury"] = float(self.treasury)
        row["output_inventory"] = float(self.output_inventory)
        row["employees"] = int(self.employees)

        self._cached_df = None  # invalidate cache
        return TR, TC, profit

    def _log_tick(self, tick: int, price: float, q: float):
        self._rows.append({
            "tick": tick,
            "quantity": float(q),
            "price": float(price),
            "revenue": 0.0,
            "cost": 0.0,
            "profit": 0.0,
            "active": bool(self.active),
            "treasury": float(self.treasury),
            "employees": int(self.employees),
            "output_inventory": float(self.output_inventory),
        })
        self._cached_df = None  # invalidate cache


def draw_resource_rights(
    province: Province,
    good,
    rng,
    n: int,
    cap: float = 1.0
) -> np.ndarray:
    """
    Allocate resource rights up to `cap` (<= 1.0) for this province+good.
    """
    used = province.rights_given.get(good, 0.0)

    # never allow cap > 1.0
    remaining = max(0.0, cap - used)

    if remaining <= 0.0:
        # no rights left under this cap
        return np.zeros(n)

    raw = rng.uniform(0.01, 0.05, size=n)
    shares = raw / raw.sum()
    rights = remaining * shares

    # update province bookkeeping
    province.rights_given[good] = used + rights.sum()
    return rights


def spawn_firms(
        good: gds.GoodID,
        firm_type: FirmType,
        rng: np.random.Generator,
        n: int,
        start_id: int = 0,
        province: Province = None,
        max_share: float = 1.0,
        ) -> List[Firm]:

    FC  = 20.0 * np.exp(rng.normal(cfg.FC_LOGMEAN, cfg.FC_LOGSD, size=n))
    MC  = np.clip(rng.normal(cfg.MC_MEAN, cfg.MC_SD, size=n), 0.5, None)
    CAP = rng.uniform(cfg.CAP_LOW, cfg.CAP_HIGH, size=n) # np.full(shape=n, fill_value=80)

    # 1) Draw resource rights for this batch so that the total is < 1.0
    # Only RGO firms need resource rights
    if firm_type is FirmType.RGO and n > 0:
        rights = draw_resource_rights(
            province,
            good,
            rng,
            n,
            cap=max_share
        )
    else:
        rights = np.zeros(n)

    # 2) Build firms, assigning resource_rights from the vector above
    firms: List[Firm] = []
    for i in range(n):
        f = Firm(
            id=start_id + i,
            FC=float(FC[i]),
            MC=float(MC[i]),
            base_capacity=float(CAP[i]),
            capacity=int(CAP[i]),
            q=0.0,
            good=good,
            province=province,
            resource_rights=float(rights[i]) if firm_type is FirmType.RGO else None,
            start_capital=float(cfg.START_CAPITAL)
        )
        firms.append(f)

        # Provinces are the real owners: register the firm there
        if province is not None:
            province.firms.append(f)

    return firms
