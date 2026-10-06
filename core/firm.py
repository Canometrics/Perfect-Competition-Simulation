from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd

import config.config as cfg
import core.goods as gds

if TYPE_CHECKING:
    from core.market import Market
    from core.province import Province

_HISTORY_COLS = [
    "tick",
    "quantity",
    "produced",
    "price",
    "input_spend",
    "wage_bill",
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
    # --- Identity ---
    id: int
    good: gds.GoodID
    province: Province

    # --- Core economics (required at construction) ---
    FC: float          # fixed cost
    MC: float          # marginal cost
    capacity: int
    q: float           # current output/quantity

    # --- Derived / computed fields (set in __post_init__, not passed in) ---
    firm_type: FirmType = field(init=False)
    input_requirements: dict[gds.GoodID, float] = field(init=False)
    input_inventory: dict[gds.GoodID, float] = field(init=False)
    labor_intensity: float = field(init=False)   # workers per unit of output, from the recipe

    # --- Costs and rights ---
    base_MC: float | None = None
    base_capacity: int | None = None
    resource_rights: float | None = None

    # --- Production state ---
    q_produced: int = 0        # new units produced this tick (q = q_produced + inventory brought to market)
    output_inventory: int = 0
    employees: int = 0
    market: Market | None = field(default=None, repr=False)  # back-reference, set by the Market that owns this firm

    # --- Input purchasing ---
    input_requested: dict[gds.GoodID, float] = field(default_factory=dict)  # what the firm tried to buy this tick
    input_bought: dict[gds.GoodID, float] = field(default_factory=dict)     # what it actually got this tick
    input_cost_per_unit: float = 0.0   # input bundle value per unit at current prices (part of MC, for planning)
    input_spend: float = 0.0           # cash actually paid for inputs this tick, booked in book_finance
    active: bool = True

    # --- Financials ---
    start_capital: float = 0.0
    treasury: float = 0.0
    neg_treasury_streak: int = 0

    # --- Internal bookkeeping / caching ---
    _rows: list[dict] = field(default_factory=list, repr=False)
    _cached_df: pd.DataFrame | None = field(default=None, repr=False)
    _last_quantity: int | None = None
    last_profit: float = 0.0      # this tick's profit, always kept (history is optional)

    @property
    def last_quantity(self) -> float | None:
        return self._last_quantity

    @property
    def history(self) -> pd.DataFrame:
        """Materialize if needed, but read-only during simulation."""
        if self._cached_df is None:
            if self._rows:
                self._cached_df = pd.DataFrame.from_records(self._rows, columns=_HISTORY_COLS)
            else:
                self._cached_df = pd.DataFrame(columns=_HISTORY_COLS)
            self._cached_df["good"] = self.good
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
        recipe = gds.PRODUCTION_RECIPES.get(self.good, {})
        self.input_requirements = recipe.get('inputs', {}).copy()
        self.labor_intensity = float(recipe.get('labor_intensity', 0.0))
        self.input_inventory = {g: 0.0 for g in self.input_requirements}

        if self.treasury == 0.0 and self.start_capital != 0.0: # <- this exists as a safeguard for good reason
            self.treasury = float(self.start_capital)

        # baseline MC before adding input costs (labor / tech)
        self.base_MC = float(self.MC)

        self.base_capacity = float(self.capacity)

    def update_input_cost(self, prices: dict[gds.GoodID, float], wage: float) -> None:
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
        labor_cost_per_unit = wage * self.labor_intensity

        # Keep the input part separately: it's in MC for planning, but in the books
        # inputs are charged at what was actually paid for them (input_spend).
        self.input_cost_per_unit = input_cost_per_unit

        # 3) Effective marginal cost
        self.MC = float(self.base_MC + input_cost_per_unit + labor_cost_per_unit)

    def plan_quantity(self, price: float) -> int:
        """
        called in country.country_step
        """
        if not self.active:
            return 0

        c = self._effective_capacity()

        if self.MC <= 0:
            return c

        if self.last_quantity is None:
            return int(c * 0.05)

        # profit margin as fraction of price: 0 when price=MC, 1 when price -> infinity
        margin = max(0.0, (price - self.MC) / price)
        target = int(c * margin)

        # smooth adjustment toward target — max 10% capacity step per tick
        step = c * 0.10
        if target > self.last_quantity:
            return int(min(target, self.last_quantity + step))
        elif target < self.last_quantity:
            return int(max(target, self.last_quantity - step))
        return int(self.last_quantity)

    def update_quantity(self, price: float, tick: int) -> None:
        """
        Decide this tick's production and how much to bring to market.

        plan_quantity gives the TOTAL the firm wants available to sell. Inventory
        covers part of that, so the firm only produces the gap, limited by the labor
        it can actually hire. New production plus all inventory goes to market;
        whatever doesn't sell returns to inventory in book_finance.
        """
        if not self.active:
            self.q = 0
            self.q_produced = 0
            self.input_requested = {}
            self.input_bought = {}
            return

        q_desired = self.plan_quantity(price)
        inv = int(max(0, self.output_inventory))
        q_to_produce = max(0, q_desired - inv)

        # Input-constrained, then labor-constrained new production
        self.q_produced = self.hire_and_fire(self.buy_inputs(q_to_produce))
        self._use_inputs(self.q_produced)

        # Everything on hand goes to market
        self.q = self.q_produced + inv
        self.output_inventory = 0

        self._log_tick(tick, price, self.q, self.q_produced)
        self._last_quantity = self.q

    def buy_inputs(self, desired_q: int) -> int:
        """
        Buy the inputs needed to produce desired_q, net of input stock already on
        hand, from what upstream firms have brought to market this tick.

        Only buys complete bundles: if one input is scarce, the firm scales down its
        purchases of the others too, so it never pays for inputs it can't combine.
        Returns the quantity the firm now has inputs for (<= desired_q).
        RGOs have no inputs and get desired_q back unchanged.
        """
        self.input_requested = {}
        self.input_bought = {}
        if not self.input_requirements or desired_q <= 0:
            return desired_q

        markets = self.market.country.markets

        # Record what we'd need to buy for the full plan (this is firm demand)
        for g, units in self.input_requirements.items():
            self.input_requested[g] = max(0.0, desired_q * units - self.input_inventory[g])

        # How much output can on-hand stock + what's still for sale support?
        q_buy = float(desired_q)
        for g, units in self.input_requirements.items():
            obtainable = self.input_inventory[g] + markets[g].available_to_firms()
            q_buy = min(q_buy, obtainable / units)
        q_buy = max(0, int(q_buy + 1e-9))

        # Buy exactly enough for q_buy
        for g, units in self.input_requirements.items():
            need = max(0.0, q_buy * units - self.input_inventory[g])
            if need > 0:
                m = markets[g]
                got = m.sell_to_firm(need)
                self.input_inventory[g] += got
                self.input_bought[g] = got
                self.input_spend += got * m.price

        return q_buy

    def _use_inputs(self, q: int) -> None:
        """Consume inputs for q units of output. Unused inputs stay in stock for next tick."""
        for g, units in self.input_requirements.items():
            self.input_inventory[g] = max(0.0, self.input_inventory[g] - q * units)

    def hire_and_fire(self, desired_q: int) -> int:
        """
        Move employment toward the headcount needed for desired_q, limited by the
        province's unemployed pool. Returns the feasible output given actual employees.
        """
        pop = self.province.population

        intensity = self.labor_intensity
        if intensity <= 0:
            return int(desired_q)  # no labor needed

        desired_headcount = max(0, math.ceil(desired_q * intensity))
        delta = desired_headcount - self.employees

        if delta > 0:
            self.employees += pop.hired(delta)   # capped by unemployed pool
        elif delta < 0:
            self.employees -= pop.fired(-delta)
        self.employees = max(self.employees, 0)

        # small epsilon so float division doesn't round e.g. 24.999999 down to 24
        q_from_labor = int(self.employees / intensity + 1e-9)
        return int(min(desired_q, q_from_labor))

    def book_finance(self, price: float, sales: float) -> tuple[float, float, float]:
        """
        Revenue comes from units sold. Costs:
          - base cost on units PRODUCED this tick (unsold units were still made)
          - wages for every worker employed this tick, at the province wage;
            this money is paid to the province's households
          - inputs at what was actually paid for them this tick (input_spend)
        MC (wage * labor_intensity + inputs at current prices) is only used for planning.
        Unsold units return to inventory and carry no further cost when sold later.
        """
        unsold = max(self.q - sales, 0.0)
        self.output_inventory += round(unsold)

        pop = self.province.population
        wage_bill = pop.wage * self.employees
        pop.receive_wages(wage_bill)

        TR = price * sales
        VC = self.base_MC * self.q_produced + wage_bill + self.input_spend
        TC = self.FC + VC
        spend = self.input_spend
        self.input_spend = 0.0
        profit = TR - TC

        self.treasury += profit
        self.last_profit = float(profit)

        if self.treasury < 0:
            self.neg_treasury_streak += 1
        else:
            self.neg_treasury_streak = 0

        # --- handle death + freeing resource rights ---
        if self.neg_treasury_streak >= cfg.TREASURY_GRACE_TICKS and self.active:
            # firm just died this tick
            self.active = False

            # Only RGOs have resource rights to free
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
        if not cfg.RECORD_FIRM_HISTORY:
            return TR, TC, profit
        row = self._rows[-1]
        row["revenue"] = float(TR)
        row["input_spend"] = float(spend)
        row["wage_bill"] = float(wage_bill)
        row["cost"] = float(TC)
        row["profit"] = float(profit)
        row["active"] = bool(self.active)
        row["treasury"] = float(self.treasury)
        row["output_inventory"] = float(self.output_inventory)
        row["employees"] = int(self.employees)

        self._cached_df = None  # invalidate cache
        return TR, TC, profit

    def _effective_capacity(self) -> int:
        """
        Capacity used for production / planning:
        - RGO firms: limited by their share of the province's resource pool
        - Manu firms: limited by their normal factory capacity

        If I ever make province resource pools dynamic, this would be helpful, but right now it is not needed
        Called in plan_quantity()
        """
        # Manufacturing: normal capacity
        if self.firm_type is not FirmType.RGO:
            return int(self.capacity)

        # RGO firms: compute resource-limited capacity
        pool = self.province.resources.get(self.good, 0)

        rights = self.resource_rights or 0.0
        self.capacity = int(max(0, pool * rights))

        return self.capacity

    def _log_tick(self, tick: int, price: float, q: float, produced: float):
        if not cfg.RECORD_FIRM_HISTORY:
            return
        self._rows.append({
            "tick": tick,
            "quantity": float(q),
            "produced": float(produced),
            "price": float(price),
            "input_spend": 0.0,
            "wage_bill": 0.0,
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
    cap: float = 1.0,
    fill_remaining: bool = False,
) -> np.ndarray:
    """
    Allocate resource rights up to `cap` (<= 1.0) for this province+good.

    fill_remaining=True (initial seeding): split everything left under `cap`
    among the n firms.
    fill_remaining=False (entry): each firm draws its own share from
    [ENTRANT_RIGHTS_LOW, ENTRANT_RIGHTS_HIGH], capped by what's left, so a lone
    entrant can't take every right freed by exits.
    """
    used = province.rights_given.get(good, 0.0)

    # never allow cap > 1.0
    remaining = max(0.0, cap - used)

    if remaining <= 0.0:
        # no rights left under this cap
        return np.zeros(n)

    if fill_remaining:
        raw = rng.uniform(0.01, 0.05, size=n)
        rights = remaining * raw / raw.sum()
    else:
        raw = rng.uniform(cfg.ENTRANT_RIGHTS_LOW, cfg.ENTRANT_RIGHTS_HIGH, size=n)
        rights = raw * min(1.0, remaining / raw.sum())

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
        fill_rights: bool = False,
        ) -> list[Firm]:

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
            cap=max_share,
            fill_remaining=fill_rights,
        )
    else:
        rights = np.zeros(n)

    # 2) Build firms, assigning resource_rights from the vector above
    firms: list[Firm] = []
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