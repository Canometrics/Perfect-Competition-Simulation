from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
from enum import Enum

import pandas as pd
import numpy as np

import config as cfg
import goods as gds
from province import Province

_HISTORY_COLS = [
    "tick", "quantity", "price",
    "revenue", "cost", "profit",
    "active", "treasury"
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
    FC: float
    MC: float
    capacity: int
    q: float

    firm_type: FirmType = field(init=False)

    province: Province  # province where the firm is located

    base_MC: Optional[float] = None #labor and overhead rn
    base_capacity: Optional[float] = None

    resource_rights: Optional[float] = None

    # what inputs per unit of this firm's output?
    input_requirements: Dict[gds.GoodID, float] = field(init=False)

    # stock of inputs (not yet fully used)
    input_inventory: Dict[gds.GoodID, int] = field(init=False)
    output_inventory: int = 0.0

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

        self.output_inventory = 0

        # production recipe: inputs per unit of this firm's output good
        self.input_requirements = gds.PRODUCTION_RECIPES.get(self.good, {}).get('inputs', {}).copy()
        self.input_inventory = {g: 0.0 for g in self.input_requirements.keys()}

        if self.treasury == 0.0 and self.start_capital != 0.0:
            self.treasury = float(self.start_capital)

        # baseline MC before adding input costs (labor / tech)
        if self.base_MC is None:
            self.base_MC = float(self.MC)

        if self.base_capacity is None:
            self.base_capacity = float(self.capacity)

    def update_input_cost(self, prices: Dict[gds.GoodID, float]) -> None:
        """
        Update this firm's MC to include the cost of its input bundle at current prices.
        Only relevant for manufacturing firms with non-empty input_requirements.
        """
        # Only Manu firms actually buy inputs
        if self.firm_type is not FirmType.Manu:
            return

        if not self.input_requirements:
            return

        # Make sure we have a baseline MC (labor / overhead piece)
        if self.base_MC is None:
            self.base_MC = float(self.MC)

        # Cost of one unit of output's input bundle at current prices
        input_cost_per_unit = 0.0
        for g, units in self.input_requirements.items():
            p_in = prices.get(g, 0.0)
            input_cost_per_unit += float(units) * float(p_in)

        # Effective marginal cost = baseline + input bundle
        self.MC = float(self.base_MC + input_cost_per_unit)

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
        pool = 0
        if hasattr(self.province, "resources") and self.province.resources is not None:
            pool = self.province.resources.get(self.good, 0)

        rights = self.resource_rights or 0.0
        possible_q = pool * rights

        return int(max(0, possible_q))

    def plan_quantity(self, price: float) -> int:
        """
        Decide desired output quantity for this tick based on price vs MC
        and last period's quantity.
        This is used both by the simulation to compute input demand
        and by update_quantity to actually set q.
        """
        if not self.active:
            return 0

        c = self._effective_capacity()

        # First production tick: start low
        if self.last_quantity is None:
            target = int(c * 0.1)
        # Price below or equal to marginal cost → scale down
        elif price <= self.MC:
            target = max(0.0, self.last_quantity - c * 0.02)
        # Price above marginal cost → scale up
        else:
            target = min(self.last_quantity + c * 0.02, c)

        return int(np.clip(target, 0.0, c))

    def update_quantity(self, price: float, tick: int) -> None:
        """
        Apply the planning rule (plan_quantity) and commit the quantity.
        """
        if not self.active:
            self.q = 0
            return

        q_new = self.plan_quantity(price)
        self.q = int(q_new)

        self._log_tick(tick, price, self.q)
        self._last_quantity = self.q

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

        # --- logging ---
        row = self._rows[-1]
        row["revenue"] = float(TR)
        row["cost"] = float(TC)
        row["profit"] = float(profit)
        row["active"] = bool(self.active)
        row["treasury"] = float(self.treasury)
        row["output_inventory"] = float(self.output_inventory)

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
    cap = min(1.0, float(cap))
    remaining = max(0.0, cap - used)

    if remaining <= 0.0:
        # no rights left under this cap
        return np.zeros(n)

    raw = rng.uniform(0.01, 1.0, size=n)
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
    CAP = rng.uniform(cfg.CAP_LOW, cfg.CAP_HIGH, size=n)

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
    return [
        Firm(
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
        for i in range(n)
    ]