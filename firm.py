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

    province: Province #CHANGE SPAWN FIRMS AND MARKET SEED TO MATCH

    base_MC: Optional[float] = None
    base_capacity: Optional[float] = None

    resource_rights: Optional[float] = None

    input_requirements: Dict[gds.GoodID, float] = field(init=False)
    input_inventory: Dict[gds.GoodID, float] = field(init=False)

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

        self.input_requirements = gds.PRODUCTION_RECIPES.get(self.good, {}).get('inputs', {}).copy()

        if self.treasury == 0.0 and self.start_capital != 0.0:
            self.treasury = float(self.start_capital)

        # self.input_requirements = gds.PRODUCTION_RECIPES[self.good]['inputs'].copy()
        # self.input_inventory = {good: 0.0 for good in self.input_requirements.keys()}



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

    # def resource_gathering(self):
    #     if self.firm_type is FirmType.RGO:
    #         resource_pool = self.province.resources
    #         possible_q = resource_pool * self.resource_rights
    #         return possible_q

    def resource_gathering(self) -> int:
        """
        For RGO firms, production capacity is their share of the province's
        resource pool for this good. For non-RGO, just fall back to capacity.
        """
        if self.firm_type is not FirmType.RGO:
            return int(self.capacity)

        # province.resources is a dict like {"grain": 300}
        pool = 0
        if hasattr(self.province, "resources") and self.province.resources is not None:
            pool = self.province.resources.get(self.good, 0)

        rights = self.resource_rights or 0.0
        possible_q = pool * rights

        return int(max(0, possible_q))


    def update_quantity(self, price: float, tick: int) -> None:
        if not self.active:
            self.q = 0
            # self._log_tick(tick, price, self.q)
            # self._last_quantity = self.q
            return

        if self.firm_type is FirmType.Manu:
            c = self.capacity

        elif self.firm_type is FirmType.RGO:
            c = self.resource_gathering()

        # First production tick: start low
        if self.last_quantity is None:
            target = int(c * 0.1)

        # Price below or equal to marginal cost → scale down
        elif price <= self.MC:
            target = max(0.0, self.last_quantity - c * 0.02)

        # Price above marginal cost → scale up
        else:
            target = min(self.last_quantity + c * 0.02, c)

        # Set final quantity (integer, capped)
        self.q = int(np.clip(target, 0.0, c))

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

        if self.neg_treasury_streak >= cfg.TREASURY_GRACE_TICKS:
            self.active = False

        row = self._rows[-1]
        row["revenue"] = float(TR)
        row["cost"] = float(TC)
        row["profit"] = float(profit)
        row["active"] = bool(self.active)
        row["treasury"] = float(self.treasury)

        self._cached_df = None  # invalidate cache
        return TR, TC, profit


def spawn_firms(
        good: gds.GoodID,
        firm_type: FirmType,
        rng: np.random.Generator,
        n: int,
        start_id: int = 0,
        province: Province = None
        ) -> List[Firm]:

    FC  = 20.0 * np.exp(rng.normal(cfg.FC_LOGMEAN, cfg.FC_LOGSD, size=n))
    MC  = np.clip(rng.normal(cfg.MC_MEAN, cfg.MC_SD, size=n), 0.5, None)
    CAP = rng.uniform(cfg.CAP_LOW, cfg.CAP_HIGH, size=n)

    # 1) Draw resource rights for this batch so that the total is < 1.0
    # Only RGO firms need resource rights
    if firm_type is FirmType.RGO and n > 0:
        # Random positive vector, normalized, then scaled to something below 1.0
        raw = rng.random(n)
        raw_sum = raw.sum()
        if raw_sum <= 0:
            shares = np.zeros(n)
        else:
            shares = raw / raw_sum

        total_rights = 0.99  # total < 1.0 by construction
        rights = total_rights * shares
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