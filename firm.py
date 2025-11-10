from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
import pandas as pd
import numpy as np

import config as cfg
import goods as gds

_HISTORY_COLS = [
    "tick", "quantity", "price",
    "revenue", "cost", "profit",
    "active", "treasury"
]

@dataclass
class Firm:
    id: int
    good: gds.GoodID
    FC: float
    MC: float
    capacity: int
    q: float

    base_MC: Optional[float] = None
    base_capacity: Optional[float] = None

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
        if self.treasury == 0.0 and self.start_capital != 0.0:
            self.treasury = float(self.start_capital)


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

    def update_quantity(self, price: float, tick: int) -> None:
        if not self.active:
            self.q = 0
            # self._log_tick(tick, price, self.q)
            # self._last_quantity = self.q
            return

        c = self.capacity

        # First production tick: start low
        if self.last_quantity is None:
            target = int(c * 0.02)

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

    # old quantity method
    # def update_quantity(self, price: float, tick: int) -> None:
    #     if not self.active:
    #         self.q = 0.0
    #         self._log_tick(tick, price, self.q)
    #         self._last_quantity = self.q
    #         return

    #     c = self.capacity

    #     # --- old decide_target() logic inlined ---
    #     if self.last_quantity is None:
    #         target = c * 0.05
    #     elif price <= self.MC:
    #         target = max(0.0, self.last_quantity - c * 0.1)
    #     else:
    #         target = min(self.last_quantity + c * 0.1, c)

    #     # --- old update_quantity gradual adjustment ---
    #     q_new = float(np.clip(self.q + cfg.ADJ_RATE * (target - self.q), 0.0, c))
    #     self.q = q_new

    #     # log + update memory
    #     self._log_tick(tick, price, self.q)
    #     self._last_quantity = self.q


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


def spawn_firms(good: gds.GoodID, rng: np.random.Generator, n: int, start_id: int = 0) -> List[Firm]:
    FC  = 20.0 * np.exp(rng.normal(cfg.FC_LOGMEAN, cfg.FC_LOGSD, size=n))
    MC  = np.clip(rng.normal(cfg.MC_MEAN, cfg.MC_SD, size=n), 0.5, None)
    CAP = rng.uniform(cfg.CAP_LOW, cfg.CAP_HIGH, size=n)

    return [
        Firm(
            id=start_id + i,
            FC=float(FC[i]),
            MC=float(MC[i]),
            base_capacity=float(CAP[i]),
            capacity=int(CAP[i]),
            q=0.0,
            good=good,
            start_capital=float(cfg.START_CAPITAL)
        )
        for i in range(n)
    ]
