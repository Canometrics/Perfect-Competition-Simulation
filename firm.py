from dataclasses import dataclass, field
from typing import Optional, Tuple
import pandas as pd
import numpy as np
from typing import List

import config as cfg
import goods as gds

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

    history: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(
        columns=[
            "tick", "quantity", "price", "revenue", "cost", "profit", "active", "treasury"
        ]
    ))

    def __post_init__(self):
        # Seed treasury from start_capital at creation
        if self.treasury == 0.0 and self.start_capital != 0.0:
            self.treasury = float(self.start_capital)

    def _last_quantity(self) -> Optional[float]:
        if len(self.history) == 0:
            return None
        return float(self.history.iloc[-1]["quantity"])

    def decide_target(self, price: float) -> float:
        if not self.active:
            return 0.0
        # Price-taking logic with constant MC: produce at capacity if P > MC; else 0
        if price > self.MC:
            return self.capacity
        elif price < self.MC:
            return 0.0
        else:
            return self.q

    def update_quantity(self, price: float, tick: int) -> None:
        # make nothing if shut down
        if not self.active:
            self.q = 0.0
            self.history.loc[len(self.history)] = {
                "tick": tick, "quantity": self.q, "price": price,
                "revenue": 0.0, "cost": 0.0, "profit": 0.0, "active": self.active, "treasury": self.treasury
            } # initialize new bookkeeping entry for firm
            return

        # return positive output if operating
        target = self.decide_target(price)
        q_proposed = float(np.clip(self.q + cfg.ADJ_RATE * (target - self.q), 0.0, self.capacity)) # clip s.t. not over capacity & not 0

        # Freeze tiny changes to reduce oscillation
        last_q = self._last_quantity()
        if last_q is not None and last_q > 0 and abs(q_proposed - last_q) / last_q < 0.05:
            q_proposed = last_q

        self.q = q_proposed
        self.history.loc[len(self.history)] = {
            "tick": tick, "quantity": self.q, "price": price,
            "revenue": 0.0, "cost": 0.0, "profit": 0.0, "active": self.active, "treasury": self.treasury
        } # initialize new bookkeeping entry for firm

    def book_finance(self, price: float, sales: float) -> Tuple[float, float, float]:
        TR = price * sales
        VC = self.MC * sales
        TC = self.FC + VC
        profit = TR - TC
        # write finance
        self.history.loc[self.history.index[-1], ["revenue", "cost", "profit"]] = [TR, TC, profit]

        # update treasury
        self.treasury += profit
        self.history.loc[self.history.index[-1], "treasury"] = self.treasury

        # Track consecutive negative-treasury ticks
        if self.treasury < 0:
            self.neg_treasury_streak += 1
        else:
            self.neg_treasury_streak = 0

        # Shutdown if negative treasury persists for grace period
        if self.neg_treasury_streak >= cfg.TREASURY_GRACE_TICKS:
            self.active = False

        # reflect active flag
        self.history.loc[self.history.index[-1], "active"] = self.active
        return TR, TC, profit



# helper to spawn firms
def spawn_firms(good: gds.GoodID, rng: np.random.Generator, n: int,start_id: int = 0, ) -> List[Firm]:
    """Vectorized firm creation for both initial seeding and late entry."""
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
            good = good,
            start_capital=float(cfg.START_CAPITAL)
        )
        for i in range(n)
    ]