from typing import Dict, Tuple
from dataclasses import dataclass, field
import math
from copy import deepcopy

import goods as gds

@dataclass
class Population:
    size: int
    income_pc: float

    goods_for_tier: Dict[gds.GoodID, Dict[str, float]] = field(
        default_factory=lambda: deepcopy(gds.NEEDS_PER_GOOD)
    )

    @property
    def budget(self) -> float:
        return self.size * self.income_pc / 100


    def needs_per_good(self, good: gds.GoodID) -> Tuple[int, int, int]:
        tiers = self.goods_for_tier.get(good, self.goods_for_tier[gds.GOODS[0]])
        q_life = math.ceil(tiers['life'] * self.size / 100)
        q_every = math.ceil(tiers['everyday'] * self.size / 100)
        q_lux = math.ceil(tiers['luxury'] * self.size / 100)
        return q_life, q_every, q_lux


    def target_qty_per_good(self, price: float, good: gds.GoodID):
        """
        Compute affordable demand at current price by walking tiers in order:
        life -> everyday -> luxury. If the next tier is not fully affordable,
        buy as much of the current tier as the remaining budget allows.
        Returns (total_qty, label_of_highest_tier_reached_or_partial).
        """
        q_life, q_every, q_lux = self.needs_per_good(good)
        p = max(0.01, price)
        B = self.budget

        # 1) Life
        buy_life = math.ceil(min(q_life, B / p))
        B -= p * buy_life
        if buy_life < q_life:
            return buy_life, "life_partial"

        # 2) Everyday
        buy_every = math.ceil(min(q_every, B / p))
        B -= p * buy_every
        if buy_every < q_every:
            return buy_life + buy_every, "life"

        # 3) Luxury
        buy_lux = math.ceil(min(q_lux, B / p))
        return buy_life + buy_every + buy_lux, ("luxury" if buy_lux >= q_lux else "everyday")

    def realized_qty(self, price: float, available_q: int, good: gds.GoodID):
        p = max(0.01, price)
        B = self.budget
        can_buy = math.ceil(min(B / p, available_q))

        q_life, q_every, q_lux = self.needs_per_good(good)

        tiers = {"life": 0, "everyday": 0, "luxury": 0}

        tiers["life"] = min(q_life, can_buy); can_buy -= tiers["life"]
        if tiers["life"] < q_life:
            return sum(tiers.values()), "life_partial", tiers

        tiers["everyday"] = min(q_every, can_buy); can_buy -= tiers["everyday"]
        if tiers["everyday"] < q_every:
            return sum(tiers.values()), "life", tiers

        tiers["luxury"] = min(q_lux, can_buy)
        label = "luxury" if tiers["luxury"] >= q_lux else "everyday"
        return sum(tiers.values()), label, tiers
