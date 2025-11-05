
from dataclasses import dataclass
import math

@dataclass
class Population:
    size: int
    income_pc: float
    NEED_LIFE: int = 30.0     # per 100 ppl (total)
    NEED_EVERY: int = 50.0    # per 100 ppl (total)
    NEED_LUX: int = 100.0     # per 100 ppl (total)

    @property
    def budget(self) -> float:
        return self.size * self.income_pc / 100

    def target_qty(self, price: float):
        """
        Compute affordable demand at current price by walking tiers in order:
        life -> everyday -> luxury. If the next tier is not fully affordable,
        buy as much of the current tier as the remaining budget allows.
        Returns (total_qty, label_of_highest_tier_reached_or_partial).
        """
        q_life = math.ceil(self.NEED_LIFE / 100.0 * self.size)
        q_every = math.ceil(self.NEED_EVERY / 100.0 * self.size)
        q_lux  = math.ceil(self.NEED_LUX  / 100.0 * self.size)

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
