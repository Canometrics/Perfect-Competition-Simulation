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
    # later implement MPS/C

    @property
    def needs_all_goods(self) -> Dict[gds.GoodID, Tuple[int, int, int]]:
        """
        Return needs for every good, dynamically,
        """
        out: Dict[gds.GoodID, Tuple[int, int, int]] = {}

        for good, tiers in self.goods_for_tier.items():
            q_life  = math.ceil(tiers['life']     * self.size / 100)
            q_every = math.ceil(tiers['everyday'] * self.size / 100)
            q_lux   = math.ceil(tiers['luxury']   * self.size / 100)

            out[good] = (q_life, q_every, q_lux)

        return out

    def needs_per_good(self, good: gds.GoodID) -> Tuple[int, int, int]:
        return self.needs_all_goods[good]


    def demand_for_all_goods(self, prices: Dict[gds.GoodID, float]) -> Dict[gds.GoodID, int]:
        needs = self.needs_all_goods
        B = self.budget
        demand: Dict[gds.GoodID, int] = {g: 0 for g in needs.keys()}


        def tier_cost_and_fill(targets: Dict[gds.GoodID, float], budget: float) -> float:
            """Attempt to buy this tier. If budget < total cost, buy proportionally and stop.
            Returns remaining budget after this tier."""
            # Build cost vector only for goods with positive prices and positive targets
            valid = [
                (g, q, prices.get(g, 0.0))
                for g, q in targets.items()
                if q > 0 and prices.get(g, 0.0) > 0
                ]
            if not valid:
                return budget

            total_cost = sum(q * p for _, q, p in valid)

            if budget >= total_cost:
                # buy all
                for g, q, _p in valid:
                    demand[g] += q
                return budget - total_cost
            else:
                # buy proportionally
                if total_cost > 0:
                    f = budget / total_cost
                else:
                    f = 0
                for g, q, _p in valid:
                    demand[g] += f * q
                return 0

        # Build tier targets
        # life = life
        life_targets = {g: t[0] for g, t in needs.items()}

        # everyday increment = everyday - life (never negative)
        every_inc = {g: max(0, t[1]) for g, t in needs.items()}

        # luxury increment = luxury - everyday (never negative)
        lux_inc = {g: max(0, t[2]) for g, t in needs.items()}

        # 1) life
        B = tier_cost_and_fill(life_targets, B)
        if B <= 0:
            return demand

        # 2) everyday increment
        B = tier_cost_and_fill(every_inc, B)
        if B <= 0:
            return demand

        # 3) luxury increment
        B = tier_cost_and_fill(lux_inc, B)
        return demand

    def demand_per_good(
            self,
            prices: Dict[gds.GoodID, float],
            good: gds.GoodID) -> int:
        return self.demand_for_all_goods(prices)[good]

    def buy_for_good(
        self,
        good: gds.GoodID,
        price: float,
        available_q: int,
        remaining_budget: float,
) -> tuple[int, str, dict[str, int], float]:
        """
        Buy for a single good in tier order using remaining_budget.
        Returns: (total_bought, label, tier_breakdown, remaining_budget_after)
        """
        p = max(0.01, float(price))
        avail = int(max(0, available_q))
        B = float(max(0.0, remaining_budget))

        # Needs for this good
        q_life, q_every, q_lux = self.needs_per_good(good)
        # Convert to increments (never negative)
        inc_life  = q_life
        inc_every = q_every
        inc_lux   = q_lux

        tiers_satisfied = {"life": 0, "everyday": 0, "luxury": 0}

        def buy(q_target: int) -> int:
            """Buy up to q_target, limited by availability and budget. Return bought qty."""
            nonlocal B, avail
            if q_target <= 0 or avail <= 0 or B < p:
                return 0
            max_by_budget = math.floor(B // p)           # whole units we can afford
            q = min(q_target, avail, max_by_budget)
            if q > 0:
                B -= q * p
                avail -= q
            return q

        # 1) Life tier
        tiers_satisfied["life"] = buy(inc_life)
        if tiers_satisfied["life"] < inc_life:
            return sum(tiers_satisfied.values()), "life_partial", tiers_satisfied

        # 2) Everyday increment
        tiers_satisfied["everyday"] = buy(inc_every)
        if tiers_satisfied["everyday"] < inc_every:
            return sum(tiers_satisfied.values()), "life", tiers_satisfied

        # 3) Luxury increment
        tiers_satisfied["luxury"] = buy(inc_lux)
        label = "luxury" if tiers_satisfied["luxury"] >= inc_lux else "everyday"

        return sum(tiers_satisfied.values()), label, tiers_satisfied
