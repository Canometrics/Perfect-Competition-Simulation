from dataclasses import dataclass

import core.goods as gds


@dataclass
class Population:
    size: int
    income_pc: float

    number_employed: int = 0

    # DEMAND MECHANICS
    @property
    def budget(self) -> float:
        return self.size * self.income_pc / 100
    # later implement MPS/C

    def pop_demand(self, prices: dict[gds.GoodID, float]) -> dict[gds.GoodID, int]:
        B = self.budget
        demand: dict[gds.GoodID, int] = {g: 0 for g in gds.COBB_DOUGLAS_WEIGHTS}

        total_weight = sum(gds.COBB_DOUGLAS_WEIGHTS.values())
        if total_weight <= 0 or B <= 0:
            return demand

        for g, w in gds.COBB_DOUGLAS_WEIGHTS.items():
            p = prices.get(g, 0.0)
            if p <= 0:
                continue
            budget_share = (w / total_weight) * B
            demand[g] = budget_share / p

        return demand

    # LABOR MARKET MECHANICS
    @property
    def number_unemployed(self) -> int:
        return max(0, self.size - self.number_employed)

    def hired(self, n: int) -> int:
        """
        Hire up to n people from this population class.
        Returns the actual number hired.
        """
        can_hire = min(n, self.number_unemployed)
        self.number_employed += can_hire
        return can_hire

    def fired(self, n: int) -> int:
        """
        Fire up to n people.
        """
        can_fire = min(n, self.number_employed)
        self.number_employed -= can_fire
        return can_fire
