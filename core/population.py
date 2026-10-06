from dataclasses import dataclass

import config.config as cfg
import core.goods as gds


@dataclass
class Population:
    size: int
    income_pc: float              # NON-labor income per 100 people per tick (exogenous for now)

    wage: float = 1.0             # this province's wage per worker per tick
    number_employed: int = 0

    # Wages are paid during the tick and spent the following tick
    wage_income: float = 0.0      # wages received last tick -> part of this tick's budget
    _wage_accum: float = 0.0      # wages received so far this tick

    # DEMAND MECHANICS
    @property
    def budget(self) -> float:
        return self.size * self.income_pc / 100 + self.wage_income
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

    # INCOME MECHANICS
    def start_tick(self) -> None:
        """Last tick's wages become this tick's wage income."""
        self.wage_income = self._wage_accum
        self._wage_accum = 0.0

    def receive_wages(self, amount: float) -> None:
        self._wage_accum += max(0.0, amount)

    # LABOR MARKET MECHANICS
    @property
    def number_unemployed(self) -> int:
        return max(0, self.size - self.number_employed)

    @property
    def employment_rate(self) -> float:
        return self.number_employed / self.size if self.size > 0 else 0.0

    def update_wage(self) -> None:
        """
        Move the wage with labor-market tightness. The gap is normalized so that
        full employment gives +1 and employment of 2*target-1 or below gives -1;
        the wage then changes by at most WAGE_ADJ_SPEED per tick.
        """
        target = cfg.TARGET_EMPLOYMENT
        slack = max(1e-9, 1.0 - target)
        gap = max(-1.0, min(1.0, (self.employment_rate - target) / slack))
        self.wage = max(cfg.WAGE_MIN, self.wage * (1.0 + cfg.WAGE_ADJ_SPEED * gap))

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