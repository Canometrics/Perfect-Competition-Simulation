# province.py
from dataclasses import dataclass, field
from typing import TypedDict, List, Dict

import core.goods as gds
from core.population import Population
from core.firm import Firm

@dataclass
class Province:
    name: str
    pop_size: int
    income_pc: float    # income per 100 people
    firm_weight: float  # relative weight for seeding entrants and initial firms
    # Optional resource pools (only used for goods in RAW_GOODS)
    resources: Dict[str, int] = field(default_factory=dict)
    firms: list[Firm] = field(default_factory=list)
    rights_given: Dict[gds.GoodID, float] = field(default_factory=lambda: {g: 0.0 for g in gds.GOODS})
    population: Population | None = None

    def attach_population(self):
        self.population = Population(
            size=self.pop_size,
            income_pc=self.income_pc
        )


def normalized_weights(specs: List[Province]) -> Dict[str, float]:
    total = sum(p.firm_weight for p in specs) or 1.0 
    return {p.name: (p.firm_weight / total) for p in specs}

PROVINCES: List[Province] = [
    Province(name="New York",    pop_size=2500, income_pc=1800.0, firm_weight=1.4, resources={"grain": 5000, "wood": 9000}),
    Province(name="Los Angeles", pop_size=2500, income_pc=2700.0, firm_weight=1.1, resources={"grain": 5000, 'wood': 8000, "iron": 8000}),
    Province(name="Chicago",     pop_size=3200,  income_pc=2000.0, firm_weight=0.9, resources={"grain": 5000, 'iron': 9000}),
]