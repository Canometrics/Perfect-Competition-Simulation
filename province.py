from dataclasses import dataclass, field
from typing import TypedDict, List, Dict

@dataclass
class Province:
    name: str
    pop_size: int            # population size for this province
    income_pc: float         # income per 100 people per tick
    firm_weight: float       # relative weight for seeding entrant/initial firms (normalized later)

def normalized_weights(specs: List[Province]) -> Dict[str, float]:
    total = sum(p["firm_weight"] for p in specs) or 1.0
    return {p["name"]: (p["firm_weight"] / total) for p in specs}

PROVINCES: list = [
    # Name, population size, income per 100 people, firm weight
    {"name": "New York",    "pop_size": 1900, "income_pc": 1500.0, "firm_weight": 1.4},
    {"name": "Los Angeles", "pop_size": 1300, "income_pc": 1300.0, "firm_weight": 1.1},
    {"name": "Chicago",     "pop_size":  900, "income_pc": 1200.0, "firm_weight": 0.9},
]