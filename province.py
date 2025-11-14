# province.py
from dataclasses import dataclass, field
from typing import TypedDict, List, Dict
import goods as gds

@dataclass
class Province:
    name: str
    pop_size: int
    income_pc: float    # income per 100 people
    firm_weight: float  # relative weight for seeding entrants and initial firms
    # Optional resource pools (only used for goods in RAW_GOODS)
    resources: Dict[str, int] = field(default_factory=dict)
    rights_given: Dict[gds.GoodID, float] = field(default_factory=lambda: {g: 0.0 for g in gds.GOODS})

def normalized_weights(specs: List[Province]) -> Dict[str, float]:
    total = sum(p.firm_weight for p in specs) or 1.0 
    return {p.name: (p.firm_weight / total) for p in specs}

PROVINCES: List[Province] = [
    Province(name="New York",    pop_size=1900, income_pc=1500.0, firm_weight=1.4, resources={"grain": 3000}),
    Province(name="Los Angeles", pop_size=1300, income_pc=1300.0, firm_weight=1.1, resources={"grain": 5000}),
    Province(name="Chicago",     pop_size=900,  income_pc=1200.0, firm_weight=0.9, resources={"grain": 1000}),
]