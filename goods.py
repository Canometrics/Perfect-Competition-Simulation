from typing import List, Dict, TypedDict

GoodID = str
Goods = List[GoodID]

GOODS: Goods = ['bread']

class TierNeeds(TypedDict):
    life: float
    everyday: float
    luxury: float

# NEEDS PER 100 POPULATION
NEEDS_PER_GOOD: Dict[GoodID, TierNeeds] = {
    'bread' : {'life': 30, 'everyday': 50, 'luxury': 100}
}