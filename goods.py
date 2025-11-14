from typing import List, Dict, TypedDict
from enum import Enum

GoodID = str
Goods = List[GoodID]

GOODS: Goods = ['bread', 'grain']

# use set here instead of list to get O(1) instead of O(n)
RAW_GOODS = {'grain'}
PROCESSED_GOODS = {'bread'}

def is_raw(good:GoodID) -> bool:
    return good in RAW_GOODS

class TierNeeds(TypedDict):
    life: float
    everyday: float
    luxury: float

# NEEDS PER 100 POPULATION
NEEDS_PER_GOOD: Dict[GoodID, TierNeeds] = {
    'bread' : {'life': 30, 'everyday': 50, 'luxury': 100},
    'grain' : {'life': 30, 'everyday': 50, 'luxury': 100},

}

PRODUCTION_RECIPES = {
    'bread': {
        'inputs': {'grain': 2},  # 2 grain -> 1 bread
        # 'labor_intensity': 1.2
    },
    'grain': {
        'inputs': {},  # no inputs - extracted from provinces
        # 'labor_intensity': 0.8
    }
}