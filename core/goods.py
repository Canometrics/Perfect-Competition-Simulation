from typing import List, Dict, TypedDict
from enum import Enum

GoodID = str
Goods = List[GoodID]

GOODS: Goods = ['consgoods', 'iron', 'wood', 'grain', 'food']

# use set here instead of list to get O(1) instead of O(n)
RAW_GOODS = {'iron', 'wood', 'grain'}
PROCESSED_GOODS = {'consgoods', 'food'}

def is_raw(good:GoodID) -> bool:
    return good in RAW_GOODS

class TierNeeds(TypedDict):
    life: float
    everyday: float
    luxury: float

INITIAL_PRICES: Dict[GoodID, float] = {
    'consgoods': 5.0,
    'iron': 2.0,
    'wood': 3.0,
    'grain': 5.0,
    'food': 5.0
}

def initial_price(good: GoodID) -> float:
    """Return the initial price for a given good."""
    return INITIAL_PRICES[good]

# only define what goods will be demanded by the population here
DEFINED_NEEDS_PER_GOOD: Dict[GoodID, TierNeeds] = { # NEEDS PER 100 POPULATION
    'consgoods' : {'life': 30, 'everyday': 50, 'luxury': 100},
    'food' : {'life': 50, 'everyday': 20, "luxury": 30}
}

def _zero_needs() -> TierNeeds:
    return {'life': 0, 'everyday': 0, 'luxury': 0}


# fill in goods for which needs are not defined as 0 need
NEEDS_PER_GOOD: Dict[GoodID, TierNeeds] = {
    g: DEFINED_NEEDS_PER_GOOD.get(g, _zero_needs())
    for g in GOODS
}


PRODUCTION_RECIPES = {
    'consgoods': {
        'inputs': {'iron': 1, 'wood': 1},  # 2 iron -> 1 consgoods
        'labor_intensity': 0.6 # employees per output
    },
    'iron': {
        'inputs': {},  # no inputs - extracted from provinces
        'labor_intensity': 0.08
    },
    'wood': {
        'inputs': {},
        'labor_intensity': 0.08
    },
    'grain': {
        'inputs': {},
        'labor_intensity': 0.08
    },
    'food': {
        'inputs': {'grain': 1},
        'labor_intensity': 0.08
    }
}