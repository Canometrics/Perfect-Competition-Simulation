GoodID = str # alias to communicate we are referring to this only i guess. may take out later
Goods = list[GoodID]

def is_raw(good:GoodID) -> bool:
    return good in RAW_GOODS

def initial_price(good: GoodID) -> float:
    """
    Return the initial price for a given good.
    """
    return INITIAL_PRICES[good]

GOODS: Goods = ['consgoods', 'iron', 'wood', 'grain', 'food']
RAW_GOODS = {'iron', 'wood', 'grain'} # use set here instead of list to get O(1) instead of O(n)
PROCESSED_GOODS = {'consgoods', 'food'}

INITIAL_PRICES: dict[GoodID, float] = {
    'consgoods': 5.0,
    'iron': 2.0,
    'wood': 3.0,
    'grain': 5.0,
    'food': 5.0
}

COBB_DOUGLAS_WEIGHTS: dict[str, float] = {
    'consgoods': 0.6,
    'food':      0.4,
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