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

# GOODS: Goods = ['consgoods', 'iron', 'wood', 'grain', 'food', 'coal', 'cotton', 'tools', 'clothes']
# RAW_GOODS = {'iron', 'wood', 'grain', 'coal', 'cotton'} # use set here instead of list to get O(1) instead of O(n)
# PROCESSED_GOODS = {'consgoods', 'food', 'tools', 'clothes'}

# INITIAL_PRICES: dict[GoodID, float] = {
#     'consgoods': 5.0,
#     'iron': 2.0,
#     'wood': 3.0,
#     'grain': 5.0,
#     'food': 5.0,
#     'coal': 2.0,
#     'cotton': 3.0,
#     'tools': 6.0,
#     'clothes': 6.0,
# }

# COBB_DOUGLAS_WEIGHTS: dict[str, float] = {
#     'consgoods': 0.4,
#     'food':      0.3,
#     'clothes':   0.2,
#     'tools':     0.1,   # tools are bought by households AND used by clothes makers
# }

# PRODUCTION_RECIPES = {
#     'consgoods': {
#         'inputs': {'iron': 1, 'wood': 1},  # 1 iron + 1 wood -> 1 consgoods
#         'labor_intensity': 0.6 # employees per output
#     },
#     'iron': {
#         'inputs': {},  # no inputs - extracted from provinces
#         'labor_intensity': 0.08
#     },
#     'wood': {
#         'inputs': {},
#         'labor_intensity': 0.08
#     },
#     'grain': {
#         'inputs': {},
#         'labor_intensity': 0.08
#     },
#     'food': {
#         'inputs': {'grain': 1},
#         'labor_intensity': 0.08
#     },
#     'coal': {
#         'inputs': {},
#         'labor_intensity': 0.08
#     },
#     'cotton': {
#         'inputs': {},
#         'labor_intensity': 0.08
#     },
#     'tools': {
#         'inputs': {'iron': 1, 'coal': 1},
#         'labor_intensity': 0.3
#     },
#     'clothes': {
#         'inputs': {'cotton': 1, 'tools': 0.25},  # tools wear out: 1 tool per 4 clothes
#         'labor_intensity': 0.2
#     },
# }