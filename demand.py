
import math
from typing import Tuple

from population import Population

def realized_tier_labels(price: float, available_q: float, pop: Population):
    p = max(0.01, price)
    B = pop.budget
    can_buy = math.ceil(min(B / p, available_q))

    q_life  = math.ceil(pop.NEED_LIFE  * pop.size / 100.0)
    q_every = math.ceil(pop.NEED_EVERY * pop.size / 100.0)
    q_lux   = math.ceil(pop.NEED_LUX   * pop.size / 100.0)

    tiers = {"life": 0, "everyday": 0, "luxury": 0}

    tiers["life"] = min(q_life, can_buy); can_buy -= tiers["life"]
    if tiers["life"] < q_life:
        return sum(tiers.values()), "life_partial", tiers

    tiers["everyday"] = min(q_every, can_buy); can_buy -= tiers["everyday"]
    if tiers["everyday"] < q_every:
        return sum(tiers.values()), "life", tiers

    tiers["luxury"] = min(q_lux, can_buy)
    label = "luxury" if tiers["luxury"] >= q_lux else "everyday"
    return sum(tiers.values()), label, tiers
