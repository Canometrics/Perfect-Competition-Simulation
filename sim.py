from typing import Tuple, List, Dict
import numpy as np
import pandas as pd
import math

import goods as gds
import config as cfg
from population import Population
from firm import Firm
from market import Market

def simulate_multi(T: int | None = None, p0: float | None = None) -> Tuple[pd.DataFrame, List[Firm]]:
    T  = cfg.T if T is None else T
    p0 = cfg.p0 if p0 is None else p0

    goods = gds.GOODS

    pop = Population(cfg.POP_SIZE, cfg.INCOME_PC)

    # RNGs
    rng_init  = np.random.default_rng(cfg.SEED)
    rng_entry = np.random.default_rng(cfg.SEED + 1)

    # Create markets per good and seed firms
    markets: Dict[gds.GoodID, Market] = {g: Market(good=g, price=p0) for g in goods}

    next_id = 0
    for g in goods:
        next_id = markets[g].seed(rng_init, n_firms=cfg.N_FIRMS, start_id=next_id)

    records: List[Dict] = []

    # MAIN LOOP
    for t in range(T + 1):
        prices = {g: markets[g].price for g in goods}
        demand = pop.demand_for_all_goods(prices)

        for g in goods:
            profit = markets[g].step(
                pop=pop,
                rng_entry=rng_entry,
                tick=t,
                records=records,
                good_label_in_record=(len(goods) > 1),
                demand=demand[g]
            )
            # entry after we have profit computed
            next_id = markets[g]._entry(rng_entry=rng_entry, next_id=next_id, tick_profit=profit)

    # collect all firms across markets
    firms: List[Firm] = [f for m in markets.values() for f in m.firms]

    # Record product identity in firm histories for final reporting
    for f in firms:
        f.history["good"] = f.good

    return pd.DataFrame.from_records(records), firms
