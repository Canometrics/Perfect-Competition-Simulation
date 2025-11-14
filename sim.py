from typing import Tuple, List, Dict
import numpy as np
import pandas as pd
import math

import goods as gds
import config as cfg
from firm import Firm
from market import Market
import province as prov
from country import Country

def simulate_multi(T: int | None = None, p0: float | None = None) -> Tuple[pd.DataFrame, List[Firm]]:
    T  = cfg.T if T is None else T
    p0 = cfg.p0 if p0 is None else p0

    goods = gds.GOODS

    # Build Country (contains provinces -> Populations)
    specs = prov.PROVINCES
    country = Country.from_specs(specs)
    province_map: Dict[str, prov.Province] = {p.name: p for p in specs}
    # this is to easily refer to provinces by name in the future, gives:
    # {
    # "New York": <Province object>,
    # "Los Angeles": <Province object>,
    # "Chicago": <Province object>,
    # }


    # RNGs
    rng_init  = np.random.default_rng(cfg.SEED)
    rng_entry = np.random.default_rng(cfg.SEED + 1)

    # NATIONAL markets (one per good), but with province weights for firm placement
    markets: Dict[gds.GoodID, Market] = {
        g: Market(good=g, price=p0, province_weights=country.weights) for g in goods
    }

    # Seed initial firms, distributed over provinces by weights
    next_id = 0
    for g in goods:
        next_id = markets[g].seed(
            rng_init,
            n_firms=cfg.N_FIRMS,
            start_id=next_id,
            provinces=province_map,   # use Province objects
        )
    records: List[Dict] = []

    prov_records: List[Dict] = []  # collect per-province panel rows

    # MAIN LOOP
    for t in range(T + 1):
        # current national prices per good
        prices: Dict[gds.GoodID, float] = {g: markets[g].price for g in goods}

        # 1) per province demand for all goods
        demand_by_prov: Dict[str, Dict[gds.GoodID, int]] = {
            pname: country.provinces[pname].demand_for_all_goods(prices)
            for pname in country.provinces.keys()
        }

        # 2) national demand per good (sum over provinces)
        demand_nat: Dict[gds.GoodID, int] = {
            g: sum(demand_by_prov[p][g] for p in demand_by_prov) for g in goods
        }

        # 3) step national markets and capture realized totals per good
        realized_nat: Dict[gds.GoodID, int] = {}
        for g in goods:
            profit = markets[g].step(
                pop=country,
                rng_entry=rng_entry,
                tick=t,
                records=records,
                good_label_in_record=(len(goods) > 1),
                demand=demand_nat[g],
            )

            # pull data directly from record we just appended
            last = records[-1]
            realized_nat[g] = last["q_realized"]
            active_firms = last["active_firms"]   # <-- HERE

            next_id = markets[g]._entry(
                rng_entry=rng_entry,
                next_id=next_id,
                tick_profit=profit,
                active_firms=active_firms,
                provinces=province_map,   # use Province objects
            )


        # 4) allocate realized to provinces by demand share, with exact reconciliation
        prov_names = list(country.provinces.keys())
        for g in goods:
            d_nat = demand_nat[g]
            if d_nat <= 0:
                # no national demand: everyone gets zero realized
                for pname in prov_names:
                    prov_records.append({
                        "tick": t, "province": pname, "good": g,
                        "q_demand": int(demand_by_prov[pname][g]),
                        "q_realized": 0,
                    })
                continue

            running_sum = 0
            alloc_rows: List[Tuple[str, int, int]] = []
            for i, pname in enumerate(prov_names):
                d_p = int(demand_by_prov[pname][g])
                if i < len(prov_names) - 1:
                    share = d_p / d_nat
                    q_real_p = int(round(share * realized_nat[g]))
                    running_sum += q_real_p
                else:
                    # reconcile last province so totals match exactly
                    q_real_p = int(realized_nat[g] - running_sum)
                alloc_rows.append((pname, d_p, q_real_p))

            for pname, d_p, q_real_p in alloc_rows:
                prov_records.append({
                    "tick": t,
                    "province": pname,
                    "good": g,
                    "q_demand": d_p,
                    "q_realized": q_real_p,
                })

    # collect firms and build df_province
    firms: List[Firm] = [f for m in markets.values() for f in m.firms]
    df_province = pd.DataFrame.from_records(prov_records)

    # annotate firm histories
    for f in firms:
        f.history["good"] = f.good
        # province is on the Firm; you can add it into snapshots when you display

    return pd.DataFrame.from_records(records), firms, df_province