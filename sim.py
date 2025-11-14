from typing import Tuple, List, Dict
import numpy as np
import pandas as pd
import math

import goods as gds
import config as cfg
from firm import Firm
import province as prov
from country import Country


def simulate_multi(T: int | None = None, p0: float | None = None) -> Tuple[pd.DataFrame, List[Firm], pd.DataFrame]:
    T  = cfg.T if T is None else T
    p0 = cfg.p0 if p0 is None else p0

    goods = gds.GOODS

    # Build Country (contains provinces -> Populations and Markets)
    specs = prov.PROVINCES
    country = Country.from_specs(specs)

    # map for province objects by name, used when seeding / entry assigns provinces
    province_map: Dict[str, prov.Province] = {p.name: p for p in specs}
    # {
    #   "New York": <Province object>,
    #   "Los Angeles": <Province object>,
    #   "Chicago": <Province object>,
    # }

    # RNGs
    rng_init  = np.random.default_rng(cfg.SEED)
    rng_entry = np.random.default_rng(cfg.SEED + 1)

    # Seed initial firms into the country's markets
    next_id = country.seed_markets(
        rng_init=rng_init,
        province_map=province_map,
        n_firms=cfg.N_FIRMS,
        start_id=0,
    )

    records: List[Dict] = []
    prov_records: List[Dict] = []  # collect per-province panel rows

    # MAIN LOOP
    for t in range(T + 1):
        # current national prices per good from the country's markets
        prices: Dict[gds.GoodID, float] = {
            g: country.markets[g].price for g in goods
        }

        # 1) per province demand for all goods
        demand_by_prov: Dict[str, Dict[gds.GoodID, int]] = {
            pname: country.provinces[pname].demand_for_all_goods(prices)
            for pname in country.provinces.keys()
        }

        # 2) national demand per good (sum over provinces)
        demand_nat: Dict[gds.GoodID, int] = {
            g: sum(demand_by_prov[p][g] for p in demand_by_prov) for g in goods
        }

        # 3) step country's markets and capture realized totals per good
        realized_nat: Dict[gds.GoodID, int] = {}

        for g in goods:
            market = country.markets[g]

            profit = market.step(
                pop=country,                        # Country provides needs_per_good
                rng_entry=rng_entry,
                tick=t,
                records=records,
                good_label_in_record=(len(goods) > 1),
                demand=demand_nat[g],
            )

            # pull data directly from record we just appended
            last = records[-1]
            realized_nat[g] = last["q_realized"]
            active_firms = last["active_firms"]

            next_id = market._entry(
                rng_entry=rng_entry,
                next_id=next_id,
                tick_profit=profit,
                active_firms=active_firms,
                provinces=province_map,   # pass Province objects
            )

        # 4) allocate realized to provinces by demand share, with exact reconciliation
        prov_names = list(country.provinces.keys())
        for g in goods:
            d_nat = demand_nat[g]
            if d_nat <= 0:
                # no national demand: everyone gets zero realized
                for pname in prov_names:
                    prov_records.append({
                        "tick": t,
                        "province": pname,
                        "good": g,
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
    firms: List[Firm] = [f for m in country.markets.values() for f in m.firms]
    df_province = pd.DataFrame.from_records(prov_records)

    # annotate firm histories
    for f in firms:
        f.history["good"] = f.good
        # province is on the Firm; you can add it into snapshots when you display

    df_market = pd.DataFrame.from_records(records)
    return df_market, firms, df_province
