from typing import Tuple, List, Dict
import numpy as np
import pandas as pd
import copy

import goods as gds
import config as cfg
from firm import Firm
import province as prov
from country import Country

def simulate_multi(T: int | None = None, p0: float | None = None) -> Tuple[pd.DataFrame, List[Firm], pd.DataFrame]:
    T = cfg.T if T is None else T

    goods = gds.GOODS

    # Build Country (contains provinces -> Populations and Markets)
    specs = [copy.deepcopy(p) for p in prov.PROVINCES] # to avoid reusing altered states from previous simulation runs
    country = Country.from_specs(specs)

    # map for province objects by name, used when seeding / entry assigns provinces
    province_map: Dict[str, prov.Province] = {p.name: p for p in specs}

    # RNGs
    rng_init = np.random.default_rng(cfg.SEED)
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

        # 0) firms update their MC based on input prices and wages
        for g in goods:
            market = country.markets[g]
            for f in market.firms:
                f.update_input_cost(prices, wage=cfg.WAGE)

        province_names = list(country.provinces.keys())
        goods = gds.GOODS

        # 1) national firm input-demand per good (based on planned output this tick)
        input_demand_nat: Dict[gds.GoodID, float] = {g: 0.0 for g in goods}

        for g_out in goods:
            market = country.markets[g_out]
            for firm in market.firms:
                # only firms with input requirements (manufacturing) create input demand
                if not getattr(firm, "input_requirements", None):
                    continue

                # how much output is this firm planning to produce at current price?
                q_plan = firm.plan_quantity(price=market.price)
                q_feasible = firm.hire_and_fire(q_plan, hypothetical=True)   # labor constrained

                for g_in, units in firm.input_requirements.items():
                    input_demand_nat[g_in] += q_feasible  * units

        # 2) per province consumer demand for all goods (no firm demand here)
        demand_by_prov: Dict[str, Dict[gds.GoodID, int]] = {}

        for pname in province_names:
            prov_obj = country.provinces[pname]
            cons_d = prov_obj.population.demand_for_all_goods(prices)   # <- key change
            total_for_p: Dict[gds.GoodID, int] = {}
            for g in goods:
                q_cons = float(cons_d.get(g, 0))
                total_for_p[g] = int(q_cons)
            demand_by_prov[pname] = total_for_p

        # 3) national consumer demand per good (sum over provinces)
        consumer_nat: Dict[gds.GoodID, int] = {
            g: sum(demand_by_prov[p][g] for p in demand_by_prov) for g in goods
        }

        # 4) step country's markets and capture realized totals per good using separate consumer vs firm demand
        realized_nat: Dict[gds.GoodID, int] = {}

        for g in goods:
            market = country.markets[g]

        for g in goods:
            market = country.markets[g]

            profit = market.step(
                pop=country,                        # Country provides needs_per_good
                q_consumer=consumer_nat[g],
                q_firm=input_demand_nat[g],
                rng_entry=rng_entry,
                tick=t,
                records=records,
                good_label_in_record=(len(goods) > 1),
            )

            # pull data directly from record we just appended
            last = records[-1]
            realized_nat[g] = last["q_realized"]
            active_firms = last["active_firms"]

            # NEW: national employment level this tick
            total_employed = 0
            for prov_obj in country.provinces.values():
                pop_obj = prov_obj.population
                if pop_obj is not None:
                    total_employed += int(getattr(pop_obj, "number_employed", 0))
            last["employment_total"] = int(total_employed)

            next_id = market._entry(
                rng_entry=rng_entry,
                next_id=next_id,
                tick_profit=profit,
                active_firms=active_firms,
                provinces=province_map,   # pass Province objects
            )


        # 6) allocate realized to provinces by consumer demand share, with exact reconciliation
        prov_names = list(country.provinces.keys())
        for g in goods:
            d_nat_cons = consumer_nat[g]
            if d_nat_cons <= 0:
                # no consumer demand: everyone gets zero realized in province records
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
                d_p = int(demand_by_prov[pname][g])  # consumer-only demand
                if i < len(prov_names) - 1:
                    share = d_p / d_nat_cons
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
