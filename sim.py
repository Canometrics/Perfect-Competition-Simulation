# sim.py
from typing import Tuple, List, Dict
import numpy as np
import pandas as pd
import math

import goods as gds
import config as cfg
from population import Population
from firm import Firm, spawn_firms


def simulate_multi(T: int | None = None, p0: float | None = None) -> Tuple[pd.DataFrame, List[Firm]]:
    # Use config defaults if not passed in
    T  = cfg.T if T is None else T
    p0 = cfg.p0 if p0 is None else p0

    # Prices per good (ready for multi-good; currently single-good)
    prices: Dict[gds.GoodID, float] = {g: p0 for g in gds.GOODS}

    # Population
    pop = Population(cfg.POP_SIZE, cfg.INCOME_PC)

    # RNGs
    rng_init  = np.random.default_rng(cfg.SEED)
    rng_entry = np.random.default_rng(cfg.SEED + 1)

    # --- Seed firms for ALL goods, with unique IDs across goods
    firms: List[Firm] = []
    next_id = 0
    for g in gds.GOODS:
        initial_firms = spawn_firms(
            good=g,
            rng=rng_init,
            n=cfg.N_FIRMS,
            start_id=next_id
        )
        firms.extend(initial_firms)
        next_id += len(initial_firms)

    profit_hist: list[float] = []
    records: List[Dict] = []

    # ========== MAIN LOOP ==========
    for t in range(T + 1):
        tick_profit_total = 0.0
        tick_TR_total = 0.0
        tick_TC_total = 0.0

        # Per-good bookkeeping this tick
        per_good_sales: Dict[int, float] = {f.id: 0.0 for f in firms}

        for g in gds.GOODS:
            p = prices[g]

            # 1) Demand for this good
            q_demand, tier_afford = pop.target_qty_per_good(p, good=g)

            # 2) Firms update quantities (only those producing this good)
            for f in firms:
                if f.good == g:
                    f.update_quantity(p, tick=t)

            # 3) Supply for this good
            q_supply = int(sum(f.q for f in firms if f.good == g))

            # 4) Realized purchases with supply constraint
            q_bought, tier_realized, tiers_bought = pop.realized_qty(p, q_supply, good=g)

            # 5) Allocate sales proportionally among firms producing g
            if q_supply > 0:
                for f in firms:
                    if f.good != g:
                        continue
                    share = f.q / q_supply
                    per_good_sales[f.id] += min(f.q, share * q_bought)

            # We'll compute finance after allocating all goods,
            # but record per-good market stats now.
            records.append({
                "tick": t,
                "good": g,
                "price": p,
                "q_demand": q_demand,
                "q_realized": q_bought,
                "q_supply": q_supply,
                "tier": tier_afford,
                "tier_realized": tier_realized,
                "life": tiers_bought["life"],
                "everyday": tiers_bought["everyday"],
                "luxury": tiers_bought["luxury"],
                # filled after finance booking below if you want per-good totals; kept 0.0 here
                "revenue_total": 0.0,
                "cost_total": 0.0,
                "profit_total": 0.0,
                "active_firms": sum(1 for f in firms if f.active and f.good == g),
            })

        # 6) Finance booking for all firms (using per_good_sales)
        #    Also accumulate per-good totals back into the last record for each good.
        #    We'll track last index per good to fill in totals.
        last_idx_for_good: Dict[gds.GoodID, int] = {}
        for idx in range(len(records) - 1, -1, -1):
            g = records[idx]["good"]
            if g not in last_idx_for_good:
                last_idx_for_good[g] = idx
            if len(last_idx_for_good) == len(gds.GOODS):
                break

        per_good_totals = {g: {"TR": 0.0, "TC": 0.0, "PROF": 0.0} for g in gds.GOODS}

        for f in firms:
            # Use this tick's price for the firm's good
            p = prices[f.good]
            sales_i = per_good_sales.get(f.id, 0.0) if f.active else 0.0
            TR_i, TC_i, PROF_i = f.book_finance(p, sales_i)

            per_good_totals[f.good]["TR"]   += TR_i
            per_good_totals[f.good]["TC"]   += TC_i
            per_good_totals[f.good]["PROF"] += PROF_i

            tick_TR_total += TR_i
            tick_TC_total += TC_i
            tick_profit_total += PROF_i

        # Fill totals back into each good's last record this tick
        for g, rec_idx in last_idx_for_good.items():
            rec = records[rec_idx]
            rec["revenue_total"] = per_good_totals[g]["TR"]
            rec["cost_total"]    = per_good_totals[g]["TC"]
            rec["profit_total"]  = per_good_totals[g]["PROF"]

        # 7) Entry logic (based on aggregate profitability)
        profit_hist.append(tick_profit_total)
        if len(profit_hist) > cfg.ENTRY_WINDOW:
            profit_hist = profit_hist[-cfg.ENTRY_WINDOW:]

        # Remove inactive firms
        if any(not f.active for f in firms):
            firms = [f for f in firms if f.active]

        active_now = sum(1 for f in firms if f.active) or 1
        avg_profit_per_firm = (np.mean(profit_hist) / active_now) if profit_hist else 0.0
        profit_pos = max(avg_profit_per_firm, 0.0)
        p_entry = 1.0 - math.exp(-cfg.ENTRY_ALPHA * profit_pos)

        # Add entrants; assign them to random goods (uniform). Works the same with 1 good.
        for _ in range(cfg.ENTRY_MAX_PER_TICK):
            if rng_entry.random() < p_entry:
                g_choice = rng_entry.choice(gds.GOODS)
                entrant = spawn_firms(
                    good=g_choice,
                    rng=rng_entry,
                    n=1,
                    start_id=next_id
                )[0]
                firms.append(entrant)
                next_id += 1

        # 8) Price update per good (tatonnement + smoothing)
        #    Needs each good's excess; recompute demand/supply quickly (OK with 1 good).
        for g in gds.GOODS:
            # To avoid recomputing demand incorrectly with multi-goods budget interactions,
            # we reuse the market-clearing approximation seen above. In single-good runs,
            # this is identical to your prior behavior.
            p = prices[g]
            q_demand, _ = pop.target_qty_per_good(p, good=g)
            q_supply = int(sum(f.q for f in firms if f.good == g))

            excess = q_demand - q_supply
            p_target = max(0.01, p + cfg.tatonnement_speed * excess)
            alpha = max(0.0, min(1.0, float(cfg.price_alpha)))
            prices[g] = max(0.01, (1 - alpha) * p + alpha * p_target)

        # 9) Shocks (capacity / MC) apply to all firms
        if cfg.USE_CAPACITY_SHOCK and t == cfg.SHOCK_TICK:
            for f in firms:
                if f.base_capacity is None:
                    f.base_capacity = float(f.capacity)
                f.capacity = int(f.base_capacity * cfg.CAP_MULT_DURING_SHOCK)

        if cfg.USE_CAPACITY_SHOCK and cfg.SHOCK_DURATION > 0 and t == cfg.SHOCK_TICK + cfg.SHOCK_DURATION:
            for f in firms:
                if f.base_capacity is not None:
                    f.capacity = int(f.base_capacity)

        if cfg.USE_MC_SHOCK and t == cfg.SHOCK_TICK:
            for f in firms:
                if f.base_MC is None:
                    f.base_MC = float(f.MC)
                f.MC = float(f.base_MC * cfg.MC_MULT_DURING_SHOCK)

        if cfg.USE_MC_SHOCK and cfg.SHOCK_DURATION > 0 and t == cfg.SHOCK_TICK + cfg.SHOCK_DURATION:
            for f in firms:
                if f.base_MC is not None:
                    f.MC = float(f.base_MC)

    return pd.DataFrame.from_records(records), firms
