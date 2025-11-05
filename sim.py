from typing import Tuple, List, Dict
import numpy as np
import pandas as pd
import math

import config as cfg
from population import Population
from firm import Firm
from demand import realized_tier_labels
from spawn import spawn_firms


def simulate_multi(T: int | None = None, p0: float | None = None) -> Tuple[pd.DataFrame, List[Firm]]:
    # pull live values from cfg unless explicitly overridden
    if T  is None: T  = cfg.T
    if p0 is None: p0 = cfg.p0

    pop = Population(cfg.POP_SIZE, cfg.INCOME_PC)
    p = p0

    # RNGs
    rng_init  = np.random.default_rng(cfg.SEED)
    rng_entry = np.random.default_rng(cfg.SEED + 101)

    firms: List[Firm] = spawn_firms(rng_init, n=cfg.N_FIRMS, start_id=0)
    next_id = len(firms)

    profit_hist: list[float] = []
    records = []

    for t in range(T + 1):
        # 1) Affordable demand
        q_demand, tier_afford = pop.target_qty(p)

        # 2) Firms update quantities
        for f in firms:
            f.update_quantity(p, tick=t)

        # 3) Aggregate supply
        q_supply = int(sum(f.q for f in firms))

        # 4) Realized purchases at price with supply constraint
        q_bought, tier_realized, tiers_bought = realized_tier_labels(p, q_supply, pop)

        # 5) Allocate sales proportionally by supply
        sales_by_firm: Dict[int, float] = {f.id: 0.0 for f in firms}
        if q_supply > 0:
            for f in firms:
                share = (f.q / q_supply) if q_supply > 0 else 0.0
                sales_by_firm[f.id] = min(f.q, share * q_bought)

        # 6) Finance per firm and shutdown updates
        TR_total = TC_total = Profit_total = 0.0
        for f in firms:
            sales_i = sales_by_firm[f.id] if f.active else 0.0
            TR_i, TC_i, PROF_i = f.book_finance(p, sales_i)
            TR_total += TR_i
            TC_total += TC_i
            Profit_total += PROF_i

        # Entry bookkeeping
        profit_hist.append(Profit_total)
        if len(profit_hist) > cfg.ENTRY_WINDOW:
            profit_hist = profit_hist[-cfg.ENTRY_WINDOW:]

        # Remove inactive
        if any(not f.active for f in firms):
            firms = [f for f in firms if f.active]

        active_now = sum(1 for f in firms if f.active) or 1
        avg_profit_per_firm = (np.mean(profit_hist) / active_now) if profit_hist else 0.0
        profit_pos = max(avg_profit_per_firm, 0.0)
        p_entry = 1.0 - math.exp(-cfg.ENTRY_ALPHA * profit_pos)

        for _ in range(cfg.ENTRY_MAX_PER_TICK):
            if rng_entry.random() < p_entry:
                entrant = spawn_firms(rng_entry, n=1, start_id=next_id)[0]
                firms.append(entrant)
                next_id += 1

        # 7) Record
        records.append({
            "tick": t,
            "price": p,
            "q_demand": q_demand,
            "q_realized": q_bought,
            "q_supply": q_supply,
            "tier": tier_afford,
            "tier_realized": tier_realized,
            "life": tiers_bought["life"],
            "everyday": tiers_bought["everyday"],
            "luxury": tiers_bought["luxury"],
            "revenue_total": TR_total,
            "cost_total": TC_total,
            "profit_total": Profit_total,
            "active_firms": sum(1 for f in firms if f.active),
        })

        # 8) Price update with inertia (tatonnement + smoothing)
        excess = q_demand - q_supply
        p_target = max(0.01, p + cfg.tatonnement_speed * excess)
        alpha = max(0.0, min(1.0, float(cfg.price_alpha)))  # clamp
        p = (1 - alpha) * p + alpha * p_target
        p = max(0.01, p)

        # 9) Shocks
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
