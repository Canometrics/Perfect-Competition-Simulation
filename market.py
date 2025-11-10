from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from collections import deque
import math
import numpy as np

import config as cfg
import goods as gds
from population import Population
from firm import Firm, spawn_firms


@dataclass
class Market:
    """
    A per-good market that encapsulates:
      - price
      - demand calculation
      - firm quantity updates
      - supply aggregation
      - proportional sales allocation
      - finance booking
      - entry/exit and shocks
      - per-tick record generation
    """
    good: gds.GoodID
    price: float
    firms: List[Firm] = field(default_factory=list)

    # Internal state
    profit_hist: deque[float] = field(default_factory=lambda: deque(maxlen=cfg.ENTRY_WINDOW))

    def seed(self, rng_init: np.random.Generator, n_firms: int, start_id: int) -> int:
        batch = spawn_firms(good=self.good, rng=rng_init, n=n_firms, start_id=start_id)
        self.firms.extend(batch)
        return start_id + len(batch)

    #legacy
    # def _demand(self, pop: Population) -> Tuple[int, str]:
    #     # 1) Aggregating demand
    #     q_demand, tier_afford = pop.qty_demand_per_good(self.price, good=self.good)
    #     return q_demand, tier_afford

    def _update_firms(self, tick: int) -> None:
        # 2) Firms update quantities
        for f in self.firms:
            f.update_quantity(self.price, tick=tick)

    def _supply(self) -> int:
        # 3) Aggregating supply
        return int(sum(f.q for f in self.firms))

    def _hhi(self, sales_by_firm: Dict[int, float], fallback_supply: int) -> float:
        """
        HHI on 0–10000 scale using sales shares this tick.
        If no sales happened, fallback to shares of supply (q) to avoid NaN.
        """
        # try based on sales
        total = sum(sales_by_firm.values())
        if total <= 0:
            # fallback: use supply shares this tick
            if fallback_supply <= 0:
                return 0.0
            shares = [(f.q / fallback_supply) for f in self.firms if f.q > 0 and f.active]
        else:
            shares = [(v / total) for v in sales_by_firm.values() if v > 0]

        if not shares:
            return 0.0

        # standard HHI in "points": sum of squared percentage shares
        # (e.g., monopoly -> 100^2 = 10,000)
        return float(sum((100.0 * s) ** 2 for s in shares))

    def _clear_market(
        self, desired_q: int, q_supply: int, needs: tuple[int, int, int]
    ) -> tuple[int, str, dict[str, int], dict[int, float]]:
        # quantity actually bought
        q_bought = min(int(desired_q), int(q_supply))

        # derive tier caps from needs
        q_life, q_every, q_lux = needs
        inc_every = max(0, q_every - q_life)
        inc_lux   = max(0, q_lux   - q_every)

        # allocate bought units into tiers
        life_bought = min(q_bought, q_life)
        rem = q_bought - life_bought
        every_bought = min(rem, inc_every)
        rem -= every_bought
        lux_bought = min(rem, inc_lux)

        tiers_bought = {
            "life": life_bought,
            "everyday": every_bought,
            "luxury": lux_bought,
        }

        # label of highest fully reached tier
        if life_bought < q_life:
            tier_realized = "life_partial"
        elif every_bought < inc_every:
            tier_realized = "life"
        elif lux_bought < inc_lux:
            tier_realized = "everyday"
        else:
            tier_realized = "luxury"

        # proportional allocation of sales to firms
        sales_by_firm: Dict[int, float] = {f.id: 0.0 for f in self.firms}
        if q_supply > 0:
            for f in self.firms:
                share = (f.q / q_supply)
                sales_by_firm[f.id] = min(f.q, share * q_bought)

        return q_bought, tier_realized, tiers_bought, sales_by_firm
    
        # add inventory for unsold goods

    def _book_finance(self, sales_by_firm: Dict[int, float]) -> Tuple[float, float, float]:
        # 6) With sales by firm, calculate finances
        TR_total = TC_total = Profit_total = 0.0
        for f in self.firms:
            sales_i = sales_by_firm.get(f.id, 0.0) if f.active else 0.0
            TR_i, TC_i, PROF_i = f.book_finance(self.price, sales_i)
            TR_total += TR_i
            TC_total += TC_i
            Profit_total += PROF_i
        return TR_total, TC_total, Profit_total

    def _remove_inactive(self) -> None:
        # Bookkeeping: Remove inactive firms
        if any(not f.active for f in self.firms):
            self.firms = [f for f in self.firms if f.active]

    def _entry(self, rng_entry: np.random.Generator, next_id: int, tick_profit: float) -> int:
        # Bookkeping: Update profit values for new firm entry window, remove old ones (those before the profit window)
        self.profit_hist.append(tick_profit)

        # Bookkeeping: track market
        active_now = sum(1 for f in self.firms if f.active) or 1
        avg_profit_per_firm = (np.mean(self.profit_hist) / active_now) if self.profit_hist else 0.0
        profit_pos = max(avg_profit_per_firm, 0.0)
        p_entry = 1.0 - math.exp(-cfg.ENTRY_ALPHA * profit_pos)  # probability of entry at current profit levels

        # calculate market entrants
        for _ in range(cfg.ENTRY_MAX_PER_TICK):
            if rng_entry.random() < p_entry:  # introduce randomness into how many firms enter
                entrant = spawn_firms(good=self.good, rng=rng_entry, n=1, start_id=next_id)[0]  # fn returns list so get 1st entry
                self.firms.append(entrant)
                next_id += 1

        return next_id

    def _price_update1(self, q_demand: int, q_supply: int) -> None:
        # 8) Price update with inertia (tatonnement + smoothing), sets price for next tick, since all other actions are done
        excess = q_demand - q_supply
        p_target = max(0.01, self.price + cfg.tatonnement_speed * excess)
        self.price = max(0.01, (1 - cfg.price_alpha) * self.price + cfg.price_alpha * p_target)

    def _price_update(self, q_demand: int, q_supply: int) -> None:
        # Percentage excess: how much demand exceeds supply relative to supply level
        excess_pct = (q_demand - q_supply) / max(1, q_supply)
        excess_pct = max(-0.9, min(2.0, excess_pct))

        # Tatonnement target based on *percentage* imbalance
        p_target = max(0.01, self.price * (1 +  cfg.tatonnement_speed * excess_pct))

        # Exponential smoothing towards target
        self.price = max(0.01, (1 - cfg.price_alpha) * self.price + cfg.price_alpha * p_target)


    def _apply_shocks(self, tick: int) -> None:
        # 9) Shocks
        if cfg.USE_CAPACITY_SHOCK and tick == cfg.SHOCK_TICK:
            for f in self.firms:
                if f.base_capacity is None:
                    f.base_capacity = float(f.capacity)
                f.capacity = int(f.base_capacity * cfg.CAP_MULT_DURING_SHOCK)

        if cfg.USE_CAPACITY_SHOCK and cfg.SHOCK_DURATION > 0 and tick == cfg.SHOCK_TICK + cfg.SHOCK_DURATION:
            for f in self.firms:
                if f.base_capacity is not None:
                    f.capacity = int(f.base_capacity)

        if cfg.USE_MC_SHOCK and tick == cfg.SHOCK_TICK:
            for f in self.firms:
                if f.base_MC is None:
                    f.base_MC = float(f.MC)
                f.MC = float(f.base_MC * cfg.MC_MULT_DURING_SHOCK)

        if cfg.USE_MC_SHOCK and cfg.SHOCK_DURATION > 0 and tick == cfg.SHOCK_TICK + cfg.SHOCK_DURATION:
            for f in self.firms:
                if f.base_MC is not None:
                    f.MC = float(f.base_MC)

    def step(self, pop: Population, demand: int, rng_entry: np.random.Generator,
            tick: int, records: List[Dict], good_label_in_record: bool = False) -> int:

        q_demand = int(demand)
        self._update_firms(tick)
        q_supply = self._supply()

        needs = pop.needs_per_good(self.good)  # (life, everyday, luxury)
        q_bought, tier_realized, tiers_bought, sales_by_firm = self._clear_market(q_demand, q_supply, needs)
        TR_total, TC_total, Profit_total = self._book_finance(sales_by_firm)
        self._remove_inactive()

        hhi = self._hhi(sales_by_firm, q_supply)

        rec = {
            "tick": tick,
            "price": self.price,
            "q_demand": q_demand,
            "q_realized": q_bought,
            "q_supply": q_supply,
            "tier_realized": tier_realized,
            "life": tiers_bought["life"],
            "everyday": tiers_bought["everyday"],
            "luxury": tiers_bought["luxury"],
            "revenue_total": TR_total,
            "cost_total": TC_total,
            "profit_total": Profit_total,
            "hhi": hhi,
            "active_firms": sum(1 for f in self.firms if f.active),
        }
        if good_label_in_record:
            rec["good"] = self.good
        records.append(rec)

        self._price_update(q_demand, q_supply)
        self._apply_shocks(tick)
        return Profit_total
