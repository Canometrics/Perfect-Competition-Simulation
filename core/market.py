from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from collections import deque
import math
import numpy as np

import config.config as cfg
import core.goods as gds
from core.population import Population
from core.firm import Firm, spawn_firms, FirmType
from core.province import Province


@dataclass
class Market:
    good: gds.GoodID
    price: float
    # national market knows how to distribute firms over provinces
    province_weights: Dict[str, float] = field(default_factory=dict)
    firms: List[Firm] = field(default_factory=list)
    profit_hist: deque[float] = field(default_factory=lambda: deque(maxlen=cfg.ENTRY_WINDOW))

    def __post_init__(self):
    # initialize firm type from good type
        if gds.is_raw(self.good):
            self.firm_type = FirmType.RGO
        else:
            self.firm_type = FirmType.Manu


    def _sample_province(self, rng: np.random.Generator) -> str:
        names = list(self.province_weights.keys())
        probs = np.array([self.province_weights[n] for n in names], dtype=float)
        probs = probs / probs.sum()
        return str(rng.choice(names, p=probs))

    def seed(
        self,
        rng_init: np.random.Generator,
        n_firms: int,
        start_id: int,
        provinces: Dict[str, Province],        # <-- keep this
    ) -> int:
        # split initial firms by weights
        names = list(self.province_weights.keys())
        probs = np.array([self.province_weights[n] for n in names], dtype=float)
        probs = probs / probs.sum()

        # multinomial draw for counts per province
        counts = np.random.default_rng(rng_init.integers(0, 2**31 - 1)).multinomial(
            n=n_firms, pvals=probs
        )
        nid = start_id

        for name, count in zip(names, counts):
            if count <= 0:
                continue
            province_obj = provinces[name]
            batch = spawn_firms(
                self.good,
                self.firm_type,
                rng_init,
                n=count,
                start_id=nid,
                province=province_obj,
                max_share=0.9,
            )
            self.firms.extend(batch)
            nid += count

        return nid

    def _entry(
        self,
        rng_entry: np.random.Generator,
        next_id: int,
        tick_profit: float,
        active_firms: int,
        provinces: Dict[str, Province],
    ) -> int:
        self.profit_hist.append(tick_profit)

        active_now = active_firms or 1
        avg_profit_per_firm = (np.mean(self.profit_hist) / active_now) if self.profit_hist else 0.0
        profit_pos = max(avg_profit_per_firm, 0.0)
        p_entry = 1.0 - math.exp(-cfg.ENTRY_ALPHA * profit_pos)

        max_new = max(1, int(active_firms * cfg.ENTRY_MAX_PER_TICK))

        MIN_RGO_SHARE = 0.05      # minimum meaningful share for a new RGO entrant
        CAP_ENTRY = 1.0           # cap for post-initial allocation

        for _ in range(max_new):
            if rng_entry.random() < p_entry:
                pname = self._sample_province(rng_entry)
                province_obj = provinces[pname]

                # For RGO firms, block entry if remaining rights < MIN_RGO_SHARE
                if self.firm_type is FirmType.RGO:
                    used = province_obj.rights_given.get(self.good, 0.0)
                    remaining = max(0.0, CAP_ENTRY - used)

                    if remaining < MIN_RGO_SHARE:
                        # No meaningful rights left; stop trying to add RGOs
                        break

                entrant = spawn_firms(
                    self.good,
                    self.firm_type,
                    rng_entry,
                    n=1,
                    start_id=next_id,
                    province=province_obj,
                    # max_share left at default (1.0) for entry
                )[0]
                self.firms.append(entrant)
                next_id += 1

        return next_id

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
        self,
        cons_desired_q: int,
        firm_desired_q: int,
        q_supply: int,
        needs: tuple[int, int, int],
    ) -> tuple[int, int, str, dict[str, int], dict[int, float]]:
        """
        Clear the market given separate consumer and firm demand.

        - cons_desired_q: consumer demand for this good
        - firm_desired_q: firm input demand for this good
        - q_supply: total quantity supplied by firms this tick
        - needs: (life, everyday, luxury) thresholds for CONSUMER only

        Returns:
            q_bought_total: total quantity bought (consumer + firm)
            q_bought_consumer: quantity bought by consumers
            tier_realized: label based on consumer tiers only
            tiers_bought: dict of consumer quantities by tier
            sales_by_firm: allocation of total sales to firms
        """
        cons_desired_q = int(max(0, cons_desired_q))
        firm_desired_q = int(max(0, firm_desired_q))
        q_supply = int(max(0, q_supply))

        # Total demand that hits the market for pricing / sales
        q_total_desired = cons_desired_q + firm_desired_q
        q_bought_total = min(q_total_desired, q_supply)

        # Consumers can never buy more than their own desired quantity
        q_bought_consumer = min(cons_desired_q, q_bought_total)

        # derive tier caps from needs (consumer side)
        q_life, q_every, q_lux = needs
        inc_every = max(0, q_every)
        inc_lux   = max(0, q_lux)

        # allocate CONSUMER purchases into tiers
        life_bought = min(q_bought_consumer, q_life)
        rem = q_bought_consumer - life_bought
        every_bought = min(rem, inc_every)
        rem -= every_bought
        lux_bought = min(rem, inc_lux)

        tiers_bought = {
            "life": life_bought,
            "everyday": every_bought,
            "luxury": lux_bought,
        }

        # label of highest fully reached consumer tier
        if life_bought < q_life:
            tier_realized = "life_partial"
        elif every_bought < inc_every:
            tier_realized = "life"
        elif lux_bought < inc_lux:
            tier_realized = "everyday"
        else:
            tier_realized = "luxury"

        # proportional allocation of TOTAL sales (consumer + firm) to firms
        sales_by_firm: Dict[int, float] = {f.id: 0.0 for f in self.firms}
        if q_supply > 0 and q_bought_total > 0:
            for f in self.firms:
                if not f.active or f.q <= 0:
                    continue
                share = f.q / q_supply
                sales_by_firm[f.id] = min(f.q, share * q_bought_total)

        return q_bought_total, q_bought_consumer, tier_realized, tiers_bought, sales_by_firm

    def _book_finance(self, sales_by_firm: Dict[int, float]) -> Tuple[float, float, float]:
        # 6) With sales by firm, calculate finances
        TR_total = TC_total = Profit_total = 0.0
        for f in self.firms:
            if not f.active:
                sales_i = 0.0
            else:
                sales_i = sales_by_firm.get(f.id, 0.0)

                # unsold = quantity brought to market minus actual sales
                unsold_i = max(f.q - sales_i, 0.0)

                # accumulate unsold output in inventory
                f.output_inventory += int(unsold_i)

            TR_i, TC_i, PROF_i = f.book_finance(self.price, sales_i)
            TR_total += TR_i
            TC_total += TC_i
            Profit_total += PROF_i

        return TR_total, TC_total, Profit_total

    def _remove_inactive(self) -> None:
        # Bookkeeping: Remove inactive firms on both the market and the province
        if not any(not f.active for f in self.firms):
            return

        dead = [f for f in self.firms if not f.active]

        # Remove them from their provinces' firm lists
        for f in dead:
            prov = getattr(f, "province", None)
            if prov is not None and hasattr(prov, "firms") and prov.firms is not None:
                # remove this exact object
                prov.firms = [pf for pf in prov.firms if pf is not f]

        # Keep only active firms in the market view
        self.firms = [f for f in self.firms if f.active]

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

    def step(self,
             pop: Population,
             q_consumer: int,
             q_firm: float,
             rng_entry: np.random.Generator,
             tick: int,
             records: List[Dict],
             good_label_in_record: bool = False
             ) -> int:
        
        # NATIONAL: 'pop' can be a Country; we only use pop.needs_per_good(self.good)
        q_consumer = int(max(0, q_consumer))
        q_firm = int(max(0, q_firm))
        q_demand_total = q_consumer + q_firm

        # 1) firms choose quantities at current price
        self._update_firms(tick)
        q_supply = self._supply()

        active_firms = sum(1 for f in self.firms if f.active)

        # 2) national needs thresholds (life, everyday, luxury) for CONSUMERS
        needs = pop.needs_per_good(self.good)

        # 3) clear market with separated consumer and firm demand
        (
            q_bought_total,
            q_bought_consumer,
            tier_realized,
            tiers_bought,
            sales_by_firm,
        ) = self._clear_market(
            cons_desired_q=q_consumer,
            firm_desired_q=q_firm,
            q_supply=q_supply,
            needs=needs,
        )

        # 4) finance and firm exit
        TR_total, TC_total, Profit_total = self._book_finance(sales_by_firm)
        self._remove_inactive()

        # 5) concentration
        hhi = self._hhi(sales_by_firm, q_supply)

        # 6) record tick
        q_realized_firm = q_bought_total - q_bought_consumer
        rec = {
            "tick": tick,
            "price": self.price,
            "q_supply": q_supply,
            "q_demand": q_demand_total,
            "q_demand_consumer": q_consumer,
            "q_demand_firm": q_firm,
            "q_realized": q_bought_total,
            "q_realized_consumer": q_bought_consumer,
            "q_realized_firm": q_realized_firm,
            "life": tiers_bought["life"],
            "everyday": tiers_bought["everyday"],
            "luxury": tiers_bought["luxury"],
            "tier_realized": tier_realized,
            "revenue_total": TR_total,
            "cost_total": TC_total,
            "profit_total": Profit_total,
            "hhi": hhi,
            "active_firms": active_firms,
        }
        if good_label_in_record:
            rec["good"] = self.good
        records.append(rec)

        # 7) price update uses TOTAL demand (consumer + firm)
        self._price_update(q_demand_total, q_supply)
        self._apply_shocks(tick)
        return Profit_total
