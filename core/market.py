from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.country import Country

# pyrefly: ignore [missing-import]
import numpy as np

import config.config as cfg
import core.goods as gds
from core.firm import Firm, FirmType, spawn_firms


@dataclass
class Market:
    # --- Required ---
    good: gds.GoodID
    price: float

    # --- Optional state ---
    firms: list[Firm] = field(default_factory=list)
    profit_hist: deque[tuple[float, int]] = field(
        default_factory=lambda: deque(maxlen=cfg.ENTRY_WINDOW)
    )
    country: Country | None = field(default=None, repr=False)  # back-reference

    # --- Per-tick state for selling to downstream firms ---
    supply_now: float = 0.0       # what this market's firms brought this tick
    sold_to_firms: float = 0.0    # units already sold to downstream firms this tick

    def __post_init__(self):
        """
        Initialize firm type from good type.
        """
        if gds.is_raw(self.good):
            self.firm_type = FirmType.RGO
        else:
            self.firm_type = FirmType.Manu

    def _province_probs(self) -> tuple[list[str], np.ndarray]:
        """
        Provinces eligible for this market's firms, with normalized placement weights.
        RGOs can only be placed where the province actually has the resource.
        """
        names = list(self.country.weights.keys())
        if self.firm_type is FirmType.RGO:
            names = [n for n in names
                     if self.country.provinces[n].resources.get(self.good, 0) > 0]
        probs = np.array([self.country.weights[n] for n in names], dtype=float)
        if probs.sum() > 0:
            probs = probs / probs.sum()
        return names, probs

    def _sample_province(self, rng: np.random.Generator) -> str | None:
        """
        Pick a random eligible province based on province weights.
        Returns None if no province can host this market's firms.
        """
        names, probs = self._province_probs()
        if not names:
            return None
        return str(rng.choice(names, p=probs))

    def seed_firms(self, rng_init: np.random.Generator, n_firms: int) -> None:
        names, probs = self._province_probs()
        if not names:
            return

        counts = np.random.default_rng(rng_init.integers(0, 2**31 - 1)).multinomial(
            n=n_firms, pvals=probs
        )

        for name, count in zip(names, counts):
            if count <= 0:
                continue
            batch = spawn_firms(
                self.good,
                self.firm_type,
                rng_init,
                n=count,
                start_id=self.country.next_id,
                province=self.country.provinces[name],
                max_share=0.9,
                fill_rights=True,
            )
            for f in batch:
                f.market = self
            self.firms.extend(batch)
            self.country.next_id += count

    def _entry(self, tick_profit: float, active_firms: int) -> None:
        self.profit_hist.append((tick_profit, max(1, active_firms)))

        avg_profit_per_firm = (
            sum(p / n for p, n in self.profit_hist) / len(self.profit_hist)
            if self.profit_hist else 0.0
        )
        p_entry = 1.0 - math.exp(-cfg.ENTRY_ALPHA * max(avg_profit_per_firm, 0.0))
        max_new = max(1, int(active_firms * cfg.ENTRY_MAX_PER_TICK))

        MIN_RGO_SHARE = 0.05
        CAP_ENTRY = 1.0
        rng = self.country.rng_entry

        for _ in range(max_new):
            if rng.random() >= p_entry:
                continue

            name = self._sample_province(rng)
            if name is None:
                break
            province_obj = self.country.provinces[name]

            if self.firm_type is FirmType.RGO:
                used = province_obj.rights_given.get(self.good, 0.0)
                if max(0.0, CAP_ENTRY - used) < MIN_RGO_SHARE:
                    continue  # this province is full; another draw may land elsewhere

            entrant = spawn_firms(
                self.good,
                self.firm_type,
                rng,
                n=1,
                start_id=self.country.next_id,
                province=province_obj,
            )[0]
            entrant.market = self
            self.firms.append(entrant)
            self.country.next_id += 1

    def _firm_purchases_by_province(self) -> dict[str, float]:
        """Units of THIS good actually bought by each province's firms this tick (buy_inputs)."""
        return {
            name: sum(f.input_bought.get(self.good, 0.0) for f in prov_obj.firms)
            for name, prov_obj in self.country.provinces.items()
        }

    def _record_provinces(
        self,
        tick: int,
        cons_by_prov: dict[str, int],
        firm_by_prov: dict[str, float],
        q_consumer: int,
        q_bought_consumer: int,
        prov_records: list[dict],
    ) -> None:
        """
        Per-province demand and realized purchases, split by buyer type.
        Firm purchases are exact (tracked per firm in buy_inputs); consumers buy from
        a national pool, so their realized purchases are split by demand share.
        Also records each province's labor market (same on every good's rows).
        """
        firm_bought = self._firm_purchases_by_province()
        names = list(cons_by_prov)
        running = 0
        for i, name in enumerate(names):
            d_p = cons_by_prov[name]
            if i < len(names) - 1:
                q_p = round(d_p / q_consumer * q_bought_consumer) if q_consumer > 0 else 0
                running += q_p
            else:
                q_p = q_bought_consumer - running  # reconcile so totals match
            prov_records.append({
                "tick": tick,
                "province": name,
                "good": self.good,
                "q_demand_consumer": d_p,
                "q_demand_firm": firm_by_prov.get(name, 0.0),
                "q_realized_consumer": q_p,
                "q_realized_firm": firm_bought.get(name, 0.0),
                "wage": self.country.provinces[name].population.wage,
                "employment_rate": self.country.provinces[name].population.employment_rate,
            })

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

    def produce(self, tick: int) -> None:
        """
        Phase 1 of the tick: firms buy inputs, hire, produce, and bring production +
        inventory to market. Country runs this upstream-first, so input markets
        already have supply on the table when these firms buy from them.

        Firms are shuffled so scarce inputs aren't always grabbed by the same
        (earliest-listed) firms.
        """
        self.sold_to_firms = 0.0
        order = self.country.rng_order.permutation(len(self.firms))
        for i in order:
            self.firms[i].update_quantity(self.price, tick=tick)
        self.supply_now = float(sum(f.q for f in self.firms))

    def available_to_firms(self) -> float:
        """Units of this good still unsold to downstream firms this tick."""
        return max(0.0, self.supply_now - self.sold_to_firms)

    def sell_to_firm(self, qty: float) -> float:
        """A downstream firm buys up to qty units now. Returns the amount actually sold."""
        got = min(max(0.0, qty), self.available_to_firms())
        self.sold_to_firms += got
        return got

    def _supply(self) -> int:
        """
        After all firms have updated their quantities, get the entire output.
        """
        return int(sum(f.q for f in self.firms))

    def _hhi(self, sales_by_firm: dict[int, float], fallback_supply: int) -> float:
        """
        HHI on 0-10000 scale using sales shares this tick.
        If no sales happened, fallback to shares of supply (q) to avoid NaN.
        """
        # try based on sales
        total = sum(sales_by_firm.values())
        if total <= 0:
            # fallback: use supply shares this tick
            if fallback_supply <= 0:
                return 0.0
            shares = [(f.q / fallback_supply) for f in self.firms if f.q > 0]
        else:
            shares = [(v / total) for v in sales_by_firm.values() if v > 0]

        if not shares:
            return 0.0

        # ( monopoly: 100^2 = 10,000)
        return float(sum((100.0 * s) ** 2 for s in shares))

    def _book_finance(self, sales_by_firm: dict[int, float]) -> tuple[float, float, float]:
        # 6) With sales by firm, calculate finances
        # Unsold units go back to inventory inside Firm.book_finance
        TR_total = TC_total = Profit_total = 0.0
        for f in self.firms:
            sales_i = sales_by_firm.get(f.id, 0.0)
            TR_i, TC_i, PROF_i = f.book_finance(self.price, sales_i)
            TR_total += TR_i
            TC_total += TC_i
            Profit_total += PROF_i

        return TR_total, TC_total, Profit_total

    def _price_update(self, q_demand: int, supply_total: int) -> None:
        # Percentage excess: how much demand exceeds supply relative to supply level
        excess_pct = (q_demand - supply_total) / max(1, supply_total)
        excess_pct = max(-0.5, min(0.5, excess_pct))

        # Tatonnement target based on pct imbalance
        p_target = max(0.01, self.price * (1 +  cfg.tatonnement_speed * excess_pct))

        # Exponential smoothing towards target, this is lowkey extra
        self.price = max(0.01, (1 - cfg.price_alpha) * self.price + cfg.price_alpha * p_target)

    def _clear_market(
        self,
        demand_from_cons: int,
        demand_from_firms: int,
        supply_total: int,
        ) -> tuple[int, int, dict[int, float]]:
        """
        Clear the market given separate consumer and firm demand.

        - demand_from_cons: consumer demand for this good
        - demand_from_firms: firm input demand for this good
        - supply_total: total quantity supplied by firms this tick

        Returns:
            q_bought_total: total quantity bought (consumer + firm)
            q_bought_consumer: quantity bought by consumers
            sales_by_firm: allocation of total sales to firms
        """
        demand_from_cons = max(0, demand_from_cons)
        demand_from_firms = max(0, demand_from_firms)
        supply_total = max(0, supply_total)

        # Firms already bought during production (buy_inputs), so they come first;
        # consumers buy from whatever is left
        q_bought_firm = min(self.sold_to_firms, supply_total)
        q_bought_consumer = min(demand_from_cons, supply_total - q_bought_firm)
        q_bought_total = q_bought_firm + q_bought_consumer

        sales_by_firm: dict[int, float] = {f.id: 0.0 for f in self.firms}

        if supply_total > 0 and q_bought_total > 0:
            for f in self.firms:
                share = f.q / supply_total
                sales_by_firm[f.id] = min(f.q, share * q_bought_total)

        return q_bought_total, q_bought_consumer, sales_by_firm

    def update_firm_costs(self, prices: dict[gds.GoodID, float]) -> None:
        for f in self.firms:
            f.update_input_cost(prices, wage=f.province.population.wage)


    def get_demand_firm(self) -> tuple[dict[str, float], float]:
        """
        Input demand for THIS good: what downstream firms tried to buy this tick in
        buy_inputs (full plan, net of their input stock), whether or not they got it.
        Returns (demand by province, national demand).
        """
        demand_province: dict[str, float] = {}

        for prov_name, prov_obj in self.country.provinces.items():
            demand_province[prov_name] = sum(
                f.input_requested.get(self.good, 0.0) for f in prov_obj.firms
            )

        demand_nat = sum(demand_province.values())
        return demand_province, demand_nat


    def get_demand_consumer(self) -> tuple[dict[str, int], int]:
        """
        Household demand for THIS good.
        Returns (demand by province, national demand).
        """
        demand_province: dict[str, int] = {}
        prices = self.country.current_prices()

        for prov_name, prov_obj in self.country.provinces.items():
            cons_d = prov_obj.population.pop_demand(prices)
            demand_province[prov_name] = int(cons_d.get(self.good, 0))

        demand_nat = sum(demand_province.values())
        return demand_province, demand_nat


    def step(self, tick: int, records: list[dict], prov_records: list[dict]) -> None:
        firm_by_prov, q_firm = self.get_demand_firm()
        cons_by_prov, q_consumer = self.get_demand_consumer()

        q_consumer = int(max(0, q_consumer))
        q_firm = int(max(0, q_firm))
        q_demand_total = q_consumer + q_firm

        # ======================= 1) supply (firms already produced in Country's production phase) =====================================
        supply_total = self._supply()
        production_total = int(sum(f.q_produced for f in self.firms))
        active_firms = sum(1 for f in self.firms if f.active)

        # ======================= 2) clear market with separated consumer and firm demand ==============================================

        (
            q_bought_total,
            q_bought_consumer,
            sales_by_firm,
        ) = self._clear_market(
            demand_from_cons=q_consumer,
            demand_from_firms=q_firm,
            supply_total=supply_total,
        )

        # ======================= 3) finance and firm exit ==============================================================================
        TR_total, TC_total, Profit_total = self._book_finance(sales_by_firm)
        self._remove_inactive()

        # ======================= 4) record tick ========================================================================================
        q_realized_firm = q_bought_total - q_bought_consumer
        hhi = self._hhi(sales_by_firm, supply_total)
        inventory_total = int(sum(f.output_inventory for f in self.firms))
        rec = {
            "tick": tick,
            "price": self.price,
            "supply_total": supply_total,
            "production_total": production_total,
            "inventory_total": inventory_total,
            "q_demand": q_demand_total,
            "q_demand_consumer": q_consumer,
            "q_demand_firm": q_firm,
            "q_realized": q_bought_total,
            "q_realized_consumer": q_bought_consumer,
            "q_realized_firm": q_realized_firm,
            "revenue_total": TR_total,
            "cost_total": TC_total,
            "profit_total": Profit_total,
            "hhi": hhi,
            "active_firms": active_firms,
            "good": self.good,
            "employment_total": self.country.employment_total(),
            "wage_avg": self.country.average_wage(),
        }
        records.append(rec)
        self._record_provinces(tick, cons_by_prov, firm_by_prov, q_consumer, q_bought_consumer, prov_records)

        # 5) price update, then entry
        self._price_update(q_demand_total, supply_total)
        self._entry(Profit_total, active_firms)