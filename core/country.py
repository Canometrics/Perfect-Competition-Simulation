from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import config.config as cfg
import core.goods as gds
import core.province as prov
from core.market import Market
from core.population import Population


@dataclass
class Country:
    provinces: dict[str, prov.Province]      # province name -> Province (each embeds a Population)
    weights: dict[str, float]                # province name -> weight for firm seeding and entry sampling
    markets: dict[gds.GoodID, Market]        # one national market per good

    @classmethod
    def build_country(cls, specs: list[prov.Province]) -> Country:
        """
        Build a Country from a list of Province specs.
        Also create one Market per good, with province weights for firm placement.
        """
        # keep the same Province instances that sim.py passes in
        provinces: dict[str, prov.Province] = {p.name: p for p in specs}

        # attach a Population to each province
        for p in provinces.values():
            p.population = Population(size=p.pop_size, income_pc=p.income_pc)

        weights = prov.normalized_weights(specs)

        # Create one national Market per good, using these weights
        markets: dict[gds.GoodID, Market] = {
            g: Market(
                good=g,
                price=gds.initial_price(g),   # per-good initial price
                province_weights=weights,
            )
            for g in gds.GOODS
        }

        return cls(provinces=provinces, weights=weights, markets=markets)

    # NATIONAL demand - sum provincial demands at given prices
    def national_demand(self, prices: dict[gds.GoodID, float]) -> dict[gds.GoodID, int]:
        agg: dict[gds.GoodID, int] = {g: 0 for g in prices}
        for province in self.provinces.values():
            pop = province.population
            if pop is None:
                continue
            d = pop.pop_demand(prices)
            for g, q in d.items():
                agg[g] = int(agg[g] + q)
        return agg

    def seed_markets(
        self,
        rng_init,
        province_map: dict[str, prov.Province],
        n_firms: int,
        start_id: int = 0,
        ) -> int:
        """
        Seed all markets with initial firms, distributed across provinces
        according to self.weights.
        """
        
        next_id = start_id
        for m in self.markets.values():
            next_id = m.seed_firms(
                rng_init=rng_init,
                n_firms=n_firms,
                start_id=next_id,
                provinces=province_map,
            )
        return next_id

    def current_prices(self) -> dict[gds.GoodID, float]: #unused at this time 10 may
        return {g: m.price for g, m in self.markets.items()}

    def country_step(
        self,
        t: int,
        goods: list[gds.GoodID],
        rng_entry: np.random.Generator,
        next_id: int,
        records: list[dict],
        prov_records: list[dict],
    ) -> int:
        """
        Run one simulation tick for the whole country.

        - Updates firms' input costs
        - Computes national input demand from firms
        - Computes provincial + national consumer demand
        - Steps each market (clearing + finance + price update)
        - Handles entry
        - Allocates realized quantities back to provinces

        Returns:
            next_id: updated next available firm id after entry
        """

        # housekeeping
        prov_names = list(self.provinces.keys())

        # 0) current prices per good from markets
        prices = self.current_prices()

        # 1) firms update their marginal cost based on input prices + wage
        for g in goods:
            market = self.markets[g]
            for f in market.firms:
                f.update_input_cost(prices, wage=cfg.WAGE)

        # 2.1) provincial firm demand
        demand_firm_province: dict[str, dict[gds.GoodID, float]] = {}

        for prov_name in prov_names:
            prov_obj = self.provinces[prov_name]
            prov_demand = {g: 0.0 for g in goods}

            for firm in prov_obj.firms:
                if not firm.input_requirements:
                    continue

                # price of the good THIS firm produces
                price_out = self.markets[firm.good].price
                q_plan = firm.plan_quantity(price=price_out)
                q_feasible = firm.hire_and_fire(q_plan, hypothetical=True)

                # convert planned output into input demand via the recipe
                for g_in, units in firm.input_requirements.items():
                    prov_demand[g_in] += q_feasible * units

            demand_firm_province[prov_name] = prov_demand

        # 2.2) national firm demand
        demand_firm_nat: dict[gds.GoodID, float] = {g: 0.0 for g in goods}

        demand_firm_nat: dict[gds.GoodID, int] = {
            g: sum(demand_firm_province[p][g] for p in demand_firm_province) for g in goods
        }

        # 3.1) per-provincial consumer demand (only households)
        demand_cons_province: dict[str, dict[gds.GoodID, int]] = {}
        for prov_name in prov_names:
            prov_obj = self.provinces[prov_name]
            cons_d = prov_obj.population.pop_demand(prices)
            total_for_p: dict[gds.GoodID, int] = {}
            for g in goods:
                total_for_p[g] = int(cons_d.get(g, 0))
            demand_cons_province[prov_name] = total_for_p

        # 3.2) national consumer demand (sum over provinces)
        demand_cons_nat: dict[gds.GoodID, int] = {
            g: sum(demand_cons_province[p][g] for p in demand_cons_province) for g in goods
        }

        # 5) step markets and entry
        realized_nat: dict[gds.GoodID, int] = {}

        for g in goods:
            market = self.markets[g]

            profit = market.step(
                q_consumer=demand_cons_nat[g],
                q_firm=demand_firm_nat[g],
                tick=t,
                records=records,
                good_label_in_record=(len(goods) > 1),
            )

            # last market record for this good / tick
            last = records[-1]
            realized_nat[g] = last["q_realized"]
            active_firms = last["active_firms"]

            # national employment this tick
            total_employed = 0
            for prov_obj in self.provinces.values():
                pop_obj = prov_obj.population
                total_employed += pop_obj.number_employed
            last["employment_total"] = int(total_employed)

            # firm entry for this good
            next_id = market._entry(
                rng_entry=rng_entry,
                next_id=next_id,
                tick_profit=profit,
                active_firms=active_firms,
                provinces=self.provinces,   # pass Province objects
            )

        # 6) allocate realized quantities back to provinces by consumer demand share
        for g in goods:
            d_nat_cons = demand_cons_nat[g]
            if d_nat_cons <= 0:
                # no consumer demand: everyone gets zero realized in province records
                for prov_name in prov_names:
                    prov_records.append({
                        "tick": t,
                        "province": prov_name,
                        "good": g,
                        "q_demand": int(demand_cons_province[prov_name][g]),
                        "q_realized": 0,
                    })
                continue

            running_sum = 0
            alloc_rows: list[tuple[str, int, int]] = []

            for i, prov_name in enumerate(prov_names):
                d_p = int(demand_cons_province[prov_name][g])  # consumer-only demand
                if i < len(prov_names) - 1:
                    share = d_p / d_nat_cons if d_nat_cons > 0 else 0.0
                    q_real_p = round(share * realized_nat[g])
                    running_sum += q_real_p
                else:
                    # reconcile last province so totals match exactly
                    q_real_p = int(realized_nat[g] - running_sum)
                alloc_rows.append((prov_name, d_p, q_real_p))

            for prov_name, d_p, q_real_p in alloc_rows:
                prov_records.append({
                    "tick": t,
                    "province": prov_name,
                    "good": g,
                    "q_demand": d_p,
                    "q_realized": q_real_p,
                })

        return next_id