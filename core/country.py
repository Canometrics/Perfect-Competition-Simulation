from __future__ import annotations
from typing import Dict, List, Tuple
from dataclasses import dataclass
import numpy as np

import core.goods as gds
import config.config as cfg
from core.population import Population
import core.province as prov
from core.market import Market


@dataclass
class Country:
    # province name -> Province object (which embeds a Population)
    provinces: Dict[str, prov.Province]
    # weights used for firm seeding and entry sampling
    weights: Dict[str, float]
    # national markets, one per good
    markets: Dict[gds.GoodID, Market]

    @classmethod
    def from_specs(cls, specs: List[prov.Province]) -> "Country":
        """
        Build a Country from a list of Province specs.
        Also create one Market per good, with province weights for firm placement.
        """
        # keep the same Province instances that sim.py passes in
        provinces: Dict[str, prov.Province] = {p.name: p for p in specs}

        # attach a Population to each province
        for p in provinces.values():
            p.population = Population(size=p.pop_size, income_pc=p.income_pc)

        weights = prov.normalized_weights(specs)

        # Create one national Market per good, using these weights
        markets: Dict[gds.GoodID, Market] = {
            g: Market(
                good=g,
                price=gds.initial_price(g),   # per-good initial price
                province_weights=weights,
            )
            for g in gds.GOODS
        }

        return cls(provinces=provinces, weights=weights, markets=markets)

    # NATIONAL demand - sum provincial demands at given prices
    def demand_for_all_goods(self, prices: Dict[gds.GoodID, float]) -> Dict[gds.GoodID, int]:
        agg: Dict[gds.GoodID, int] = {g: 0 for g in prices.keys()}
        for province in self.provinces.values():
            pop = province.population
            if pop is None:
                continue
            d = pop.demand_for_all_goods(prices)
            for g, q in d.items():
                agg[g] = int(agg[g] + q)
        return agg

    # NATIONAL needs threshold (for tier labels) - sum provincial needs
    def needs_per_good(self, good: gds.GoodID) -> Tuple[int, int, int]:
        life = every = lux = 0
        for province in self.provinces.values():
            pop = province.population
            if pop is None:
                continue
            l, e, x = pop.needs_per_good(good)
            life += l
            every += e
            lux += x
        return life, every, lux

    def seed_markets(
        self,
        rng_init,
        province_map: Dict[str, prov.Province],
        n_firms: int,
        start_id: int = 0,
    ) -> int:
        """
        Seed all markets with initial firms, distributed across provinces
        according to self.weights.
        """
        next_id = start_id
        for g, m in self.markets.items():
            next_id = m.seed(
                rng_init=rng_init,
                n_firms=n_firms,
                start_id=next_id,
                provinces=province_map,
            )
        return next_id

    def current_prices(self) -> Dict[gds.GoodID, float]:
        return {g: m.price for g, m in self.markets.items()}

    def country_step(
        self,
        t: int,
        goods: List[gds.GoodID],
        rng_entry: np.random.Generator,
        next_id: int,
        records: List[Dict],
        prov_records: List[Dict],
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

        # 0) current prices per good from markets
        prices: Dict[gds.GoodID, float] = {
            g: self.markets[g].price for g in goods
        }

        # 1) firms update their marginal cost based on input prices + wage
        for g in goods:
            market = self.markets[g]
            for f in market.firms:
                f.update_input_cost(prices, wage=cfg.WAGE)

        province_names = list(self.provinces.keys())

        # 2) national firm input-demand per good (based on feasible output)
        input_demand_nat: Dict[gds.GoodID, float] = {g: 0.0 for g in goods}

        for g_out in goods:
            market_out = self.markets[g_out]
            for firm in market_out.firms:
                # only firms with input requirements (manufacturing) create input demand
                if not getattr(firm, "input_requirements", None):
                    continue

                # planned output at current price
                q_plan = firm.plan_quantity(price=market_out.price)
                # feasible output given labor constraint (hypothetical = don't actually hire)
                q_feasible = firm.hire_and_fire(q_plan, hypothetical=True)

                for g_in, units in firm.input_requirements.items():
                    input_demand_nat[g_in] += q_feasible * units

        # 3) per-province consumer demand (only households)
        demand_by_prov: Dict[str, Dict[gds.GoodID, int]] = {}
        for pname in province_names:
            prov_obj = self.provinces[pname]
            cons_d = prov_obj.population.demand_for_all_goods(prices)
            total_for_p: Dict[gds.GoodID, int] = {}
            for g in goods:
                q_cons = float(cons_d.get(g, 0))
                total_for_p[g] = int(q_cons)
            demand_by_prov[pname] = total_for_p

        # 4) national consumer demand (sum over provinces)
        consumer_nat: Dict[gds.GoodID, int] = {
            g: sum(demand_by_prov[p][g] for p in demand_by_prov) for g in goods
        }

        # 5) step markets and entry
        realized_nat: Dict[gds.GoodID, int] = {}

        for g in goods:
            market = self.markets[g]

            profit = market.step(
                pop=self,                        # Country has needs_per_good(...)
                q_consumer=consumer_nat[g],
                q_firm=input_demand_nat[g],
                rng_entry=rng_entry,
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
                if pop_obj is not None:
                    total_employed += int(getattr(pop_obj, "number_employed", 0))
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
        prov_names = list(self.provinces.keys())
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
                    share = d_p / d_nat_cons if d_nat_cons > 0 else 0.0
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

        return next_id
