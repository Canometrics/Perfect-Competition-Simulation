from __future__ import annotations
from typing import Dict, List, Tuple
from dataclasses import dataclass

import goods as gds
import config as cfg
from population import Population
import province as prov
from market import Market


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
