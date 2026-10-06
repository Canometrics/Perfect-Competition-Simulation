from __future__ import annotations

from dataclasses import dataclass, field

# pyrefly: ignore [missing-import]
import numpy as np

import core.goods as gds
import core.province as prov
from core.market import Market
from core.population import Population


@dataclass
class Country:
    provinces: dict[str, prov.Province]
    weights: dict[str, float]
    markets: dict[gds.GoodID, Market]
    rng_entry: np.random.Generator = field(default_factory=np.random.default_rng, repr=False)
    rng_order: np.random.Generator = field(default_factory=np.random.default_rng, repr=False)  # firm order within a tick
    next_id: int = 0

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
            )
            for g in gds.GOODS
        }

        country = cls(provinces=provinces, weights=weights, markets=markets)
        for m in markets.values():
            m.country = country
        return country

    def seed_markets(self, rng_init: np.random.Generator, n_firms: int) -> None:
        for m in self.markets.values():
            m.seed_firms(rng_init=rng_init, n_firms=n_firms)

    def current_prices(self) -> dict[gds.GoodID, float]:
        return {g: m.price for g, m in self.markets.items()}

    def employment_total(self) -> int:
        return sum(p.population.number_employed for p in self.provinces.values())

    def market_order(self) -> list[Market]:
        """
        Markets sorted so each good comes after the goods it uses as inputs
        (raw goods first), from the production recipes.
        """
        depth: dict[gds.GoodID, int] = {}

        def d(g: gds.GoodID) -> int:
            if g not in depth:
                inputs = gds.PRODUCTION_RECIPES.get(g, {}).get("inputs", {})
                depth[g] = 0 if not inputs else 1 + max(d(i) for i in inputs)
            return depth[g]

        return sorted(self.markets.values(), key=lambda m: d(m.good))

    def country_step(self, t: int, records: list[dict], prov_records: list[dict]) -> None:
        # one price snapshot so every firm's MC uses the same prices
        prices = self.current_prices()
        for m in self.markets.values():
            m.update_firm_costs(prices)

        ordered = self.market_order()

        # Phase 1, upstream first: raw firms produce, then manufacturers buy inputs
        # from that supply, hire, and produce
        for m in ordered:
            m.produce(tick=t)

        # Phase 2: each market clears; firm purchases are already settled,
        # consumers buy from what's left
        for m in ordered:
            m.step(tick=t, records=records, prov_records=prov_records)