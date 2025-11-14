from __future__ import annotations
from typing import Dict, List, Tuple
from dataclasses import dataclass

import goods as gds
from population import Population
import province as prov

@dataclass
class Country:
    provinces: Dict[str, Population]           # province name -> Population
    weights: Dict[str, float]                  # for firm seeding / entry sampling

    @classmethod
    def from_specs(cls, specs: List[prov.Province]) -> "Country":
        pops = {
            p.name: Population(size=p.pop_size, income_pc=p.income_pc)  # Changed from p["name"] to p.name, etc.
            for p in specs
        }
        weights = prov.normalized_weights(specs)  # This should now work with the fixed normalized_weights
        return cls(provinces=pops, weights=weights)

    # NATIONAL demand: sum provincial demands at given prices
    def demand_for_all_goods(self, prices: Dict[gds.GoodID, float]) -> Dict[gds.GoodID, int]:
        agg: Dict[gds.GoodID, int] = {g: 0 for g in prices.keys()}
        for pop in self.provinces.values():
            d = pop.demand_for_all_goods(prices)
            for g, q in d.items():
                agg[g] = int(agg[g] + q)
        return agg

    # NATIONAL needs threshold (for tier labels): sum provincial needs
    # Returns (life, everyday, luxury) counts at national scale
    def needs_per_good(self, good: gds.GoodID) -> Tuple[int, int, int]:
        life = every = lux = 0
        for pop in self.provinces.values():
            l, e, x = pop.needs_per_good(good)
            life += l; every += e; lux += x
        return life, every, lux