from __future__ import annotations
import copy
from typing import Dict, Tuple
import numpy as np

import config.config as cfg
import core.goods as gds
import core.province as prov
from core.country import Country


def initialize_world(
    seed: int | None = None,
    n_firms: int | None = None,
    start_id: int = 0,
) -> Tuple[Country, Dict[str, prov.Province], np.random.Generator, int]:
    """
    Initialize Country, Provinces, Markets, and seed initial firms.

    Returns:
        country: Country instance
        province_map: dict of Province objects
        rng_entry: RNG for entry process
        next_id: next available firm id after initial seeding
    """
    seed = cfg.SEED if seed is None else seed
    n_firms = cfg.N_FIRMS if n_firms is None else n_firms

    # Safe copy of province specs
    specs = [copy.deepcopy(p) for p in prov.PROVINCES]
    country = Country.from_specs(specs)
    province_map = country.provinces

    # RNGs
    rng_init = np.random.default_rng(seed)       # only for seeding
    rng_entry = np.random.default_rng(seed + 1)  # used every tick during entry

    # Initial firm spawning
    next_id = country.seed_markets(
        rng_init=rng_init,
        province_map=province_map,
        n_firms=n_firms,
        start_id=start_id,
    )

    # rng_init not needed afterwards
    return country, province_map, rng_entry, next_id
