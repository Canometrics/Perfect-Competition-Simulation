from __future__ import annotations

import copy

# pyrefly: ignore [missing-import]
import numpy as np

import config.config as cfg
import core.province as prov
from core.country import Country


def initialize_world(seed: int = cfg.SEED, n_firms: int = cfg.N_FIRMS) -> Country:
    specs = [copy.deepcopy(p) for p in prov.PROVINCES]
    country = Country.build_country(specs)
    country.rng_entry = np.random.default_rng(seed + 1)
    country.rng_order = np.random.default_rng(seed + 2)
    country.seed_markets(rng_init=np.random.default_rng(seed), n_firms=n_firms)
    return country