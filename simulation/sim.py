from typing import Tuple, List, Dict
import numpy as np
import pandas as pd
import copy

import core.goods as gds
import config.config as cfg
from core.firm import Firm
import core.province as prov
from core.country import Country
from services.initialize import initialize_world

def simulate_multi(T: int | None = None, p0: float | None = None) -> Tuple[pd.DataFrame, List[Firm], pd.DataFrame]:
    T = cfg.T if T is None else T

    goods = gds.GOODS

    country, province_map, rng_entry, next_id = initialize_world(
        seed=cfg.SEED,
        n_firms=cfg.N_FIRMS,
        start_id=0,
    )

    records: List[Dict] = []
    prov_records: List[Dict] = []  # collect per-province panel rows

    # MAIN LOOP – delegate tick logic to the Country
    for t in range(T + 1):
        next_id = country.country_step(
            t=t,
            goods=goods,
            rng_entry=rng_entry,
            next_id=next_id,
            records=records,
            prov_records=prov_records,
        )

    # collect firms from provinces (provinces own firms)
    firms: List[Firm] = [
        f
        for prov_obj in country.provinces.values()
        for f in prov_obj.firms
    ]

    df_province = pd.DataFrame.from_records(prov_records)

    # annotate firm histories
    for f in firms:
        f.history["good"] = f.good

    df_market = pd.DataFrame.from_records(records)
    return df_market, firms, df_province
