import pandas as pd

import config.config as cfg
import core.goods as gds
from core.firm import Firm
from services.initialize import initialize_world


def simulate_multi(T: int | None = None, p0: float | None = None) -> tuple[pd.DataFrame, list[Firm], pd.DataFrame]:
    T = cfg.T if T is None else T

    goods = gds.GOODS

    country, _province_map, rng_entry, next_id = initialize_world(start_id=0)

    records: list[dict] = []
    prov_records: list[dict] = []  # collect per-province panel rows

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
    firms: list[Firm] = [
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