import pandas as pd

import config.config as cfg
from core.firm import Firm
from services.initialize import initialize_world


def simulate_multi(T: int | None = None) -> tuple[pd.DataFrame, list[Firm], pd.DataFrame]:
    T = cfg.T if T is None else T

    country = initialize_world()

    records: list[dict] = []
    prov_records: list[dict] = []

    for t in range(T + 1):
        country.country_step(t=t, records=records, prov_records=prov_records)

    # provinces own firms
    firms: list[Firm] = [
        f
        for prov_obj in country.provinces.values()
        for f in prov_obj.firms
    ]

    for f in firms:
        f.history["good"] = f.good

    df_market = pd.DataFrame.from_records(records)
    df_province = pd.DataFrame.from_records(prov_records)

    return df_market, firms, df_province