
from typing import List
import numpy as np

from config import FC_LOGMEAN, FC_LOGSD, MC_MEAN, MC_SD, CAP_LOW, CAP_HIGH, START_CAPITAL
from firm import Firm

def spawn_firms(rng: np.random.Generator, n: int, start_id: int = 0) -> List[Firm]:
    """Vectorized firm creation for both initial seeding and late entry."""
    # FC ~ lognormal (scaled), MC ~ normal clipped, capacity ~ uniform
    FC = 20.0 * np.exp(rng.normal(FC_LOGMEAN, FC_LOGSD, size=n))
    MC = np.clip(rng.normal(MC_MEAN, MC_SD, size=n), 0.5, None)
    CAP = rng.uniform(CAP_LOW, CAP_HIGH, size=n)

    firms = [
        Firm(
            id=start_id + i,
            FC=float(FC[i]),
            MC=float(MC[i]),
            base_capacity=float(CAP[i]),
            capacity=int(CAP[i]),
            q=0.0,
            start_capital=float(START_CAPITAL)
        )
        for i in range(n)
    ]
    return firms
