import pandas as pd
from sim import simulate_multi
from plots import plot_market, plot_tier_ladder
import config as cfg
if __name__ == "__main__":
    df_market, firms = simulate_multi()
    df_market.to_csv("simulation_results_multi.csv", index=False)
    plot_market(df_market, cfg.SHOCK_TICK)
    plot_tier_ladder(df_market, cfg.SHOCK_TICK)

    # final snapshot
    last_rows = []
    for f in firms:
        h = f.history
        if len(h) == 0:
            continue
        last = h.iloc[-1]
        last_rows.append({
            "firm": f.id,
            "active": f.active,
            "FC": round(f.FC, 2),
            "MC": round(f.MC, 2),
            "capacity": round(f.capacity, 2),
            "q_final": round(float(last["quantity"]), 2),
            "profit_final": round(float(last["profit"]), 2),
            "treasury": round(float(f.treasury), 2),
            "loss_streak": f.loss_streak,
        })
    df_firms_final = pd.DataFrame(last_rows).sort_values(["active", "profit_final"], ascending=[False, False])
    print(df_firms_final.to_string(index=False))
