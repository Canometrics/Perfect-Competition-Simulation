# plots.py
import matplotlib.pyplot as plt

def _groups(df):
    if "good" in df.columns and df["good"].nunique() > 1:
        return [(g, d.sort_values("tick")) for g, d in df.groupby("good", sort=False)]
    else:
        return [("all", df.sort_values("tick"))]

def plot_market(df_market, SHOCK_TICK: int):
    for label, df in _groups(df_market):
        tag = "" if label == "all" else f" [{label}]"

        plt.figure()
        plt.plot(df["tick"], df["q_demand"], label="Quantity Demanded")
        plt.plot(df["tick"], df["q_realized"], label="Quantity Bought")
        plt.plot(df["tick"], df["q_supply"], label="Quantity Supplied")
        plt.axvline(SHOCK_TICK, linestyle=":", linewidth=1)
        plt.xlabel("Tick"); plt.ylabel("Units"); plt.legend(); plt.title(f"Quantities over time{tag}"); plt.show()

        plt.figure()
        plt.plot(df["tick"], df["price"], label="Price")
        plt.axvline(SHOCK_TICK, linestyle=":", linewidth=1)
        plt.xlabel("Tick"); plt.ylabel("Price"); plt.legend(); plt.title(f"Price over time{tag}"); plt.show()

        plt.figure()
        plt.plot(df["tick"], df["profit_total"], label="Total Profit")
        plt.axvline(SHOCK_TICK, linestyle=":", linewidth=1)
        plt.xlabel("Tick"); plt.ylabel("Profit"); plt.legend(); plt.title(f"Total Profit over time{tag}"); plt.show()

        plt.figure()
        plt.plot(df["tick"], df["active_firms"], label="Active firms")
        plt.axvline(SHOCK_TICK, linestyle=":", linewidth=1)
        plt.xlabel("Tick"); plt.ylabel("Count"); plt.legend(); plt.title(f"Active firms over time{tag}"); plt.show()

def plot_tier_ladder(df_market, SHOCK_TICK: int):
    tier_to_level = {"life_partial":0,"life":1,"everyday":2,"luxury":3}
    level_names = ["Partial Life","Life","Everyday","Luxury"]

    def _to_levels(s): return s.map(tier_to_level).fillna(0)

    for label, df in _groups(df_market):
        if "tier_realized" not in df.columns: 
            continue
        lvl_real = _to_levels(df["tier_realized"])
        tag = "" if label == "all" else f" [{label}]"

        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(df["tick"], lvl_real, drawstyle="steps-post", linestyle="--", label="Realized")
        plt.axvline(SHOCK_TICK, linestyle=":", linewidth=1)
        plt.title(f"Highest Spending Tier Reached{tag}")
        plt.yticks([0,1,2,3], level_names)
        plt.xlabel("Tick"); plt.ylabel("Tier level")
        plt.legend(); plt.grid(True); plt.show()
