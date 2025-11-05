
import matplotlib.pyplot as plt

def plot_market(df_market, SHOCK_TICK: int):
    plt.figure()
    plt.plot(df_market["tick"], df_market["q_demand"], label="Quantity Demanded")
    plt.plot(df_market["tick"], df_market["q_realized"], label="Quantity Bought")
    plt.plot(df_market["tick"], df_market["q_supply"], label="Quantity Supplied")
    plt.axvline(SHOCK_TICK, linestyle=":", linewidth=1)
    plt.xlabel("Tick"); plt.ylabel("Units"); plt.legend(); plt.title("Quantities over time"); plt.show()

    plt.figure()
    plt.plot(df_market["tick"], df_market["price"], label="Price")
    plt.axvline(SHOCK_TICK, linestyle=":", linewidth=1)
    plt.xlabel("Tick"); plt.ylabel("Price"); plt.legend(); plt.title("Price over time"); plt.show()

    plt.figure()
    plt.plot(df_market["tick"], df_market["profit_total"], label="Total Profit")
    plt.axvline(SHOCK_TICK, linestyle=":", linewidth=1)
    plt.xlabel("Tick"); plt.ylabel("Profit"); plt.legend(); plt.title("Total Profit over time"); plt.show()

    plt.figure()
    plt.plot(df_market["tick"], df_market["active_firms"], label="Active firms")
    plt.axvline(SHOCK_TICK, linestyle=":", linewidth=1)
    plt.xlabel("Tick"); plt.ylabel("Count"); plt.legend(); plt.title("Active firms over time"); plt.show()


def plot_tier_ladder(df_market, SHOCK_TICK: int):
    tier_to_level = {
        "life_partial": 0,
        "life": 1,
        "everyday": 2,
        "luxury": 3,
    }
    level_names = ["Partial Life", "Life", "Everyday", "Luxury"]

    def _to_levels(s):
        return s.map(tier_to_level).fillna(0)

    has_realized = "tier_realized" in df_market.columns
    if has_realized:
        lvl_real = _to_levels(df_market["tier_realized"])

    plt.figure()
    if has_realized:
        plt.plot(df_market["tick"], lvl_real, drawstyle="steps-post", linestyle="--", label="Realized")
    plt.axvline(SHOCK_TICK, linestyle=":", linewidth=1)
    plt.title("Highest Spending Tier Reached")
    plt.yticks([0, 1, 2, 3], level_names)
    plt.xlabel("Tick"); plt.ylabel("Tier level")
    plt.legend(); plt.grid(True); plt.show()
