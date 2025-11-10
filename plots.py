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

# =========================
# Province-aware additions
# =========================

def _groups_prov(df):
    """Group helper for province panel: returns [(label, df_sorted)]."""
    if "good" in df.columns and df["good"].nunique() > 1:
        return [(g, d.sort_values(["tick","province"])) for g, d in df.groupby("good", sort=False)]
    return [("all", df.sort_values(["tick","province"]))]

def plot_province_demand(df_province, SHOCK_TICK: int):
    """
    Line plot of per-province demand over time (one line per province).
    Expects columns: tick, province, q_demand, and optional good.
    """
    for label, df in _groups_prov(df_province):
        tag = "" if label == "all" else f" [{label}]"
        for prov, d in df.groupby("province", sort=False):
            plt.figure()
            plt.plot(d["tick"], d["q_demand"], label=f"{prov}")
            plt.axvline(SHOCK_TICK, linestyle=":", linewidth=1)
            plt.xlabel("Tick"); plt.ylabel("Units")
            plt.title(f"Province demand over time{tag}")
            plt.legend(); plt.grid(True); plt.show()

def plot_province_realized(df_province, SHOCK_TICK: int):
    """
    Line plot of per-province realized quantity (allocated from national clearing).
    Expects columns: tick, province, q_realized, and optional good.
    """
    if "q_realized" not in df_province.columns:
        return
    for label, df in _groups_prov(df_province):
        tag = "" if label == "all" else f" [{label}]"
        for prov, d in df.groupby("province", sort=False):
            plt.figure()
            plt.plot(d["tick"], d["q_realized"], label=f"{prov}")
            plt.axvline(SHOCK_TICK, linestyle=":", linewidth=1)
            plt.xlabel("Tick"); plt.ylabel("Units")
            plt.title(f"Province realized purchases over time{tag}")
            plt.legend(); plt.grid(True); plt.show()

def plot_province_shares_stacked(df_province, kind: str = "demand"):
    """
    Stacked area of provincial shares (demand or realized) summing to 1 each tick.
    kind in {"demand","realized"} -> uses q_demand or q_realized.
    """
    col = "q_demand" if kind == "demand" else "q_realized"
    if col not in df_province.columns:
        return

    for label, df in _groups_prov(df_province):
        tag = "" if label == "all" else f" [{label}]"
        piv = df.pivot_table(index="tick", columns="province", values=col, aggfunc="sum").fillna(0.0)
        totals = piv.sum(axis=1).replace(0, 1.0)
        shares = piv.div(totals, axis=0)

        shares.plot.area()  # default colors; one stacked area per province
        plt.xlabel("Tick"); plt.ylabel("Share")
        title_kind = "Demand" if kind == "demand" else "Realized"
        plt.title(f"Provincial {title_kind} Shares over time{tag}")
        plt.ylim(0, 1); plt.legend(title="Province", bbox_to_anchor=(1.04, 1), loc="upper left")
        plt.tight_layout(); plt.show()
