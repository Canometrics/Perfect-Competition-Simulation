import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# You can keep these imports if you plan to use plots.py helpers elsewhere,
# but below we draw province charts inline for Streamlit-friendly rendering.
from plots import (
    plot_market, plot_tier_ladder,
    plot_province_demand, plot_province_realized, plot_province_shares_stacked
)

import importlib
import time

# Local modules
import config as cfg
import sim as sim_module

st.set_page_config(page_title="Supply–Demand Simulation", layout="wide")

st.title("Interactive Supply–Demand Simulation")

with st.sidebar:
    st.header("Configuration")
    # Core
    seed = st.number_input("Seed", min_value=0, value=cfg.SEED, step=1)
    T = st.number_input("Ticks (T)", min_value=10, max_value=2000, value=cfg.T, step=10)
    tatonnement_speed = st.number_input(
        "Price adjustment speed",
        min_value=0.0001, max_value=1.0,
        value=float(getattr(cfg, "tatonnement_speed", 0.007)),
        step=0.001, format="%.4f"
    )
    price_alpha = st.number_input(
        "Price smoothing factor",
        min_value=0.0, max_value=1.0,
        value=float(getattr(cfg, "price_alpha", 0.30)),
        step=0.1, format="%.4f"
    )

    st.markdown("---")
    st.subheader("Firms and Entry")
    N_FIRMS = st.number_input("Initial # firms", min_value=1, max_value=10000, value=int(cfg.N_FIRMS), step=1)
    ENTRY_ALPHA = st.number_input("Entry alpha", min_value=0.0, max_value=0.1, value=float(cfg.ENTRY_ALPHA), step=0.0005, format="%.4f")
    ENTRY_WINDOW = st.number_input("Entry window (ticks)", min_value=1, max_value=200, value=int(cfg.ENTRY_WINDOW), step=1)
    ENTRY_MAX_PER_TICK = st.number_input("Entry max per tick (pct of all firms)", min_value=0.0, max_value=1.0, value=float(cfg.ENTRY_MAX_PER_TICK), step=0.01)

    st.markdown("---")
    st.subheader("Shock")
    SHOCK_TICK = st.number_input("Shock tick", min_value=0, max_value=100000, value=int(cfg.SHOCK_TICK), step=1)
    SHOCK_DURATION = st.number_input("Shock duration (0 = permanent)", min_value=0, max_value=100000, value=int(cfg.SHOCK_DURATION), step=1)
    USE_MC_SHOCK = st.checkbox("Use MC (input cost) shock", value=getattr(cfg, "USE_MC_SHOCK", True))
    MC_MULT_DURING_SHOCK = st.number_input("MC multiplier during shock", min_value=0.01, max_value=100.0, value=float(getattr(cfg, "MC_MULT_DURING_SHOCK", 2.0)), step=0.1)
    USE_CAPACITY_SHOCK = st.checkbox("Use capacity shock", value=getattr(cfg, "USE_CAPACITY_SHOCK", False))
    CAP_MULT_DURING_SHOCK = st.number_input("Capacity multiplier during shock", min_value=0.01, max_value=10.0, value=float(getattr(cfg, "CAP_MULT_DURING_SHOCK", 0.5)), step=0.05)

    st.markdown("---")
    st.subheader("Treasury")
    START_CAPITAL = st.number_input("Starting capital per firm", min_value=0.0, max_value=1e9, value=float(getattr(cfg, "START_CAPITAL", 6000.0)), step=500.0)
    TREASURY_GRACE_TICKS = st.number_input("Grace ticks with negative treasury", min_value=0, max_value=100, value=int(getattr(cfg, "TREASURY_GRACE_TICKS", 2)), step=1)

    st.markdown("---")
    run_btn = st.button("Run Simulation")

def set_config():
    # Assign chosen params to the config module
    cfg.SEED = seed
    cfg.T = int(T)
    cfg.tatonnement_speed = float(tatonnement_speed)
    cfg.price_alpha = float(price_alpha)

    cfg.N_FIRMS = int(N_FIRMS)
    cfg.ENTRY_ALPHA = float(ENTRY_ALPHA)
    cfg.ENTRY_WINDOW = int(ENTRY_WINDOW)
    cfg.ENTRY_MAX_PER_TICK = float(ENTRY_MAX_PER_TICK)

    cfg.SHOCK_TICK = int(SHOCK_TICK)
    cfg.SHOCK_DURATION = int(SHOCK_DURATION)
    cfg.USE_MC_SHOCK = bool(USE_MC_SHOCK)
    cfg.MC_MULT_DURING_SHOCK = float(MC_MULT_DURING_SHOCK)
    cfg.USE_CAPACITY_SHOCK = bool(USE_CAPACITY_SHOCK)
    cfg.CAP_MULT_DURING_SHOCK = float(CAP_MULT_DURING_SHOCK)

    # Treasury
    cfg.START_CAPITAL = float(START_CAPITAL)
    cfg.TREASURY_GRACE_TICKS = int(TREASURY_GRACE_TICKS)

# =========================
# Run
# =========================
if run_btn:
    # 1) push sidebar values into cfg
    set_config()

    # 2) optional but helpful if any submodules did `from config import ...`
    importlib.reload(sim_module)

    with st.spinner("Simulating…"):
        t0 = time.perf_counter()
        # Updated to receive df_province from simulate_multi
        df_market, firms, df_province = sim_module.simulate_multi(T=cfg.T)
        runtime_s = time.perf_counter() - t0
    st.success("Done!")
    st.metric("Runtime", f"{runtime_s:.3f} s")

    # =========================
    # Helper: split by good if present
    # =========================
    def _groups(df: pd.DataFrame):
        if "good" in df.columns and df["good"].nunique() > 1:
            return [(str(g), d.sort_values("tick")) for g, d in df.groupby("good", sort=False)]
        return [("Market", df.sort_values("tick"))]

    groups = _groups(df_market)
    tabs = st.tabs([name for name, _ in groups])

    for tab, (name, df_g) in zip(tabs, groups):
        with tab:
            st.caption(f"Good: {name}")

            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Quantities over time")
                fig, ax = plt.subplots()
                if "q_demand" in df_g.columns:
                    ax.plot(df_g["tick"], df_g["q_demand"], label="Quantity Demanded")
                if "q_realized" in df_g.columns:
                    ax.plot(df_g["tick"], df_g["q_realized"], label="Quantity Bought")
                if "q_supply" in df_g.columns:
                    ax.plot(df_g["tick"], df_g["q_supply"], label="Quantity Supplied")
                ax.axvline(cfg.SHOCK_TICK, linestyle=":", linewidth=1)
                ax.set_xlabel("Tick"); ax.set_ylabel("Units"); ax.legend(); ax.set_title("Quantities")
                ax.grid(True)
                st.pyplot(fig)

            with c2:
                st.subheader("Price over time")
                fig, ax = plt.subplots()
                if "price" in df_g.columns:
                    ax.plot(df_g["tick"], df_g["price"], label="Price")
                ax.axvline(cfg.SHOCK_TICK, linestyle=":", linewidth=1)
                ax.set_xlabel("Tick"); ax.set_ylabel("Price"); ax.legend(); ax.set_title("Price")
                ax.grid(True)
                st.pyplot(fig)

            c3, c4 = st.columns(2)
            with c3:
                st.subheader("Total Profit")
                fig, ax = plt.subplots()
                if "profit_total" in df_g.columns:
                    ax.plot(df_g["tick"], df_g["profit_total"], label="Total Profit")
                ax.axvline(cfg.SHOCK_TICK, linestyle=":", linewidth=1)
                ax.set_xlabel("Tick"); ax.set_ylabel("Profit"); ax.legend(); ax.set_title("Profit")
                ax.grid(True)
                st.pyplot(fig)

            with c4:
                st.subheader("Active Firms")
                fig, ax = plt.subplots()
                if "active_firms" in df_g.columns:
                    ax.plot(df_g["tick"], df_g["active_firms"], label="Active Firms")
                ax.axvline(cfg.SHOCK_TICK, linestyle=":", linewidth=1)
                ax.set_xlabel("Tick"); ax.set_ylabel("Count"); ax.legend(); ax.set_title("Active Firms")
                ax.grid(True)
                st.pyplot(fig)

            with c4:
                st.subheader("HHI (0–10,000)")
                fig, ax = plt.subplots()
                if "hhi" in df_g.columns:
                    ax.plot(df_g["tick"], df_g["hhi"], label="HHI")
                ax.axvline(cfg.SHOCK_TICK, linestyle=":", linewidth=1)
                ax.set_xlabel("Tick"); ax.set_ylabel("HHI"); ax.legend(); ax.set_title("Concentration")
                ax.grid(True)
                st.pyplot(fig)

    # =========================
    # National employment (level)
    # =========================
    if "employment_total" in df_market.columns:
        st.header("National Employment (level)")

        # One value per tick (since it's duplicated across goods)
        emp = (
            df_market[["tick", "employment_total"]]
            .drop_duplicates(subset=["tick"])
            .sort_values("tick")
        )

        fig, ax = plt.subplots()
        ax.plot(emp["tick"], emp["employment_total"], label="Employed (national)")
        ax.axvline(cfg.SHOCK_TICK, linestyle=":", linewidth=1)
        ax.set_xlabel("Tick")
        ax.set_ylabel("Workers")
        ax.set_title("National Employment over time")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)


    # =========================
    # Combined spending-tier plot (all goods together)
    # Both goods equally important -> overall tier is the minimum tier across goods at each tick.
    # =========================
    if "tier_realized" in df_market.columns:
        st.subheader("Spending Tier (combined across goods)")
        tier_to_level = {"life_partial": 0, "life": 1, "everyday": 2, "luxury": 3}
        labels = ["Partial Life", "Life", "Everyday", "Luxury"]

        df_tier = df_market[["tick", "good", "tier_realized"]].copy()
        df_tier["level"] = df_tier["tier_realized"].map(tier_to_level).fillna(0)

        # For each tick, take the minimum level across goods (bottleneck logic)
        combined = df_tier.groupby("tick", as_index=False)["level"].min()

        fig, ax = plt.subplots()
        ax.plot(combined["tick"], combined["level"], drawstyle="steps-post", linestyle="--", label="Combined (min across goods)")
        ax.axvline(cfg.SHOCK_TICK, linestyle=":", linewidth=1)
        ax.set_yticks([0, 1, 2, 3], labels=labels)
        ax.set_xlabel("Tick"); ax.set_ylabel("Tier level")
        ax.set_title("Highest Spending Tier Reached (Combined)")
        ax.legend(); ax.grid(True)
        st.pyplot(fig)

    # =========================
    # Province views (demand, realized, and stacked shares)
    # =========================
    if isinstance(df_province, pd.DataFrame) and not df_province.empty:
        st.header("Provinces")

        # Split by good for province charts if multiple goods exist
        def _groups_prov(df: pd.DataFrame):
            if "good" in df.columns and df["good"].nunique() > 1:
                return [(str(g), d.sort_values(["tick", "province"])) for g, d in df.groupby("good", sort=False)]
            return [("Market", df.sort_values(["tick", "province"]))]

        prov_groups = _groups_prov(df_province)
        prov_tabs = st.tabs([name for name, _ in prov_groups])

        for tab, (name, df_p) in zip(prov_tabs, prov_groups):
            with tab:
                st.caption(f"Good: {name}")

                # Line plot of per-province demand
                st.subheader("Province demand over time")
                prov_names = list(df_p["province"].unique())
                fig, ax = plt.subplots()
                for prov in prov_names:
                    d = df_p[df_p["province"] == prov]
                    ax.plot(d["tick"], d["q_demand"], label=f"{prov}")
                ax.axvline(cfg.SHOCK_TICK, linestyle=":", linewidth=1)
                ax.set_xlabel("Tick"); ax.set_ylabel("Units")
                ax.set_title("Demand by province")
                ax.legend(ncols=2, fontsize=8)
                ax.grid(True)
                st.pyplot(fig)

                # Line plot of per-province realized quantity
                if "q_realized" in df_p.columns:
                    st.subheader("Province realized purchases over time")
                    fig, ax = plt.subplots()
                    for prov in prov_names:
                        d = df_p[df_p["province"] == prov]
                        ax.plot(d["tick"], d["q_realized"], label=f"{prov}")
                    ax.axvline(cfg.SHOCK_TICK, linestyle=":", linewidth=1)
                    ax.set_xlabel("Tick"); ax.set_ylabel("Units")
                    ax.set_title("Realized purchases by province")
                    ax.legend(ncols=2, fontsize=8)
                    ax.grid(True)
                    st.pyplot(fig)

                # Stacked shares over time (demand)
                st.subheader("Provincial demand shares")
                piv = df_p.pivot_table(index="tick", columns="province", values="q_demand", aggfunc="sum").fillna(0.0)
                totals = piv.sum(axis=1).replace(0, 1.0)
                shares = piv.div(totals, axis=0)
                fig, ax = plt.subplots()
                shares.plot.area(ax=ax)
                ax.set_xlabel("Tick"); ax.set_ylabel("Share")
                ax.set_ylim(0, 1)
                ax.set_title("Demand shares (stacked)")
                ax.legend(title="Province", bbox_to_anchor=(1.04, 1), loc="upper left")
                st.pyplot(fig)

        # Province table and download
        with st.expander("Province panel data"):
            st.dataframe(df_province)
            st.download_button(
                "Download province CSV",
                data=df_province.to_csv(index=False).encode("utf-8"),
                file_name="province_timeseries.csv",
                mime="text/csv",
            )

    # =========================
    # Treasury and firm snapshots (aggregate)
    # =========================
    if "treasury_total" in df_market.columns:
        c5, c6 = st.columns(2)
        with c5:
            st.subheader("Aggregate Treasury and Negative-Firm Count")
            fig, ax = plt.subplots()
            ax.plot(df_market["tick"], df_market["treasury_total"], label="Total Treasury")
            if "neg_treasury_firms" in df_market.columns:
                ax.plot(df_market["tick"], df_market["neg_treasury_firms"], label="# Firms with negative treasury")
            ax.axvline(cfg.SHOCK_TICK, linestyle=":", linewidth=1)
            ax.set_xlabel("Tick"); ax.set_ylabel("Value"); ax.legend(); ax.grid(True)
            st.pyplot(fig)

        with c6:
            st.subheader("Final-Tick Treasury Distribution")
            treas = [float(getattr(f, "treasury", 0.0)) for f in firms]
            fig, ax = plt.subplots()
            ax.hist(treas, bins=20)
            ax.set_xlabel("Firm Treasury"); ax.set_ylabel("Count")
            ax.set_title("Distribution of Firm Treasuries (Final Tick)")
            ax.grid(True)
            st.pyplot(fig)

    # =========================
    # Data (all ticks) and downloads
    # =========================
    st.header("Market data")
    st.dataframe(df_market)  # show all ticks
    csv = df_market.to_csv(index=False).encode("utf-8")
    st.download_button("Download market CSV", data=csv, file_name="market_timeseries.csv", mime="text/csv")

    # =========================
    # Final firm snapshot
    # =========================
    st.header("Final firm snapshot")
    last_rows = []
    for f in firms:
        if getattr(f, "history", None) is None or len(f.history) == 0:
            continue
        last = f.history.iloc[-1]
        row = {
            "id": getattr(f, "id", None),
            "province": getattr(f.province, "name", "National"),
            "good": getattr(f, "good", None),
            "active": bool(getattr(f, "active", True)),
            "MC": round(float(getattr(f, "MC", 0.0)), 4),
            "FC": round(float(getattr(f, "FC", 0.0)), 2),
            "capacity": round(float(getattr(f, "capacity", 0.0)), 2),
            "q_final": round(float(last.get("quantity", 0.0)), 2),
            "profit_final": round(float(last.get("profit", 0.0)), 2),
            
        }
        row["treasury"] = round(float(getattr(f, "treasury", 0.0)), 2)
        row["resource_rights"] = round(float(getattr(f, "resource_rights", 0.0) or 0.0), 4)

        last_rows.append(row)

    if last_rows:
        df_final = pd.DataFrame(last_rows).sort_values(["active", "profit_final"], ascending=[False, False])
        st.dataframe(df_final)
        st.download_button(
            "Download firm snapshot CSV",
            data=df_final.to_csv(index=False).encode("utf-8"),
            file_name="firms_final.csv",
            mime="text/csv",
        )
    else:
        st.write("No firm records.")

else:
    st.info("Set parameters in the sidebar and click Run simulation.")
