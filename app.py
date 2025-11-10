
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    p0 = st.number_input("Initial price p0", min_value=0.01, value=float(cfg.p0), step=0.5, format="%.2f")
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
    ADJ_RATE = st.number_input("Quantity adj. rate (ADJ_RATE)", min_value=0.0001, max_value=1.0, value=float(cfg.ADJ_RATE), step=0.001, format="%.4f")

    st.markdown("---")
    st.subheader("Population")
    POP_SIZE = st.number_input("Population size", min_value=100, max_value=1_000_000, value=int(cfg.POP_SIZE), step=100)
    INCOME_PC = st.number_input("Income per 100 people", min_value=0.0, value=float(cfg.INCOME_PC), step=50.0)

    st.markdown("---")
    st.subheader("Firms & Entry")
    N_FIRMS = st.number_input("Initial # firms", min_value=1, max_value=10000, value=int(cfg.N_FIRMS), step=1)
    ENTRY_ALPHA = st.number_input("Entry alpha", min_value=0.0, max_value=0.1, value=float(cfg.ENTRY_ALPHA), step=0.0005, format="%.4f")
    ENTRY_WINDOW = st.number_input("Entry window (ticks)", min_value=1, max_value=200, value=int(cfg.ENTRY_WINDOW), step=1)
    ENTRY_MAX_PER_TICK = st.number_input("Entry max per tick", min_value=0, max_value=50, value=int(cfg.ENTRY_MAX_PER_TICK), step=1)

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
    cfg.p0 = float(p0)
    cfg.tatonnement_speed = float(tatonnement_speed)
    cfg.price_alpha = float(price_alpha)
    cfg.ADJ_RATE = float(ADJ_RATE)

    cfg.POP_SIZE = int(POP_SIZE)
    cfg.INCOME_PC = float(INCOME_PC)

    cfg.N_FIRMS = int(N_FIRMS)
    cfg.ENTRY_ALPHA = float(ENTRY_ALPHA)
    cfg.ENTRY_WINDOW = int(ENTRY_WINDOW)
    cfg.ENTRY_MAX_PER_TICK = int(ENTRY_MAX_PER_TICK)

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
        df_market, firms = sim_module.simulate_multi(T=cfg.T, p0=cfg.p0)
        runtime_s = time.perf_counter() - t0
    st.success("Done!")
    # Show runtime on the page
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

            # Per-good realized spending tier (optional view)
            if "tier_realized" in df_g.columns:
                st.subheader("Highest Spending Tier Reached (Realized, per-good)")
                tier_to_level = {"life_partial": 0, "life": 1, "everyday": 2, "luxury": 3}
                level_names = ["Partial Life", "Life", "Everyday", "Luxury"]
                lvl_real = df_g["tier_realized"].map(tier_to_level).fillna(0)
                fig, ax = plt.subplots()
                ax.plot(df_g["tick"], lvl_real, drawstyle="steps-post", linestyle="--", label="Realized")
                ax.axvline(cfg.SHOCK_TICK, linestyle=":", linewidth=1)
                ax.set_yticks([0, 1, 2, 3], labels=level_names)
                ax.set_xlabel("Tick"); ax.set_ylabel("Tier level"); ax.set_title("Spending Tier (per-good)")
                ax.legend(); ax.grid(True)
                st.pyplot(fig)

    # =========================
    # NEW: Combined spending-tier plot (all goods together)
    # - Both goods equally important -> overall tier is the minimum tier across goods at each tick.
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
    # Treasury & firm snapshots (aggregate)
    # =========================
    if "treasury_total" in df_market.columns:
        c5, c6 = st.columns(2)
        with c5:
            st.subheader("Aggregate Treasury & Negative-Firm Count")
            fig, ax = plt.subplots()
            ax.plot(df_market["tick"], df_market["treasury_total"], label="Total Treasury")
            if "neg_treasury_firms" in df_market.columns:
                ax.plot(df_market["tick"], df_market["neg_treasury_firms"], label="# Firms < 0 Treasury")
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
    # Data (all ticks) + download
    # =========================
    st.header("Data")
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
            "active": bool(getattr(f, "active", True)),
            "MC": round(float(getattr(f, "MC", 0.0)), 4),
            "FC": round(float(getattr(f, "FC", 0.0)), 2),
            "capacity": round(float(getattr(f, "capacity", 0.0)), 2),
            "q_final": round(float(last.get("quantity", 0.0)), 2),
            "profit_final": round(float(last.get("profit", 0.0)), 2),
        }
        if hasattr(f, "treasury"):
            row["treasury"] = round(float(getattr(f, "treasury", 0.0)), 2)
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
    st.info("Set parameters in the sidebar and click “Run simulation”.")
