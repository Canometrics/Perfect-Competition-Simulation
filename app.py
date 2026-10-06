import importlib
import time

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import config.config as cfg
import services.initialize as init_module
import simulation.sim as sim_module

st.set_page_config(page_title="Supply-Demand Simulation", layout="wide")
st.title("Interactive Supply-Demand Simulation")


# =========================
# Helpers
# =========================
def line_chart(df: pd.DataFrame, cols: dict[str, str], title: str, ylabel: str,
               x: str = "tick", legend_kw: dict | None = None) -> None:
    """Plot columns of df against x as lines. Does nothing if none of the columns exist."""
    present = {c: label for c, label in cols.items() if c in df.columns}
    if not present:
        return
    fig, ax = plt.subplots()
    for col, label in present.items():
        ax.plot(df[x], df[col], label=label)
    ax.set(xlabel="Tick", ylabel=ylabel, title=title)
    ax.legend(**(legend_kw or {}))
    ax.grid(True)
    st.pyplot(fig)
    plt.close(fig)


def split_by_good(df: pd.DataFrame, sort_cols: list[str]) -> list[tuple[str, pd.DataFrame]]:
    """Return [(good_name, sub_df), ...], or a single ("Market", df) if there is one good."""
    if "good" in df.columns and df["good"].nunique() > 1:
        return [(str(g), d.sort_values(sort_cols)) for g, d in df.groupby("good", sort=False)]
    return [("Market", df.sort_values(sort_cols))]


def download_csv(df: pd.DataFrame, label: str, file_name: str) -> None:
    st.download_button(label, data=df.to_csv(index=False).encode("utf-8"),
                       file_name=file_name, mime="text/csv")


# =========================
# Sidebar
# =========================
def sidebar_settings() -> tuple[dict, bool]:
    """Render the sidebar. Returns ({cfg_attribute: value}, run_clicked)."""
    with st.sidebar:
        st.header("Configuration")
        s = {
            "SEED": st.number_input("Seed", min_value=0, value=int(cfg.SEED), step=1),
            "T": st.number_input("Ticks (T)", min_value=10, max_value=2000, value=int(cfg.T), step=10),
            "tatonnement_speed": st.number_input(
                "Price adjustment speed", min_value=0.0001, max_value=1.0,
                value=float(cfg.tatonnement_speed), step=0.001, format="%.4f"),
            "price_alpha": st.number_input(
                "Price smoothing factor", min_value=0.0, max_value=1.0,
                value=float(cfg.price_alpha), step=0.1, format="%.4f"),
        }

        st.markdown("---")
        st.subheader("Firms and Entry")
        s.update({
            "N_FIRMS": st.number_input("Initial # firms", min_value=1, max_value=10000,
                                       value=int(cfg.N_FIRMS), step=1),
            "ENTRY_ALPHA": st.number_input("Entry alpha", min_value=0.0, max_value=0.1,
                                           value=float(cfg.ENTRY_ALPHA), step=0.0005, format="%.4f"),
            "ENTRY_WINDOW": st.number_input("Entry window (ticks)", min_value=1, max_value=200,
                                            value=int(cfg.ENTRY_WINDOW), step=1),
            "ENTRY_MAX_PER_TICK": st.number_input("Entry max per tick (pct of all firms)",
                                                  min_value=0.0, max_value=1.0,
                                                  value=float(cfg.ENTRY_MAX_PER_TICK), step=0.01),
        })

        st.markdown("---")
        st.subheader("Treasury")
        s.update({
            "START_CAPITAL": st.number_input("Starting capital per firm", min_value=0.0, max_value=1e9,
                                             value=float(cfg.START_CAPITAL), step=500.0),
            "TREASURY_GRACE_TICKS": st.number_input("Grace ticks with negative treasury",
                                                    min_value=0, max_value=100,
                                                    value=int(cfg.TREASURY_GRACE_TICKS), step=1),
        })

        st.markdown("---")
        run = st.button("Run Simulation")
    return s, run


def apply_settings(settings: dict) -> None:
    for name, value in settings.items():
        setattr(cfg, name, value)
    # initialize_world() reads cfg.SEED / cfg.N_FIRMS in its default arguments, which are
    # evaluated at import time. Reload it (then sim, which imports it) so new values apply.
    importlib.reload(init_module)
    importlib.reload(sim_module)


# =========================
# Sections
# =========================
def render_market_tabs(df_market: pd.DataFrame) -> None:
    groups = split_by_good(df_market, ["tick"])
    for tab, (name, df_g) in zip(st.tabs([n for n, _ in groups]), groups):
        with tab:
            st.caption(f"Good: {name}")

            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Quantities over time")
                line_chart(df_g, {"q_demand": "Quantity Demanded",
                                  "q_realized": "Quantity Bought",
                                  "supply_total": "Quantity Supplied"},
                           title="Quantities", ylabel="Units")
            with c2:
                st.subheader("Price over time")
                line_chart(df_g, {"price": "Price"}, title="Price", ylabel="Price")

            c3, c4 = st.columns(2)
            with c3:
                st.subheader("Total Profit")
                line_chart(df_g, {"profit_total": "Total Profit"}, title="Profit", ylabel="Profit")
                # Only drawn once the sim records these columns
                line_chart(df_g, {"treasury_total": "Total Treasury",
                                  "neg_treasury_firms": "# Firms with negative treasury"},
                           title="Treasury", ylabel="Value")
            with c4:
                st.subheader("Active Firms")
                line_chart(df_g, {"active_firms": "Active Firms"}, title="Active Firms", ylabel="Count")
                st.subheader("HHI (0–10,000)")
                line_chart(df_g, {"hhi": "HHI"}, title="Concentration", ylabel="HHI")


def render_employment(df_market: pd.DataFrame) -> None:
    if "employment_total" not in df_market.columns:
        return
    st.header("National Employment (level)")
    # employment_total is duplicated across goods; keep one row per tick
    emp = df_market[["tick", "employment_total"]].drop_duplicates("tick").sort_values("tick")
    line_chart(emp, {"employment_total": "Employed (national)"},
               title="National Employment over time", ylabel="Workers")


def render_provinces(df_province: pd.DataFrame) -> None:
    if not isinstance(df_province, pd.DataFrame) or df_province.empty:
        return
    st.header("Provinces")

    prov_legend = {"ncols": 2, "fontsize": 8}
    groups = split_by_good(df_province, ["tick", "province"])
    for tab, (name, df_p) in zip(st.tabs([n for n, _ in groups]), groups):
        with tab:
            st.caption(f"Good: {name}")
            demand = df_p.pivot_table(index="tick", columns="province",
                                      values="q_demand", aggfunc="sum").fillna(0.0)
            provs = {p: p for p in demand.columns}

            st.subheader("Province demand over time")
            line_chart(demand.reset_index(), provs, title="Demand by province",
                       ylabel="Units", legend_kw=prov_legend)

            if "q_realized" in df_p.columns:
                realized = df_p.pivot_table(index="tick", columns="province",
                                            values="q_realized", aggfunc="sum").fillna(0.0)
                st.subheader("Province realized purchases over time")
                line_chart(realized.reset_index(), provs, title="Realized purchases by province",
                           ylabel="Units", legend_kw=prov_legend)

            st.subheader("Provincial demand shares")
            shares = demand.div(demand.sum(axis=1).replace(0, 1.0), axis=0)
            fig, ax = plt.subplots()
            shares.plot.area(ax=ax)
            ax.set(xlabel="Tick", ylabel="Share", title="Demand shares (stacked)", ylim=(0, 1))
            ax.legend(title="Province", bbox_to_anchor=(1.04, 1), loc="upper left")
            st.pyplot(fig)
            plt.close(fig)

    with st.expander("Province panel data"):
        st.dataframe(df_province)
        download_csv(df_province, "Download province CSV", "province_timeseries.csv")


def render_treasury_histogram(firms: list) -> None:
    st.subheader("Final-Tick Treasury Distribution")
    fig, ax = plt.subplots()
    ax.hist([f.treasury for f in firms], bins=20)
    ax.set(xlabel="Firm Treasury", ylabel="Count", title="Distribution of Firm Treasuries (Final Tick)")
    ax.grid(True)
    st.pyplot(fig)
    plt.close(fig)


def firm_snapshot(firms: list) -> pd.DataFrame:
    rows = []
    for f in firms:
        if f.history.empty:
            continue
        last = f.history.iloc[-1]
        rows.append({
            "id": f.id,
            "province": f.province.name if f.province else "National",
            "good": f.good,
            "active": f.active,
            "MC": round(f.MC, 4),
            "FC": round(f.FC, 2),
            "capacity": round(f.capacity, 2),
            "q_final": round(last["quantity"], 2),
            "profit_final": round(last["profit"], 2),
            "output_inventory": int(f.output_inventory),
            "treasury": round(f.treasury, 2),
            "resource_rights": round(f.resource_rights or 0.0, 4),
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["active", "profit_final"], ascending=[False, False])


# =========================
# Main
# =========================
settings, run_clicked = sidebar_settings()

if not run_clicked:
    st.info("Set parameters in the sidebar and click Run simulation.")
    st.stop()

apply_settings(settings)

with st.spinner("Simulating…"):
    t0 = time.perf_counter()
    df_market, firms, df_province = sim_module.simulate_multi(T=cfg.T)
    runtime_s = time.perf_counter() - t0
st.success("Done!")
st.metric("Runtime", f"{runtime_s:.3f} s")

render_market_tabs(df_market)
render_employment(df_market)
render_provinces(df_province)
render_treasury_histogram(firms)

st.header("Market data")
st.dataframe(df_market)
download_csv(df_market, "Download market CSV", "market_timeseries.csv")

st.header("Final firm snapshot")
df_final = firm_snapshot(firms)
if df_final.empty:
    st.write("No firm records.")
else:
    st.dataframe(df_final)
    download_csv(df_final, "Download firm snapshot CSV", "firms_final.csv")