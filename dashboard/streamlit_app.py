"""
Live-updating Streamlit dashboard for baseline vs AI HVAC control.

  PYTHONPATH=. streamlit run dashboard/streamlit_app.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_ROOT = Path(__file__).resolve().parents[1]
_DASH_DIR = Path(__file__).resolve().parent

# Streamlit puts this file's directory first on sys.path, which makes
# `import dashboard` resolve to a non-package path. Prefer the project root.
_dash_str = str(_DASH_DIR)
while _dash_str in sys.path:
    sys.path.remove(_dash_str)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _load_dashboard_module():
    """Load dashboard/dashboard.py without relying on package import resolution."""
    try:
        from dashboard import dashboard as mod  # type: ignore

        return mod
    except ModuleNotFoundError:
        path = _DASH_DIR / "dashboard.py"
        spec = importlib.util.spec_from_file_location("eco_loop_dashboard", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load {path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod


_dash_mod = _load_dashboard_module()
compute_metrics = _dash_mod.compute_metrics
_load_sensors = _dash_mod._load_sensors
COMFORT_TRADEOFF_NOTE = _dash_mod.COMFORT_TRADEOFF_NOTE
ALL_HOURS_COMFORT_CONTEXT = _dash_mod.ALL_HOURS_COMFORT_CONTEXT

from src import config  # noqa: E402


st.set_page_config(page_title="Eco-Loop HVAC Dashboard", layout="wide")
st.title("Eco-Loop — Physical AI HVAC Control")
st.caption("Baseline vs LLM-controlled EnergyPlus closed loop")

auto = st.sidebar.checkbox("Auto-refresh (5s)", value=True)
if auto:
    try:
        from streamlit_autorefresh import st_autorefresh

        st_autorefresh(interval=5000, key="refresh")
    except Exception:
        st.sidebar.info("Install streamlit-autorefresh for live reload, or click Rerun.")

col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Data sources")
    st.code(f"baseline: {config.BASELINE_CSV.name}\nai:       {config.AI_CSV.name}")
with col_b:
    if st.button("Recompute static PNGs"):
        _dash_mod.main()
        st.success("Charts written to dashboard/output/")
        st.cache_data.clear()


@st.cache_data(ttl=4)
def load_pair():
    baseline = _load_sensors(config.BASELINE_CSV)
    ai = _load_sensors(config.AI_CSV)
    metrics = compute_metrics(baseline, ai)
    return baseline, ai, metrics


try:
    baseline, ai, metrics = load_pair()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

baseline_kwh = float(metrics["baseline_kwh"])
ai_kwh = float(metrics["ai_kwh"])
# Positive => AI used less energy than baseline (good).
pct_change = (
    (baseline_kwh - ai_kwh) / baseline_kwh * 100.0 if baseline_kwh > 0 else 0.0
)
occ = metrics.get("ai_comfort_occupied") or metrics["ai_comfort"]
occ_pct = float(occ["pct_in_band"])
lo, hi = metrics["comfort_lo"], metrics["comfort_hi"]

# ---- Hero strip (10-second scan) ----
if pct_change > 0:
    energy_md = (
        f"<div style='background:#c6f6d5;color:#276749;padding:1.2rem;border-radius:0.6rem;"
        f"text-align:center'><div style='font-size:2.8rem;font-weight:800'>↓ {pct_change:.1f}%</div>"
        f"<div style='font-size:1.1rem;font-weight:600'>energy saved vs baseline</div>"
        f"<div style='font-size:0.9rem;opacity:0.85'>{baseline_kwh:.1f} → {ai_kwh:.1f} kWh</div></div>"
    )
elif pct_change < 0:
    energy_md = (
        f"<div style='background:#fed7d7;color:#c53030;padding:1.2rem;border-radius:0.6rem;"
        f"text-align:center'><div style='font-size:2.8rem;font-weight:800'>↑ {abs(pct_change):.1f}%</div>"
        f"<div style='font-size:1.1rem;font-weight:600'>energy increase vs baseline</div>"
        f"<div style='font-size:0.9rem;opacity:0.85'>{baseline_kwh:.1f} → {ai_kwh:.1f} kWh</div></div>"
    )
else:
    energy_md = (
        "<div style='background:#e2e8f0;color:#4a5568;padding:1.2rem;border-radius:0.6rem;"
        "text-align:center'><div style='font-size:2.8rem;font-weight:800'>0.0%</div>"
        "<div style='font-size:1.1rem;font-weight:600'>no energy change vs baseline</div></div>"
    )

comfort_md = (
    f"<div style='background:#bee3f8;color:#2b6cb0;padding:1.2rem;border-radius:0.6rem;"
    f"text-align:center'><div style='font-size:2.8rem;font-weight:800'>{occ_pct:.0f}%</div>"
    f"<div style='font-size:1.1rem;font-weight:600'>% of occupied time within comfort band</div>"
    f"<div style='font-size:0.9rem;opacity:0.85'>Occupied hours 07–19 · band {lo:.0f}–{hi:.0f}°C"
    f" · baseline occupied {float((metrics.get('baseline_comfort_occupied') or {}).get('pct_in_band', 100)):.0f}%"
    f"</div></div>"
)

h1, h2 = st.columns(2)
with h1:
    st.markdown(energy_md, unsafe_allow_html=True)
with h2:
    st.markdown(comfort_md, unsafe_allow_html=True)

st.info(COMFORT_TRADEOFF_NOTE)

all_hours_ai = float(metrics["ai_comfort"]["pct_in_band"])
all_hours_base = float(metrics["baseline_comfort"]["pct_in_band"])
st.caption(
    f"All-hours in-band (not the primary score): baseline {all_hours_base:.0f}% · "
    f"AI {all_hours_ai:.0f}% — {ALL_HOURS_COMFORT_CONTEXT}."
)

st.divider()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Baseline kWh", f"{baseline_kwh:.1f}")
m2.metric("AI kWh", f"{ai_kwh:.1f}")
m3.metric("kWh saved", f"{baseline_kwh - ai_kwh:.1f}")
m4.metric("AI timesteps logged", f"{len(ai)}")


st.subheader("Cumulative energy (kWh)")
n = min(len(baseline), len(ai))
chart_e = pd.DataFrame(
    {
        "baseline": baseline["cumulative_kwh"].iloc[:n].to_numpy(),
        "ai": ai["cumulative_kwh"].iloc[:n].to_numpy(),
    }
)
st.line_chart(chart_e)

st.subheader("Zone temperature (°C)")
chart_t = {
    "baseline": baseline["zone_temp_c"].iloc[:n].to_numpy(),
    "ai": ai["zone_temp_c"].iloc[:n].to_numpy(),
}
if "outdoor_temp_c" in ai.columns:
    chart_t["outdoor"] = ai["outdoor_temp_c"].iloc[:n].to_numpy()
st.line_chart(pd.DataFrame(chart_t))
st.caption(f"Comfort band {metrics['comfort_lo']}–{metrics['comfort_hi']}°C")

zone_cols = [c for c in ai.columns if c.startswith("zone_temp_") and c != "zone_temp_c"]
if zone_cols:
    st.subheader("AI multi-zone temperatures")
    st.line_chart(ai[zone_cols])

if config.AI_CSV.is_file():
    raw = pd.read_csv(config.AI_CSV)
    dec = (
        raw[raw["event_type"] == "decision"]
        if "event_type" in raw.columns
        else raw.iloc[0:0]
    )
    st.subheader("Recent agent decisions")
    if len(dec):
        st.dataframe(
            dec[["timestamp", "action_json", "reasoning"]].tail(15),
            use_container_width=True,
        )
    else:
        st.info("No decision rows yet — AI loop may still be running.")

metrics_path = config.DASHBOARD_DIR / "output" / "metrics.json"
if metrics_path.is_file():
    with st.expander("metrics.json"):
        st.json(json.loads(metrics_path.read_text()))
