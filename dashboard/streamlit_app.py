"""
Live-updating Streamlit dashboard for baseline vs AI HVAC control.

  PYTHONPATH=. streamlit run dashboard/streamlit_app.py

All headline numbers are computed from logs — nothing is hardcoded in the UI.
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

_dash_str = str(_DASH_DIR)
while _dash_str in sys.path:
    sys.path.remove(_dash_str)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _load_dashboard_module():
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
format_tradeoff_note = _dash_mod.format_tradeoff_note
ALL_HOURS_COMFORT_CONTEXT = _dash_mod.ALL_HOURS_COMFORT_CONTEXT

from src import config  # noqa: E402

st.set_page_config(
    page_title="Eco-Loop HVAC",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,600;12..96,700;12..96,800&family=Public+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&display=swap');

html, body, [class*="css"], .stMarkdown, .stCaption, p, span, label {
  font-family: 'Public Sans', sans-serif !important;
}
.stApp {
  background:
    linear-gradient(165deg, #f4f7f5 0%, #e8eef0 42%, #dfe8e4 100%);
}
h1, h2, h3, .eco-brand, .eco-card .value, .eco-kpi .v,
div[data-testid="stMarkdownContainer"] h1,
div[data-testid="stMarkdownContainer"] h2,
div[data-testid="stMarkdownContainer"] h3 {
  font-family: 'Bricolage Grotesque', sans-serif !important;
  letter-spacing: -0.03em;
  font-weight: 700 !important;
}
div[data-testid="stSidebar"] {
  background: #14241f !important;
  border-right: 1px solid rgba(255,255,255,0.06);
}
div[data-testid="stSidebar"] * {
  color: #d7e4dd !important;
  font-family: 'Public Sans', sans-serif !important;
}
div[data-testid="stSidebar"] h1,
div[data-testid="stSidebar"] h2,
div[data-testid="stSidebar"] h3 {
  font-family: 'Bricolage Grotesque', sans-serif !important;
  color: #f3faf6 !important;
}
.eco-top {
  display: flex;
  flex-direction: column;
  gap: 0.2rem;
  margin-bottom: 1.35rem;
}
.eco-brand {
  font-size: 2.35rem !important;
  font-weight: 800 !important;
  color: #12241e !important;
  margin: 0 !important;
  line-height: 1.05;
}
.eco-sub {
  margin: 0 !important;
  color: #4a635a !important;
  font-size: 1.02rem !important;
  font-weight: 400 !important;
  max-width: 42rem;
}
.eco-hero {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1rem;
  margin: 0 0 1.1rem 0;
}
@media (max-width: 900px) {
  .eco-hero { grid-template-columns: 1fr; }
}
.eco-card {
  border-radius: 1.25rem;
  padding: 1.45rem 1.55rem 1.35rem;
  position: relative;
  overflow: hidden;
  border: 1px solid rgba(18, 36, 30, 0.08);
  box-shadow: 0 12px 28px rgba(18, 36, 30, 0.07);
  backdrop-filter: blur(6px);
}
.eco-card .value {
  font-size: 3rem;
  font-weight: 800;
  line-height: 0.95;
  margin-bottom: 0.45rem;
}
.eco-card .label {
  font-size: 0.98rem;
  font-weight: 600;
  letter-spacing: 0.01em;
}
.eco-card .sub {
  font-size: 0.88rem;
  opacity: 0.82;
  margin-top: 0.4rem;
  font-weight: 400;
}
.eco-save {
  background: linear-gradient(145deg, #dff5ea 0%, #b8e6d0 100%);
  color: #14532d;
}
.eco-waste {
  background: linear-gradient(145deg, #fde8e8 0%, #f8c9c9 100%);
  color: #7f1d1d;
}
.eco-flat {
  background: linear-gradient(145deg, #e7ece9 0%, #d5ddd8 100%);
  color: #334155;
}
.eco-comfort {
  background: linear-gradient(145deg, #e4eef5 0%, #c5d9e8 100%);
  color: #0c4a6e;
}
.eco-note {
  background: rgba(255,255,255,0.82);
  border: 1px solid rgba(18, 36, 30, 0.1);
  border-left: 3px solid #1f7a55;
  border-radius: 0.85rem;
  padding: 1.05rem 1.2rem;
  color: #1f2d28;
  line-height: 1.6;
  margin: 0.4rem 0 1.15rem 0;
  font-size: 0.95rem;
}
.eco-kpi-row {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 0.75rem;
  margin: 0.85rem 0 1.35rem 0;
}
@media (max-width: 900px) {
  .eco-kpi-row { grid-template-columns: 1fr 1fr; }
}
.eco-kpi {
  background: rgba(255,255,255,0.88);
  border: 1px solid rgba(18, 36, 30, 0.08);
  border-radius: 1rem;
  padding: 0.95rem 0.85rem;
  text-align: center;
  box-shadow: 0 4px 14px rgba(18, 36, 30, 0.04);
}
.eco-kpi .k {
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: #64756e;
  font-weight: 600;
}
.eco-kpi .v {
  font-size: 1.5rem;
  font-weight: 700;
  color: #12241e;
  margin-top: 0.28rem;
}
.block-container { padding-top: 1.5rem !important; max-width: 1180px; }
div[data-testid="stVerticalBlockBorderWrapper"] {
  border-radius: 1rem;
}
</style>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Eco-Loop")
    st.caption("Physical AI · EnergyPlus closed loop")
    auto = st.checkbox("Auto-refresh (5s)", value=True)
    if st.button("Recompute static PNGs", use_container_width=True):
        code = _dash_mod.main()
        st.cache_data.clear()
        if code == 0:
            st.success("Charts updated from current logs")
        else:
            st.warning(f"Dashboard exited with code {code}")
    st.markdown("---")
    st.markdown("**Data sources**")
    st.code(f"{config.BASELINE_CSV.name}\n{config.AI_CSV.name}", language=None)

if auto:
    try:
        from streamlit_autorefresh import st_autorefresh

        st_autorefresh(interval=5000, key="refresh")
    except Exception:
        st.sidebar.info("Install streamlit-autorefresh for live reload.")


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
savings_pct = float(metrics["savings_pct"])
occ = metrics.get("ai_comfort_occupied") or metrics["ai_comfort"]
occ_pct = float(occ["pct_in_band"])
base_occ = float((metrics.get("baseline_comfort_occupied") or {}).get("pct_in_band", 0.0))
all_hours_ai = float(metrics["ai_comfort"]["pct_in_band"])
all_hours_base = float(metrics["baseline_comfort"]["pct_in_band"])
lo, hi = float(metrics["comfort_lo"]), float(metrics["comfort_hi"])
tradeoff = metrics.get("tradeoff_note") or format_tradeoff_note(metrics)

if savings_pct > 0:
    energy_class = "eco-save"
    energy_value = f"↓ {savings_pct:.1f}%"
    energy_label = "energy saved vs baseline"
elif savings_pct < 0:
    energy_class = "eco-waste"
    energy_value = f"↑ {abs(savings_pct):.1f}%"
    energy_label = "energy increase vs baseline"
else:
    energy_class = "eco-flat"
    energy_value = "0.0%"
    energy_label = "no energy change vs baseline"

st.markdown(
    f"""
<div class="eco-top">
  <p class="eco-brand">Eco-Loop</p>
  <p class="eco-sub">Baseline vs LLM-controlled EnergyPlus — metrics computed live from run logs</p>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    f"""
<div class="eco-hero">
  <div class="eco-card {energy_class}">
    <div class="value">{energy_value}</div>
    <div class="label">{energy_label}</div>
    <div class="sub">{baseline_kwh:.1f} → {ai_kwh:.1f} kWh</div>
  </div>
  <div class="eco-card eco-comfort">
    <div class="value">{occ_pct:.1f}%</div>
    <div class="label">occupied-hours in comfort band</div>
    <div class="sub">07–19h · band {lo:.0f}–{hi:.0f}°C · baseline {base_occ:.1f}%</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(f'<div class="eco-note">{tradeoff}</div>', unsafe_allow_html=True)

st.caption(
    f"All-hours in-band (not the primary score): baseline {all_hours_base:.1f}% · "
    f"AI {all_hours_ai:.1f}% — {ALL_HOURS_COMFORT_CONTEXT}."
)

kpi_html = '<div class="eco-kpi-row">'
for label, value in (
    ("Baseline kWh", f"{baseline_kwh:.1f}"),
    ("AI kWh", f"{ai_kwh:.1f}"),
    ("kWh saved", f"{baseline_kwh - ai_kwh:.1f}"),
    ("AI timesteps", f"{len(ai):,}"),
):
    kpi_html += (
        f'<div class="eco-kpi"><div class="k">{label}</div>'
        f'<div class="v">{value}</div></div>'
    )
kpi_html += "</div>"
st.markdown(kpi_html, unsafe_allow_html=True)

st.markdown("### Cumulative energy")
n = min(len(baseline), len(ai))
chart_e = pd.DataFrame(
    {
        "Baseline": baseline["cumulative_kwh"].iloc[:n].to_numpy(),
        "AI": ai["cumulative_kwh"].iloc[:n].to_numpy(),
    }
)
st.line_chart(chart_e, height=300, color=["#1b3a32", "#1f7a55"])

st.markdown("### Zone temperature")
chart_t: dict[str, object] = {
    "Baseline": baseline["zone_temp_c"].iloc[:n].to_numpy(),
    "AI": ai["zone_temp_c"].iloc[:n].to_numpy(),
}
if "outdoor_temp_c" in ai.columns:
    chart_t["Outdoor"] = ai["outdoor_temp_c"].iloc[:n].to_numpy()
st.line_chart(pd.DataFrame(chart_t), height=300)
st.caption(f"Comfort band {lo:.0f}–{hi:.0f}°C")

zone_cols = [c for c in ai.columns if c.startswith("zone_temp_") and c != "zone_temp_c"]
if zone_cols:
    st.markdown("### AI multi-zone temperatures")
    st.line_chart(ai[zone_cols], height=280)

if config.AI_CSV.is_file():
    raw = pd.read_csv(config.AI_CSV)
    dec = (
        raw[raw["event_type"] == "decision"]
        if "event_type" in raw.columns
        else raw.iloc[0:0]
    )
    st.markdown("### Recent agent decisions")
    if len(dec):
        show_cols = [c for c in ("timestamp", "action_json", "reasoning") if c in dec.columns]
        st.dataframe(dec[show_cols].tail(15), use_container_width=True, hide_index=True)
    else:
        st.info("No decision rows yet — AI loop may still be running.")

hero_png = config.DASHBOARD_DIR / "output" / "results_hero.png"
if hero_png.is_file():
    with st.expander("Static hero chart"):
        st.image(str(hero_png), use_container_width=True)

metrics_path = config.DASHBOARD_DIR / "output" / "metrics.json"
if metrics_path.is_file():
    with st.expander("metrics.json"):
        st.json(json.loads(metrics_path.read_text()))
