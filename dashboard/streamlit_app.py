"""
Eco-Loop HVAC — Ultra-Modern Physical AI Dashboard

  PYTHONPATH=. streamlit run dashboard/streamlit_app.py

Computes baseline vs AI control metrics dynamically from CSV logs.
Features clean glassmorphism styling, isolated SVG 2D floorplan thermal map,
interactive time-series analytics, and LLM reasoning trace inspector.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import textwrap
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

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
    page_title="Eco-Loop HVAC | Physical AI Dashboard",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-End Glassmorphic Dark Theme CSS (Strictly scoped)
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&family=Outfit:wght@500;600;700;800;900&display=swap');

:root {
  --bg-primary: #0B1320;
  --bg-card: rgba(17, 28, 45, 0.85);
  --border-card: rgba(255, 255, 255, 0.08);
  --accent-emerald: #10B981;
  --accent-cyan: #06B6D4;
  --accent-amber: #F59E0B;
  --accent-coral: #EF4444;
  --text-main: #F8FAFC;
  --text-sub: #94A3B8;
}

.stApp {
  background: radial-gradient(circle at 50% 0%, #152338 0%, #0B1320 70%, #070C14 100%) !important;
  color: var(--text-main) !important;
  font-family: 'Inter', sans-serif !important;
}

.stApp p, .stApp span, .stApp label {
  font-family: 'Inter', sans-serif !important;
}

/* Sidebar styling */
div[data-testid="stSidebar"] {
  background: rgba(11, 19, 32, 0.95) !important;
  border-right: 1px solid var(--border-card) !important;
}
div[data-testid="stSidebar"] * {
  color: #CBD5E1 !important;
}

/* Header styling */
.eco-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1.2rem 1.6rem;
  background: var(--bg-card);
  border: 1px solid var(--border-card);
  border-radius: 1.25rem;
  backdrop-filter: blur(16px);
  margin-bottom: 1.5rem;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
}

.eco-title-group {
  display: flex;
  flex-direction: column;
}

.eco-title {
  font-family: 'Outfit', sans-serif !important;
  font-size: 2.1rem !important;
  font-weight: 800 !important;
  margin: 0 !important;
  background: linear-gradient(135deg, #FFFFFF 0%, #CBD5E1 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  display: flex;
  align-items: center;
  gap: 0.6rem;
}

.eco-badge-row {
  display: flex;
  gap: 0.6rem;
  margin-top: 0.4rem;
  flex-wrap: wrap;
}

.eco-badge {
  font-size: 0.72rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  padding: 0.25rem 0.65rem;
  border-radius: 2rem;
  background: rgba(255, 255, 255, 0.06);
  border: 1px solid rgba(255, 255, 255, 0.1);
  color: var(--text-sub);
}

.eco-badge.active {
  background: rgba(16, 185, 129, 0.15);
  border-color: rgba(16, 185, 129, 0.3);
  color: #34D399;
}

/* Executive Cards */
.eco-kpi-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 1rem;
  margin-bottom: 1.5rem;
}
@media (max-width: 1024px) {
  .eco-kpi-grid { grid-template-columns: repeat(2, 1fr); }
}

.eco-kpi-card {
  background: var(--bg-card);
  border: 1px solid var(--border-card);
  border-radius: 1.15rem;
  padding: 1.35rem 1.4rem;
  position: relative;
  overflow: hidden;
  backdrop-filter: blur(16px);
  transition: all 0.25s ease;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
}
.eco-kpi-card:hover {
  transform: translateY(-2px);
  border-color: rgba(255, 255, 255, 0.18);
  box-shadow: 0 12px 32px rgba(0, 0, 0, 0.35);
}

.eco-kpi-card.save-glow {
  border: 1px solid rgba(16, 185, 129, 0.3);
  background: radial-gradient(circle at top right, rgba(16, 185, 129, 0.15) 0%, rgba(17, 28, 45, 0.8) 70%);
}

.eco-kpi-card.comfort-glow {
  border: 1px solid rgba(6, 182, 212, 0.3);
  background: radial-gradient(circle at top right, rgba(6, 182, 212, 0.15) 0%, rgba(17, 28, 45, 0.8) 70%);
}

.eco-kpi-label {
  font-size: 0.76rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.07em;
  color: var(--text-sub);
  margin-bottom: 0.4rem;
}

.eco-metric-value {
  font-family: 'Outfit', sans-serif !important;
  font-size: 2.5rem;
  font-weight: 800;
  line-height: 1.0;
  margin-bottom: 0.35rem;
}

.eco-kpi-sub {
  font-size: 0.83rem;
  color: var(--text-sub);
}

/* Tradeoff Narrative Banner */
.eco-narrative {
  background: rgba(30, 41, 59, 0.6);
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-left: 4px solid var(--accent-cyan);
  border-radius: 1rem;
  padding: 1.1rem 1.4rem;
  margin-bottom: 1.5rem;
  line-height: 1.6;
  font-size: 0.94rem;
  color: #E2E8F0;
  backdrop-filter: blur(12px);
}

.stTabs [data-baseweb="tab-list"] {
  gap: 0.5rem;
  background: rgba(17, 28, 45, 0.6);
  padding: 0.4rem;
  border-radius: 0.85rem;
  border: 1px solid var(--border-card);
}
.stTabs [data-baseweb="tab"] {
  border-radius: 0.6rem;
  color: var(--text-sub);
  font-weight: 600;
  padding: 0.45rem 1rem;
}
.stTabs [aria-selected="true"] {
  background: rgba(255, 255, 255, 0.08) !important;
  color: var(--text-main) !important;
}

.block-container {
  padding-top: 1.2rem !important;
  max-width: 1240px;
}
</style>
""",
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    st.markdown("### ◈ Eco-Loop HVAC")
    st.caption("Physical AI • Closed-Loop HVAC Control")
    st.markdown("---")

    auto_ref = st.checkbox("Live Auto-Refresh (5s)", value=True)
    if st.button("Recompute Static Charts", use_container_width=True):
        res = _dash_mod.main()
        st.cache_data.clear()
        if res == 0:
            st.success("Charts updated successfully!")
        else:
            st.error(f"Recompute exited with code {res}")

    st.markdown("---")
    st.markdown("**Simulation Meta**")
    st.markdown("""
    - **Engine**: EnergyPlus 25.2
    - **Building**: RefBldg Small Office
    - **Location**: Chicago O'Hare (EPW)
    - **Period**: Jun 15 – Jul 14 (30d)
    - **LLM**: Groq Llama-3.1 Tool-Calling
    """)
    st.markdown("---")
    st.markdown("**Log Sources**")
    st.code(f"{config.BASELINE_CSV.name}\n{config.AI_CSV.name}", language=None)

if auto_ref:
    try:
        from streamlit_autorefresh import st_autorefresh

        st_autorefresh(interval=5000, key="eco_refresh")
    except Exception:
        st.sidebar.info("streamlit-autorefresh recommended for live runs.")


@st.cache_data(ttl=4)
def load_data_pair():
    baseline = _load_sensors(config.BASELINE_CSV)
    ai = _load_sensors(config.AI_CSV)
    metrics = compute_metrics(baseline, ai)
    return baseline, ai, metrics


try:
    baseline_df, ai_df, metrics = load_data_pair()
except Exception as exc:
    st.error(f"Could not load run logs: {exc}")
    st.stop()

# Parse metrics
b_kwh = float(metrics["baseline_kwh"])
a_kwh = float(metrics["ai_kwh"])
sav_kwh = b_kwh - a_kwh
savings_pct = float(metrics["savings_pct"])

occ_stats = metrics.get("ai_comfort_occupied") or metrics["ai_comfort"]
occ_pct = float(occ_stats["pct_in_band"])
base_occ_pct = float((metrics.get("baseline_comfort_occupied") or {}).get("pct_in_band", 0.0))

lo_temp, hi_temp = float(metrics["comfort_lo"]), float(metrics["comfort_hi"])
tradeoff_text = metrics.get("tradeoff_note") or format_tradeoff_note(metrics)
co2_saved_kg = max(0.0, sav_kwh * 0.45)  # Est. ~0.45 kg CO2/kWh Chicago grid

# Header Banner
st.markdown(
    """
<div class="eco-header">
  <div class="eco-title-group">
    <h1 class="eco-title">◈ Eco-Loop Control Center</h1>
    <div class="eco-badge-row">
      <span class="eco-badge active">● Closed-Loop Live</span>
      <span class="eco-badge">EnergyPlus 25.2 API</span>
      <span class="eco-badge">DOE Small Office</span>
      <span class="eco-badge">Chicago Summer 30-Day</span>
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# Executive KPI Grid
save_color = "#10B981" if savings_pct >= 0 else "#EF4444"
save_arrow = "↓" if savings_pct >= 0 else "↑"
save_label = "Energy Saved" if savings_pct >= 0 else "Energy Increase"

st.markdown(
    f"""
<div class="eco-kpi-grid">
  <div class="eco-kpi-card save-glow">
    <div class="eco-kpi-label">{save_label}</div>
    <div class="eco-metric-value" style="color: {save_color};">{save_arrow} {abs(savings_pct):.1f}%</div>
    <div class="eco-kpi-sub">{sav_kwh:+.1f} kWh net delta vs baseline</div>
  </div>
  
  <div class="eco-kpi-card comfort-glow">
    <div class="eco-kpi-label">Occupied Comfort In-Band</div>
    <div class="eco-metric-value" style="color: #38BDF8;">{occ_pct:.1f}%</div>
    <div class="eco-kpi-sub">07:00–19:00 • Band {lo_temp:.0f}–{hi_temp:.0f}°C (Base {base_occ_pct:.1f}%)</div>
  </div>
  
  <div class="eco-kpi-card">
    <div class="eco-kpi-label">Electricity Consumption</div>
    <div class="eco-metric-value" style="color: #F8FAFC;">{a_kwh:,.1f} <span style="font-size:1.1rem; color:#94A3B8;">kWh</span></div>
    <div class="eco-kpi-sub">Baseline: {b_kwh:,.1f} kWh</div>
  </div>
  
  <div class="eco-kpi-card">
    <div class="eco-kpi-label">Est. Carbon Offset</div>
    <div class="eco-metric-value" style="color: #34D399;">-{co2_saved_kg:.1f} <span style="font-size:1.1rem; color:#94A3B8;">kg CO₂</span></div>
    <div class="eco-kpi-sub">Based on regional grid diurnal factor</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# Tradeoff Note
st.markdown(
    f"""
<div class="eco-narrative">
  <strong>💡 Closed-Loop Tradeoff Insight:</strong> {tradeoff_text}
</div>
""",
    unsafe_allow_html=True,
)

# Floorplan Heatmap Component (Interactive Zone State Visualizer)
st.markdown("### 🏢 Building Thermal Floorplan Visualizer")

# Timestep Scrubber for Floorplan
max_steps = min(len(baseline_df), len(ai_df)) - 1
if max_steps > 0:
    selected_step = st.slider(
        "Simulation Timestep Scrubber (0 - 4,319 steps / 30 Days)",
        min_value=0,
        max_value=max_steps,
        value=max_steps,
        step=1,
    )
else:
    selected_step = 0

current_row = ai_df.iloc[selected_step] if len(ai_df) > 0 else None


def get_zone_temp(zone_key: str, default_val: float = 23.0) -> float:
    if current_row is not None and zone_key in current_row and not pd.isna(current_row[zone_key]):
        return float(current_row[zone_key])
    return default_val


t_core = get_zone_temp("zone_temp_c", 23.0)
t_p1 = get_zone_temp("zone_temp_Perimeter_ZN_1", t_core)
t_p2 = get_zone_temp("zone_temp_Perimeter_ZN_2", t_core)
t_p3 = get_zone_temp("zone_temp_Perimeter_ZN_3", t_core)
t_p4 = get_zone_temp("zone_temp_Perimeter_ZN_4", t_core)
t_outdoor = float(current_row["outdoor_temp_c"]) if current_row is not None and "outdoor_temp_c" in current_row else 25.0
ts_str = str(current_row["timestamp"]) if current_row is not None and "timestamp" in current_row else "Step " + str(selected_step)


def temp_to_color(t: float) -> str:
    if t < 21.0:
        return "#06B6D4"  # Cool cyan
    elif 21.0 <= t <= 25.0:
        return "#10B981"  # Optimal emerald
    elif 25.0 < t <= 26.5:
        return "#F59E0B"  # Warm amber
    else:
        return "#EF4444"  # Overheat red


# Render SVG in an isolated Component HTML frame to guarantee clean parsing
svg_component_code = textwrap.dedent(f"""
<!DOCTYPE html>
<html>
<head>
  <style>
    body {{
      margin: 0;
      padding: 0;
      background: transparent;
      font-family: 'Inter', sans-serif;
      color: #F8FAFC;
    }}
    .svg-card {{
      background: rgba(17, 28, 45, 0.9);
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 1rem;
      padding: 1.2rem;
      text-align: center;
      box-sizing: border-box;
    }}
    .header-bar {{
      display: flex;
      justify-content: space-between;
      margin-bottom: 0.8rem;
      font-size: 0.88rem;
      color: #94A3B8;
    }}
    .legend-bar {{
      display: flex;
      justify-content: center;
      gap: 1.2rem;
      margin-top: 0.6rem;
      font-size: 0.78rem;
      color: #94A3B8;
    }}
  </style>
</head>
<body>
  <div class="svg-card">
    <div class="header-bar">
      <span>Timestep: <b style="color:#F8FAFC;">{ts_str}</b></span>
      <span>Outdoor Temp: <b style="color:#F59E0B;">{t_outdoor:.1f}°C</b></span>
    </div>
    <svg viewBox="0 0 500 320" style="width: 100%; max-height: 250px;">
      <!-- Perimeter 1 (North) -->
      <rect x="100" y="20" width="300" height="60" rx="8" fill="{temp_to_color(t_p1)}" fill-opacity="0.3" stroke="{temp_to_color(t_p1)}" stroke-width="2"/>
      <text x="250" y="48" fill="#F8FAFC" font-size="12" font-weight="600" text-anchor="middle">Perimeter North (ZN 1)</text>
      <text x="250" y="66" fill="#CBD5E1" font-size="14" font-weight="700" text-anchor="middle">{t_p1:.1f}°C</text>

      <!-- Perimeter 4 (West) -->
      <rect x="20" y="80" width="80" height="160" rx="8" fill="{temp_to_color(t_p4)}" fill-opacity="0.3" stroke="{temp_to_color(t_p4)}" stroke-width="2"/>
      <text x="60" y="150" fill="#F8FAFC" font-size="11" font-weight="600" text-anchor="middle">West (ZN 4)</text>
      <text x="60" y="170" fill="#CBD5E1" font-size="13" font-weight="700" text-anchor="middle">{t_p4:.1f}°C</text>

      <!-- Core Zone (Center) -->
      <rect x="110" y="90" width="280" height="140" rx="10" fill="{temp_to_color(t_core)}" fill-opacity="0.4" stroke="{temp_to_color(t_core)}" stroke-width="3"/>
      <text x="250" y="150" fill="#FFFFFF" font-size="15" font-weight="800" text-anchor="middle">CORE ZONE (Primary)</text>
      <text x="250" y="175" fill="#FFFFFF" font-size="18" font-weight="800" text-anchor="middle">{t_core:.2f}°C</text>

      <!-- Perimeter 2 (East) -->
      <rect x="400" y="80" width="80" height="160" rx="8" fill="{temp_to_color(t_p2)}" fill-opacity="0.3" stroke="{temp_to_color(t_p2)}" stroke-width="2"/>
      <text x="440" y="150" fill="#F8FAFC" font-size="11" font-weight="600" text-anchor="middle">East (ZN 2)</text>
      <text x="440" y="170" fill="#CBD5E1" font-size="13" font-weight="700" text-anchor="middle">{t_p2:.1f}°C</text>

      <!-- Perimeter 3 (South) -->
      <rect x="100" y="240" width="300" height="60" rx="8" fill="{temp_to_color(t_p3)}" fill-opacity="0.3" stroke="{temp_to_color(t_p3)}" stroke-width="2"/>
      <text x="250" y="268" fill="#F8FAFC" font-size="12" font-weight="600" text-anchor="middle">Perimeter South (ZN 3)</text>
      <text x="250" y="286" fill="#CBD5E1" font-size="14" font-weight="700" text-anchor="middle">{t_p3:.1f}°C</text>
    </svg>
    <div class="legend-bar">
      <span><span style="color:#06B6D4;">■</span> Cool (&lt;21°C)</span>
      <span><span style="color:#10B981;">■</span> Comfort (21–25°C)</span>
      <span><span style="color:#F59E0B;">■</span> Warm (25–26.5°C)</span>
      <span><span style="color:#EF4444;">■</span> High (&gt;26.5°C)</span>
    </div>
  </div>
</body>
</html>
""")

col_svg, col_info = st.columns([1.6, 1])

with col_svg:
    components.html(svg_component_code, height=330)

with col_info:
    st.markdown(
        f"""
        <div style="background: rgba(17, 28, 45, 0.85); border: 1px solid rgba(255,255,255,0.08); border-radius: 1rem; padding: 1.25rem;">
          <h4 style="margin-top:0; font-size:1.05rem; color:#F8FAFC;">Step Snapshot Info</h4>
          <p style="margin-bottom:0.4rem; font-size:0.88rem; color:#CBD5E1;"><b>Timestep</b>: {selected_step} / {max_steps}</p>
          <p style="margin-bottom:0.4rem; font-size:0.88rem; color:#CBD5E1;"><b>Timestamp</b>: {ts_str}</p>
          <p style="margin-bottom:0.4rem; font-size:0.88rem; color:#CBD5E1;"><b>Core Zone Temp</b>: {t_core:.2f}°C</p>
          <p style="margin-bottom:0.4rem; font-size:0.88rem; color:#CBD5E1;"><b>Outdoor Temp</b>: {t_outdoor:.1f}°C</p>
          <hr style="border-color: rgba(255,255,255,0.08); margin: 0.8rem 0;">
          <p style="font-size:0.82rem; color:#94A3B8; margin:0;">
            The AI agent dynamically balances cooling demand across 5 zones using tool calling while respecting comfort guardrails.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# Tabbed Time-Series & Agent Inspector
st.markdown("### 📊 Live Analytics & Agent Reasoning")
tab_energy, tab_temps, tab_demand, tab_agent = st.tabs(
    [
        "⚡ Cumulative Energy (kWh)",
        "🌡️ Multi-Zone Temperatures (°C)",
        "🔌 Electrical Demand (kW)",
        "🧠 LLM Reasoning & Tool Calls",
    ]
)

n_pts = min(len(baseline_df), len(ai_df))

with tab_energy:
    chart_e_df = pd.DataFrame(
        {
            "Baseline": baseline_df["cumulative_kwh"].iloc[:n_pts].to_numpy(),
            "AI Closed-Loop": ai_df["cumulative_kwh"].iloc[:n_pts].to_numpy(),
        }
    )
    st.line_chart(chart_e_df, height=320, color=["#64748B", "#10B981"])
    st.caption("30-Day Cumulative Electricity Consumption (kWh) — Baseline vs AI Agent")

with tab_temps:
    chart_t_df: dict[str, np.ndarray] = {
        "Baseline Core": baseline_df["zone_temp_c"].iloc[:n_pts].to_numpy(),
        "AI Core": ai_df["zone_temp_c"].iloc[:n_pts].to_numpy(),
    }
    if "outdoor_temp_c" in ai_df.columns:
        chart_t_df["Outdoor Temp"] = ai_df["outdoor_temp_c"].iloc[:n_pts].to_numpy()

    st.line_chart(pd.DataFrame(chart_t_df), height=320)
    st.caption(f"Primary Core Zone Air Temperature vs Outdoor EPW — Comfort Band {lo_temp:.0f}–{hi_temp:.0f}°C")

with tab_demand:
    if "electricity_demand_w" in ai_df.columns and "electricity_demand_w" in baseline_df.columns:
        chart_p_df = pd.DataFrame(
            {
                "Baseline kW": baseline_df["electricity_demand_w"].iloc[:n_pts].to_numpy() / 1000.0,
                "AI Agent kW": ai_df["electricity_demand_w"].iloc[:n_pts].to_numpy() / 1000.0,
            }
        )
        st.line_chart(chart_p_df, height=320, color=["#64748B", "#06B6D4"])
        st.caption("Instantaneous HVAC Power Demand Spikes (kW)")
    else:
        st.info("Power demand fields not present in current log schema.")

with tab_agent:
    if config.AI_CSV.is_file():
        raw_ai = pd.read_csv(config.AI_CSV)
        dec_df = (
            raw_ai[raw_ai["event_type"] == "decision"].copy()
            if "event_type" in raw_ai.columns
            else raw_ai.iloc[0:0]
        )

        if len(dec_df) > 0:
            st.markdown(f"**Logged Agent Decisions**: `{len(dec_df)} tool calls executed`")

            show_cols = [c for c in ("timestamp", "action_json", "reasoning") if c in dec_df.columns]
            st.dataframe(
                dec_df[show_cols].tail(20),
                use_container_width=True,
                hide_index=True,
            )

            with st.expander("Inspect Latest Tool-Calling Payload & Reasoning"):
                latest_dec = dec_df.iloc[-1]
                st.markdown(f"**Timestamp**: `{latest_dec.get('timestamp', 'N/A')}`")
                st.markdown("**Executing Action JSON**:")
                try:
                    act_obj = json.loads(str(latest_dec.get("action_json", "{}")))
                    st.json(act_obj)
                except Exception:
                    st.code(str(latest_dec.get("action_json", "")), language="json")

                st.markdown("**LLM Agent Reasoning**:")
                st.info(str(latest_dec.get("reasoning", "No detailed reasoning text logged.")))
        else:
            st.info("No decision tool-call rows logged yet in current run.")
    else:
        st.warning("AI CSV log file not found.")

st.markdown("---")

# Expandable Static Export PNGs and Raw JSON
col_exp1, col_exp2 = st.columns(2)

with col_exp1:
    hero_png = config.DASHBOARD_DIR / "output" / "results_hero.png"
    if hero_png.is_file():
        with st.expander("View Exported High-Res Summary Chart"):
            st.image(str(hero_png), use_container_width=True)

with col_exp2:
    metrics_path = config.DASHBOARD_DIR / "output" / "metrics.json"
    if metrics_path.is_file():
        with st.expander("View Raw metrics.json Schema"):
            st.json(json.loads(metrics_path.read_text()))
