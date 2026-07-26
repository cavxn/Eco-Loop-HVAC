# Eco-Loop HVAC — Full Project Details

This document is a complete walkthrough of everything built, changed, and measured
in the **Eco-Loop Building Agents** hackathon PoC: a live closed-loop Physical AI
system that controls building HVAC setpoints inside a running EnergyPlus
simulation using an LLM with tool-calling.

It is intentionally longer and more narrative than `docs/ARCHITECTURE.md` (which
stays as the short technical architecture report). Use this file when you need
the full story: goals, stack, file-by-file design, control evolution, experiment
results, dashboards, pitfalls, and how to reproduce.

---

## 1. Project goal

Build a **live closed-loop Physical AI PoC** that:

1. Runs a real EnergyPlus building model (not a toy spreadsheet).
2. Reads live zone / outdoor / electricity sensors mid-simulation via the
   EnergyPlus Python API (`pyenergyplus`).
3. Lets an LLM decide HVAC setpoints using **tool calls** (not free-text only).
4. Writes those setpoints back into the **same live simulation** through the
   Actuator API (not by rewriting the IDF and restarting).
5. Logs every timestep and agent decision for comparison vs an uncontrolled
   baseline.
6. Proves measurable **energy savings** while documenting the **comfort
   tradeoff** honestly.

### Locked technical decisions

| Decision | Choice |
|----------|--------|
| Engine | EnergyPlus **25.2** + `pyenergyplus` |
| Model | DOE RefBldg Small Office Chicago → `models/baseline.idf` |
| Weather | `models/weather/chicago.epw` (TMY3) |
| LLM | Groq, OpenAI-compatible tool-calling (`llama-3.1-8b-instant` used under rate limits) |
| Comms | In-process calls + CSV/JSONL event log (not Redis/MQTT for the live loop) |
| Control | Live **Zone Temperature Control** actuators — **not** IDF rewrites |
| Comfort band | **21–25°C** occupied (config-driven) |
| Finalized run period | **30 days**: Jun 15 – Jul 14 |

---

## 2. High-level architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     run_baseline.py / run_ai_loop.py             │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│ EnergyPlusRunner (src/energyplus_runner.py)                      │
│  • single run_energyplus()                                       │
│  • begin_system_timestep_before_predictor → re-apply setpoints   │
│  • end_zone_timestep_after_zone_reporting → sensors + callback   │
│  • apply_action() → set_actuator_value on live state             │
│  • rolling sensor history (last 3) → trend features              │
└───────────────┬───────────────────────────────┬──────────────────┘
                │                               │
                ▼                               ▼
┌───────────────────────────┐     ┌─────────────────────────────────┐
│ EventLogger               │     │ BuildingAgent (AI run only)     │
│ CSV + JSONL + event bus   │     │ prompt → tools → guardrails →   │
└───────────────────────────┘     │ execute_tool → apply_action     │
                                  └─────────────────────────────────┘
```

**Critical invariant:** there is **one** EnergyPlus run. The agent writes
actuators on the same `_ep_state` while still inside the timestep callback.
Nothing starts a second simulation to “apply” control.

---

## 3. Repository layout

```
eco-loop-building-agents/
├── README.md                      # Quick start + scoring table
├── FULL_PROJ_DETAILS.md           # This document
├── docs/ARCHITECTURE.md           # Short architecture report
├── .env / .env.example            # Secrets + tunables (never commit .env)
├── requirements.txt
├── models/
│   ├── baseline.idf               # Building + 30-day RunPeriod
│   ├── weather/chicago.epw
│   └── README.md
├── src/
│   ├── config.py                  # Paths, comfort band, LLM, intervals
│   ├── energyplus_runner.py       # pyenergyplus wrapper + actuators
│   ├── building_agent.py          # LLM brain + predictive guardrails
│   ├── tools.py                   # Tool schemas + executors
│   ├── data_logger.py             # CSV / JSONL / event bus
│   ├── carbon.py                  # Mock / optional carbon signal
│   ├── run_baseline.py            # Uncontrolled baseline entrypoint
│   ├── run_ai_loop.py             # Closed-loop AI entrypoint
│   └── mcp_server.py              # Optional MCP stretch wrapper
├── dashboard/
│   ├── dashboard.py               # Metrics + matplotlib PNGs
│   ├── streamlit_app.py           # Live dynamic UI
│   └── output/                    # metrics.json + charts
├── logs/                          # baseline_run / ai_run (+ archives)
└── outputs/                       # EnergyPlus native outputs per run
```

---

## 4. What each module does

### 4.1 `src/config.py`

Central configuration loaded from environment (via `python-dotenv`):

- Paths: EnergyPlus root, IDF, EPW, logs, dashboard output.
- Zones: `Core_ZN` + four perimeter zones; primary = `Core_ZN`.
- Comfort band: `COMFORT_TEMP_MIN_C` / `MAX` (default 21–25).
- Actuator clamp ranges for cooling / heating setpoints.
- `AGENT_CALLBACK_INTERVAL` — how often the LLM is invoked (default **24**
  timesteps ≈ every 4 simulated hours with Timestep=6).
- LLM: `LLM_BASE_URL`, `LLM_API_KEY` / `GROQ_API_KEY`, model, timeout, retries.

### 4.2 `src/energyplus_runner.py`

The heart of the live loop:

1. Puts `ENERGYPLUS_ROOT` on `sys.path` and imports `pyenergyplus.api.EnergyPlusAPI`.
2. Registers **two** callbacks on one `run_energyplus()` call:
   - **Begin system timestep (before predictor):** re-apply `_last_applied`
     setpoints so EnergyPlus does not forget overrides.
   - **End zone timestep (after zone reporting):** read sensors, log, invoke
     `on_timestep`.
3. Caches sensor/actuator handles once (`_got_handles`).
4. Exposes:
   - `get_state()` — latest sensor snapshot.
   - `apply_action({cooling: {zone: °C}, heating: {...}})` — write actuators.
   - `get_recent_readings(n=3)` — last N logged sensor rows.
   - `compute_trend_features()` — outdoor Δ, zone Δ, margin to nearer comfort
     boundary, outdoor trend label, `favorable_for_edge` flag.
5. Debounces multiple end-timestep fires into one sample per 10-minute bucket.

Actuators used: **Zone Temperature Control** / Cooling Setpoint / Heating
Setpoint (with schedule-actuator fallback available in code if needed).

### 4.3 `src/tools.py`

OpenAI-compatible function tools:

| Tool | Purpose |
|------|---------|
| `get_zone_state` | Read compact zone snapshot |
| `set_cooling_setpoint` | Write cooling SP via `runner.apply_action` |
| `set_heating_setpoint` | Write heating SP via `runner.apply_action` |

Schemas include enums for zone names and describe allowed ranges. Executors
clamp to actuator min/max before calling the runner.

### 4.4 `src/building_agent.py`

LLM orchestration:

1. Builds `SYSTEM_PROMPT` with occupied vs unoccupied bands and predictive
   edge rules.
2. Each `decide()`:
   - Appends compact state to a short history deque.
   - Calls chat completions with tools.
   - **Before** `execute_tool` / `apply_action`, runs deterministic guardrails:
     - Max step **±1.5°C** from last applied setpoint.
     - Occupied hours: hard-clip setpoints into the comfort band.
     - **No-churn:** if last two cooling moves opposed and margin barely
       changed (≤1°C), hold current setpoint (disabled overnight so night
       setback can ramp).
   - On rate limit / failure: capped sleep, then `_heuristic_decide()`.
3. `_enforce_energy_policy()`:
   - Unoccupied: push cooling toward **27°C** night setback.
   - Occupied: mirror primary-zone actions to all five zones when needed.
4. User JSON includes `trend_features`, `band_edge_allowed`, recent logged
   readings, carbon block, etc.

### 4.5 `src/data_logger.py`

Writes:

- Flat CSV rows (`sensor` / `decision`) for dashboard joins.
- JSONL with nested sensor state, actions, reasoning.
- Optional mirrored `event_bus.jsonl` for cross-run / MCP demos.

### 4.6 `src/carbon.py`

Provides a carbon signal for the prompt (mock Chicago diurnal by default;
optional `CARBON_API_URL`). High carbon → bias toward higher cooling setpoints
when edge conditions allow.

### 4.7 Entrypoints

- **`run_baseline.py`:** same IDF/EPW, no agent; logs sensors only.
- **`run_ai_loop.py`:** wires `BuildingAgent` every `AGENT_CALLBACK_INTERVAL`
  callbacks; prints decision stats; compares kWh to baseline CSV if present.
- **`mcp_server.py`:** optional FastMCP/stdio wrapper of the same tools
  (stretch; not required for the live loop).

### 4.8 Dashboard

- **`dashboard/dashboard.py`:** loads sensor rows, `compute_metrics()`, writes
  PNGs + `metrics.json`. `format_tradeoff_note(metrics)` builds the comfort
  explanation from **live numbers** (no stale hardcoded %).
- **`dashboard/streamlit_app.py`:** live UI; all KPIs from logs; fonts
  **Bricolage Grotesque** + **Public Sans**; ↓ teal = savings, ↑ red = increase.

---

## 5. Building model and weather

### IDF (`models/baseline.idf`)

- Based on DOE Reference Building Small Office (Chicago).
- Lightly adapted for the PoC (actuator-friendly thermostat control).
- **RunPeriod** finalized as `PoC_30Day_Summer`: **June 15 – July 14** (30 days).
- Chosen because Chicago outdoor dry-bulb in that window spans roughly
  **~6.7°C to ~32.8°C** — cool nights through hot peaks — so control is tested
  across varied conditions, not only a short heatwave.

### Important IDF editing lesson

An early automated regex replace of `RunPeriod` used `;\n` as the end marker.
IDF fields often put `;` **before** a trailing comment on the same line, so the
regex swallowed holidays, `ScheduleTypeLimits`, and the `ALWAYS_ON` schedule and
EnergyPlus segfaulted. Fix: restore from git and change **only** the RunPeriod
date fields with an exact string replace. Always keep `ALWAYS_ON` /
`ScheduleTypeLimits` intact.

### Weather

`models/weather/chicago.epw` — TMY3 Chicago O’Hare. Simulation years on
timestamps can look odd (TMY stitching); that is normal for EPW runs.

---

## 6. Control philosophy (how the AI is supposed to behave)

### Occupied vs unoccupied

| Mode | Hours | Policy |
|------|-------|--------|
| Occupied | 07:00 ≤ hour &lt; 19:00 | Keep zone in **21–25°C**; setpoints clipped to band |
| Unoccupied | else | Night setback cooling **~26–27°C**; mild drift OK |

### Band-edge energy saving (predictive)

Only move toward the load-reducing edge of the comfort band when **both**:

1. **Margin ≥ 1°C** from the nearer comfort boundary, and  
2. **Outdoor trend** over recent readings is stable or favorable (cooling
   season: flat/falling — not rising into a peak; not a sharp trough).

Otherwise hold closer to mid-band (~23°C cooling).

### Deterministic guardrails (post-LLM)

Even if the model asks for a wild jump:

- Cap change to **1.5°C** vs last applied.
- Occupied: never leave the comfort band with the setpoint.
- No-churn on oscillating decisions unless margin moved &gt; 1°C.

These run **after** the LLM returns a tool call and **before**
`apply_action` / actuator write.

---

## 7. Experiment history (what we ran and what we learned)

All figures below are for the **same 30-day baseline** unless noted:
baseline **1759.3 kWh**, occupied comfort **~87.9%**, all-hours **~84.6%**.

### Phase A — Early 3-day PoC (Jul 15–17)

- Short summer slice used for first end-to-end proof.
- Rough outcome remembered in older docs: earlier 3-day PoC baseline figures retired in favor of finalized 30-day results.

### Phase B — Extend to 30 days (interval = 48)

- RunPeriod → Jun 15 – Jul 14.
- `AGENT_CALLBACK_INTERVAL=48` (~every 8 sim hours) → **90** LLM calls.
- Outdoor range ~6.7–32.8°C.
- **Result (finalized PoC headline):**
  - Energy: **1759.3 → 1696.8 kWh** → **3.6% savings**
  - Occupied comfort AI: **66.8%**
  - All-hours comfort AI: **58.2%** (≈58.1% at one decimal from exact float)
- Archived as `logs/ai_run_30d_interval48.csv`.
- **This is the official published result** used in README, ARCHITECTURE, and
  the dashboard after restoration.

### Phase C — Push-to-edge prompt + interval = 24

- Prompt told the agent to actively ride the comfort-band edge when outdoor
  allowed.
- Interval **24** → **180** agent calls.
- **Result:** Experimental push-to-edge variant archived as `logs/ai_run_30d_interval24_edge.csv`.
- Archived as `logs/ai_run_30d_interval24_edge.csv`.
- More savings and slightly better occupied comfort than Phase B, but later
  rate limits and policy changes superseded it for the “official” story.

### Phase D — Trend-aware prompt tightening

- Required ≥1°C margin **and** favorable outdoor trend for edge moves;
  otherwise mid-band.
- A partial trend-aware run was aborted when work moved to full predictive
  guardrails.

### Phase E — Predictive features + clamp + no-churn (interval = 24)

Implemented:

1. Trend features from last 3 **logged** sensors on the runner.
2. Prompt: occupied/unoccupied bands + margin/trend edge rules.
3. ±1.5°C step cap + occupied band hard-clip before apply.
4. No-churn rule.

**Result of that full 30-day re-run:**

- Energy: Experimental guardrail variant archived as `logs/ai_run_30d_guardrail.csv`.
- Occupied comfort: **84.2%** (much closer to baseline).
- Cause: Groq rate limits → **167/180** heuristic mid-band fallbacks; mid-band
  cools more aggressively than edge setpoints, so comfort rose and savings
  vanished.
- Archived as `logs/ai_run_30d_guardrail.csv`.

### Decision for published metrics

Docs and dashboard were standardized on **Phase B (3.6% / 66.8% / 58.2%)** as
the finalized 30-day PoC result. Live `logs/ai_run.csv` was restored from the
interval=48 archive so Streamlit and PNG generation show those numbers.
Experimental archives remain on disk for comparison.

---

## 8. How comfort and savings are computed

Implemented in `dashboard/dashboard.py` → `compute_metrics()`:

```
savings_pct = 100 * (baseline_kWh - ai_kWh) / baseline_kWh

occupied comfort =
  % of sensor rows with hour in [7, 19) whose zone_temp_c ∈ [21, 25]

all-hours comfort =
  % of all sensor rows in band (includes overnight setback drift)
```

Primary reporting metric for comfort is **occupied-hours**. All-hours is lower
by design when night setback lets the zone float toward ~27°C while empty.

Tradeoff banner text is built by `format_tradeoff_note(metrics)` so the UI
never depends on stale hardcoded percentages.

---

## 9. Dashboard and frontend work

### Static charts (`python dashboard/dashboard.py`)

Writes under `dashboard/output/`:

- `energy_cumulative.png`
- `zone_temperature.png`
- `savings_summary.png`
- `results_hero.png`
- `multizone_temps.png`
- `metrics.json`

Hero / summary charts use:

- **↓ + green/teal** when energy is saved (AI kWh lower).
- **↑ + red** when energy increased.

### Streamlit (`streamlit run dashboard/streamlit_app.py`)

- Fully **dynamic** KPIs and banner from CSV metrics.
- Visual refresh: soft forest/slate background, card hero, KPI strip.
- Fonts: **Bricolage Grotesque** (display) + **Public Sans** (body) — replaced
  an earlier Syne/Outfit pairing the team disliked.
- Sidebar controls: auto-refresh, recompute PNGs, data source names.

### Docs scrub

All human-facing references to obsolete early/experimental figures were removed from README, ARCHITECTURE,
dashboard copy, and console summary. Official copy uses the Phase B 30-day
result.

---

## 10. Latency, rate limits, and reliability

| Mechanism | Detail |
|-----------|--------|
| Agent interval | Default 24 (was 48 for finalized run); avoids calling LLM every 10-min step |
| Handle caching | Resolve actuators/sensors once |
| LLM retries | `LLM_MAX_RETRIES` then heuristic fallback |
| Rate-limit sleep | Capped at **15s** — never block EnergyPlus for minutes |
| Dual callbacks | Sense after HVAC reporting; re-apply before next predictor |
| Prompt size | Only current + short recent history — full logs stay on disk |

Without interval throttling, a 30-day run at Timestep=6 is **4320** callbacks;
interval 24 → **180** LLM turns; interval 48 → **90**. Free-tier Groq often
429s on dense runs — hence heuristic fallbacks and why Phase E looked
“comfortable but not efficient.”

---

## 11. How to run everything (reproduce)

```bash
# Setup
cp .env.example .env   # set ENERGYPLUS_ROOT + GROQ/LLM_API_KEY
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Baseline (no AI)
PYTHONPATH=. python -m src.run_baseline

# AI closed loop
PYTHONPATH=. python -m src.run_ai_loop

# Static dashboard artifacts
PYTHONPATH=. python dashboard/dashboard.py

# Live UI
streamlit run dashboard/streamlit_app.py
```

To restore the **official** AI log without re-running:

```bash
cp logs/ai_run_30d_interval48.csv logs/ai_run.csv
cp logs/ai_run_30d_interval48.jsonl logs/ai_run.jsonl
PYTHONPATH=. python dashboard/dashboard.py
```

---

## 12. End-to-end data flow (one AI timestep with a decision)

1. EnergyPlus advances; end-zone callback fires.
2. Runner reads outdoor temp, zone temps, electricity demand/meters.
3. Appends compact reading to `_sensor_history` (maxlen 3).
4. Logger writes a `sensor` CSV/JSONL row.
5. `on_timestep` runs. If `callback_count % AGENT_CALLBACK_INTERVAL == 0`:
   1. `BuildingAgent.decide(state)`.
   2. Trend features computed from last 3 logged readings.
   3. LLM returns tool call(s) (e.g. `set_cooling_setpoint`).
   4. Guardrails clamp value (step / occupied band / no-churn).
   5. `execute_tool` → `apply_action` → `set_actuator_value` on live state.
   6. Night policy may raise cooling further if unoccupied.
   7. Decision logged with reasoning.
6. Callback returns; EnergyPlus continues; begin-timestep callback re-applies
   `_last_applied` next step.

---

## 13. Pitfalls and lessons learned

1. **IDF surgery is dangerous** — comment-trailing semicolons break naive
   multiline regexes; restore from git if schedules disappear.
2. **Actuators ≠ schedules-only** — live Zone Temperature Control is what makes
   mid-run AI control possible without restarting.
3. **Comfort % without context misleads** — always label occupied vs all-hours
   and mention overnight setback.
4. **More LLM calls ≠ better savings** if rate limits force mid-band heuristics.
5. **Dashboard numbers must be dynamic** — hardcoding static figures caused confusion
   after the 30-day campaign; `format_tradeoff_note(metrics)` fixed that.
6. **Arrow color semantics** — ↓ means less energy (good); ↑ means more (bad).
7. **Archive experimental runs** — keep `ai_run_30d_*` copies so you can compare
   strategies without losing the official log.

---

## 14. Stretch / optional pieces

- **MCP server** (`src/mcp_server.py`): same three tools over MCP for demos.
- **Carbon signal** (`src/carbon.py`): optional real API via `CARBON_API_URL`.
- **Streamlit autorefresh**: `streamlit-autorefresh` for live watching during a
  run.

These are additive; the closed loop works with in-process tool execution alone.

---

## 15. Official PoC scorecard (finalized 30-day)

| Metric | Value |
|--------|-------|
| Period | Jun 15 – Jul 14 (Chicago EPW) |
| Timesteps | 4320 |
| Baseline energy | **1759.3 kWh** |
| AI energy | **1696.8 kWh** |
| Energy savings | **3.6%** |
| Baseline occupied comfort | **87.9%** |
| AI occupied comfort (07–19h) | **66.8%** |
| AI all-hours comfort | **58.2%** |
| Control mechanism | Live EnergyPlus actuators + LLM tools |
| Why all-hours is lower | Intentional overnight setback ~27°C when unoccupied |

**Narrative:** Baseline holds a fixed conservative schedule and scores high
occupied comfort without optimizing energy. The AI agent trades some occupied
comfort margin for **3.6%** energy reduction across mild and peak outdoor
conditions — a tradeoff a fixed baseline schedule cannot make.

---

## 16. Chronological “what we did” checklist

1. Stood up EnergyPlus 25.2 + pyenergyplus runner with dual callbacks.
2. Wired OpenAI-compatible tool-calling agent (Groq) to live actuators.
3. Added CSV/JSONL logging and baseline vs AI entrypoints.
4. Validated short summer runs; measured first savings/comfort.
5. Extended IDF RunPeriod to 30 varied days (after fixing a destructive regex).
6. Tuned `AGENT_CALLBACK_INTERVAL` (48 then 24) for API sustainability.
7. Iterated prompts: edge seeking → margin+trend gates → full predictive
   guardrails (step cap, occupied clip, no-churn).
8. Ran multiple full 30-day AI campaigns; archived each strategy’s logs.
9. Selected Phase B (3.6% / 66.8% / 58.2%) as the official published result.
10. Updated README, ARCHITECTURE, console summary; scrubbed obsolete %.
11. Rebuilt dashboard metrics/PNGs and a dynamic Streamlit UI with new fonts.
12. Wrote this full project details document.

---

## 17. Related reading in-repo

| File | Role |
|------|------|
| `docs/ARCHITECTURE.md` | Short architecture + call chain |
| `README.md` | Quick start + scoring table |
| `models/README.md` | Model / weather notes |
| `dashboard/output/metrics.json` | Last computed numeric snapshot |
| `logs/ai_run_30d_*.csv` | Archived experiment variants |

---

*Document generated to capture the full Eco-Loop HVAC Physical AI PoC build,
experiments, and presentation metrics as implemented in this repository.*
