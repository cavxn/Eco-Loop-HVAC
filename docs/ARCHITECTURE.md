# Eco-Loop System Architecture

Short report of what this repo **actually** implements.

## 1. Tool-calling architecture

Three concerns live in separate modules:

| Concern | Module |
|---------|--------|
| EnergyPlus API wrapper | `src/energyplus_runner.py` |
| LLM orchestration + tools | `src/building_agent.py`, `src/tools.py` |
| Communication bus / logs | `src/data_logger.py` (`EventLogger` → CSV + JSONL + optional `event_bus.jsonl`) |

### Call chain (AI run — one live simulation)

```
run_ai_loop.main()
  └─ EnergyPlusRunner(on_timestep=on_timestep, logger=EventLogger(...))
       └─ runner.run(idf, epw)                    # single run_energyplus()
            ├─ register callback_begin_system_timestep_before_predictor
            │    └─ _on_begin_system_timestep
            │         └─ _write_actuators(_last_applied)   # re-apply setpoints
            │
            └─ register callback_end_zone_timestep_after_zone_reporting
                 └─ _on_end_system_timestep
                      ├─ _read_sensors(state) → _current_state
                      ├─ logger.log_sensor(...)              # bus: sensor row
                      └─ on_timestep(runner)                 # from run_ai_loop
                           ├─ state = runner.get_state()
                           ├─ every N callbacks:
                           │    BuildingAgent(runner).decide(state)
                           │      ├─ OpenAI-compatible chat.completions
                           │      │    with TOOL_SCHEMAS from tools.py
                           │      ├─ parse tool_calls
                           │      └─ execute_tool(runner, name, args)
                           │           └─ set_cooling/heating_setpoint
                           │                └─ runner.apply_action(...)
                           │                     └─ exchange.set_actuator_value(
                           │                          state, handle, °C)
                           └─ runner.log_decision(action, reasoning)  # bus
```

`BuildingAgent` holds the **same** `EnergyPlusRunner` instance that owns the live
`_ep_state`. Actuator writes happen **inside** the EnergyPlus callback stack
before the callback returns and the sim advances. They are **not** applied by
starting a second EnergyPlus run.

Optional stretch: `src/mcp_server.py` exposes the same three tools over MCP
stdio for demos; the live loop does **not** require MCP IPC.

## 2. Prompt engineering strategy

**System prompt** (`SYSTEM_PROMPT` in `building_agent.py`) states:

1. Keep zone air in `COMFORT_TEMP_MIN_C`–`COMFORT_TEMP_MAX_C`
2. Minimize cooling electricity
3. Summer rules: raising cooling SP saves energy; night (hour &lt; 7 or ≥ 19) prefer 26–27°C; multi-zone mirroring; carbon_level high → prefer higher cooling SP
4. Always emit a tool call

**User message** each decide() is compact JSON only:

- `comfort_band_c`, `comfort_status`, `primary_zone`, `zones`
- `carbon` block from `src/carbon.py` (`get_carbon_signal(state)` — mock Chicago diurnal unless `CARBON_API_URL` is set)
- `current`: timestamp, outdoor/zone temps, demand W, cumulative kWh (rounded)
- `recent_readings`: up to the last 2 prior compact snapshots from an in-memory `deque(maxlen=3)` (current is appended first; recent excludes the duplicate current)

Full CSV/JSONL history is **not** pasted into the prompt.

On malformed / rate-limited LLM responses: retry up to `LLM_MAX_RETRIES`, then
`_heuristic_decide()` (deterministic multi-zone setpoints) so the sim never dies.

After a successful LLM action, `_enforce_energy_policy()` may raise night cooling
to 27°C and expand single-zone writes to all five zones.

### Comfort vs energy (how to read the scores)

Primary comfort metric is **occupied-hours** compliance (07–19h within the
configured band). Finalized **30-day** PoC result (Jun 15 – Jul 14, Chicago EPW):

| | Value |
|--|--|
| Energy savings | **3.6%** |
| Occupied-hours comfort (AI) | **66.8%** |
| All-hours comfort (AI) | **58.2%** |
| Baseline / AI energy | **1759.3 kWh** → **1696.8 kWh** |
| Baseline occupied-hours comfort | **87.9%** |

Baseline achieves 87.9% occupied-hours comfort because it holds a fixed
conservative setpoint with no optimization objective; the AI agent achieves
66.8% occupied-hours compliance while reducing energy use by 3.6% over a
30-day period spanning both mild and peak outdoor conditions — a tradeoff the
fixed baseline schedule cannot make. All-hours comfort (58.2%) is lower
because it includes intentional overnight setback to ~27°C during unoccupied
periods, which is by design.

## 3. Latency management

- **Agent interval:** `AGENT_CALLBACK_INTERVAL` (default **24**). With Timestep=6
  (10 simulated minutes), that is roughly every **4 simulated hours**. Calling
  the LLM every timestep would dominate wall-clock time and burn API quota.
- **Handle caching:** on first ready callback, `_cache_handles()` resolves variable,
  meter, and actuator handles once (`_got_handles`); later steps only call
  `get_variable_value` / `set_actuator_value`.
- **Variables requested once** before `run_energyplus` via `request_variable`.
- **Rate limits:** sleep capped at **15s**, then heuristic fallback (never honor a
  multi-minute Retry-After inside the E+ callback).
- **Dual callbacks:** sense/decide after zone reporting (energy reflects HVAC that
  ran); re-apply actuators before the next predictor so the new setpoints affect
  the following HVAC step.

## 4. Handling lengthy simulation logs

| Path | What it holds | Who reads it |
|------|----------------|--------------|
| LLM prompt | Current + ≤2 prior compact readings | `BuildingAgent.decide` |
| `logs/ai_run.csv` / `baseline_run.csv` | Flat sensor + decision rows every timestep | Dashboard |
| `logs/*.jsonl` | Nested `sensor_state`, `action`, `reasoning` | Debug |
| `logs/event_bus.jsonl` | Append-only cross-run bus (incl. MCP actions) | Architecture artifact |

**Why:** prompt size and latency stay bounded; nothing is discarded — the event
logger persists the full mid-run history for charts and review. Truncation is
**only** in the LLM context window, not in storage.
