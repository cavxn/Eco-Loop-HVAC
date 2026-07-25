"""
BuildingAgent — LLM tool-calling brain for closed-loop HVAC setpoint control.

Talks to any OpenAI-compatible endpoint (Groq / Together / Fireworks). On failure
or malformed tool calls: retry once, then no-op so the simulation never crashes.
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from openai import OpenAI, RateLimitError, APIStatusError

from src import config
from src.carbon import get_carbon_signal
from src.tools import TOOL_SCHEMAS, execute_tool

if TYPE_CHECKING:
    from src.energyplus_runner import EnergyPlusRunner


SYSTEM_PROMPT = f"""You are an autonomous HVAC control agent for a small office building
simulated in EnergyPlus (summer, cooling-dominated). Objectives, in priority order:

1. Keep zone air temperature inside the applicable comfort band for the hour.
2. Minimize cooling electricity.

Occupied vs unoccupied bands (formal day/night split):
- OCCUPIED hours (07:00 ≤ hour < 19:00): tighter comfort band
  {config.COMFORT_TEMP_MIN_C}–{config.COMFORT_TEMP_MAX_C} °C. Setpoints must stay
  inside this band. Prefer energy-saving edges only when predictive features allow.
- UNOCCUPIED hours (hour < 7 or hour ≥ 19): looser band — raise cooling to
  26–27°C (night setback) to avoid unnecessary overnight cooling. Mild zone
  drift above {config.COMFORT_TEMP_MAX_C}°C is acceptable when unoccupied.

Predictive + guardrail policy (use `trend_features` in the user JSON):
- Move a setpoint toward the load-reducing comfort-band edge ONLY if BOTH:
  (a) margin_to_nearer_boundary_c ≥ 1.0°C, AND
  (b) outdoor_trend is stable or favorable (cooling season: flat or falling,
      not rising / not sharply approaching a peak or trough).
- Otherwise hold a more conservative setpoint closer to the CENTER of the
  occupied band (~23°C cooling).
- When (a)+(b) hold: warmer cooling SP (toward {config.COMFORT_TEMP_MAX_C}°C)
  when cooling; cooler heating SP (toward {config.COMFORT_TEMP_MIN_C}°C) when heating.
- Raising cooling saves energy; lowering it wastes energy. Do NOT cool to 22–23°C
  in occupied hours unless the zone is clearly above the occupied band.
- Heating setpoint: keep near {config.COMFORT_TEMP_MIN_C}°C in summer (deadband).
- Always issue a tool call. Prefer set_cooling_setpoint on {config.PRIMARY_ZONE}.
- Multi-zone: after setting Core_ZN, also set Perimeter_ZN_1..4 to the SAME
  cooling setpoint unless a perimeter zone is >0.8°C hotter than Core (then
  give that zone 0.5°C lower cooling SP).
- When carbon_level is "high", bias toward the highest acceptable cooling SP
  only if (a)+(b) still hold; otherwise stay conservative / mid-band.
- One short sentence of reasoning max.

Hard bounds (actuators):
- Cooling setpoints: [{config.COOLING_SETPOINT_MIN_C}, {config.COOLING_SETPOINT_MAX_C}] °C
- Heating setpoints: [{config.HEATING_SETPOINT_MIN_C}, {config.HEATING_SETPOINT_MAX_C}] °C
"""

# Deterministic post-LLM guardrails
MAX_SETPOINT_STEP_C = 1.5
EDGE_MARGIN_MIN_C = 1.0
NO_CHURN_MARGIN_DELTA_C = 1.0
OCCUPIED_HOUR_START = 7
OCCUPIED_HOUR_END = 19


@dataclass
class AgentDecision:
    """Result of one agent turn, ready for EventLogger.log_decision."""

    action: dict[str, Any] | None
    reasoning: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    noop: bool = False
    error: str | None = None

    def as_log_action(self) -> dict[str, Any] | None:
        if self.noop and not self.action:
            return {"noop": True, "error": self.error} if self.error else {"noop": True}
        return self.action


class BuildingAgent:
    """LLM orchestrator that turns sensor state into EnergyPlus actuator writes."""

    def __init__(
        self,
        runner: EnergyPlusRunner,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        history_len: int = 3,
        timeout_s: float | None = None,
    ) -> None:
        self.runner = runner
        self.api_key = api_key if api_key is not None else config.LLM_API_KEY
        self.base_url = base_url or config.LLM_BASE_URL
        self.model = model or config.LLM_MODEL
        self.timeout_s = timeout_s if timeout_s is not None else config.LLM_TIMEOUT_S
        self._history: deque[dict[str, Any]] = deque(maxlen=history_len)
        self._client: OpenAI | None = None
        # No-churn: signed cooling deltas from the last two decisions + margin.
        self._decision_dirs: deque[float] = deque(maxlen=2)
        self._last_decision_margin_c: float | None = None
        self._guardrail_notes: list[str] = []

        if not self.api_key:
            raise ValueError(
                "No LLM API key found. Set LLM_API_KEY or GROQ_API_KEY in .env "
                "(see .env.example)."
            )

        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout_s,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decide(self, state: dict[str, Any] | None = None) -> AgentDecision:
        """
        One agent turn: prompt → tool call(s) → guardrail clamp → execute.

        Retries once on malformed / empty tool responses, then falls back to a
        deterministic energy-saving heuristic (never blocks the sim for minutes).
        """
        state = state or self.runner.get_state()
        self._history.append(self._compact_state(state))
        self._guardrail_notes = []
        self._pre_decision_cooling = self._last_applied_setpoint(
            "cooling", config.PRIMARY_ZONE
        )

        last_error: str | None = None
        attempts = 1 + max(0, config.LLM_MAX_RETRIES)
        rate_limited = False

        for attempt in range(attempts):
            try:
                decision = self._call_llm_and_execute(state, attempt=attempt)
                if decision.tool_calls or decision.action:
                    decision = self._enforce_energy_policy(state, decision)
                    self._record_decision_churn(state, decision)
                    return decision
                last_error = decision.error or "empty tool call"
            except RateLimitError as exc:
                rate_limited = True
                last_error = f"RateLimitError: {exc}"
                # Cap sleep — never stall the EnergyPlus callback for minutes.
                wait_s = min(15.0, 5.0 * (attempt + 1))
                print(f"[BuildingAgent] rate limited — sleeping {wait_s:.1f}s then fallback")
                time.sleep(wait_s)
            except APIStatusError as exc:
                last_error = f"APIStatusError: {exc}"
                if getattr(exc, "status_code", None) == 429:
                    rate_limited = True
                    time.sleep(min(15.0, 5.0 * (attempt + 1)))
                else:
                    break
            except Exception as exc:  # noqa: BLE001 — never crash the sim
                last_error = f"{type(exc).__name__}: {exc}"

        # Keep the closed loop alive with a safe energy-saving policy.
        fallback = self._heuristic_decide(state)
        fallback.error = last_error
        fallback.reasoning = (
            f"Heuristic fallback ({'rate-limit' if rate_limited else 'llm-error'}): "
            + fallback.reasoning
        )
        self._record_decision_churn(state, fallback)
        return fallback

    def _is_occupied(self, state: dict[str, Any]) -> bool:
        hour = self._sim_hour(state)
        return OCCUPIED_HOUR_START <= hour < OCCUPIED_HOUR_END

    def _last_applied_setpoint(self, kind: str, zone: str) -> float | None:
        applied = self.runner._last_applied or {}  # noqa: SLF001 — intentional
        bucket = applied.get(kind) or {}
        if isinstance(bucket, dict) and zone in bucket:
            try:
                return float(bucket[zone])
            except (TypeError, ValueError):
                return None
        # Flat fallback
        if kind == "cooling" and "cooling_setpoint" in applied:
            try:
                return float(applied["cooling_setpoint"])
            except (TypeError, ValueError):
                return None
        return None

    def _current_margin_c(self, state: dict[str, Any]) -> float | None:
        features = self.runner.compute_trend_features(3)
        m = features.get("margin_to_nearer_boundary_c")
        if isinstance(m, (int, float)):
            return float(m)
        zt = state.get("zone_temp_c")
        if isinstance(zt, (int, float)):
            lo, hi = config.COMFORT_TEMP_MIN_C, config.COMFORT_TEMP_MAX_C
            return min(float(zt) - lo, hi - float(zt))
        return None

    def _no_churn_hold(self, state: dict[str, Any]) -> bool:
        """Hold if last two moves opposed and margin hasn't shifted much.

        Disabled during unoccupied hours so night setback can still ramp.
        """
        if not self._is_occupied(state):
            return False
        if len(self._decision_dirs) < 2:
            return False
        d0, d1 = self._decision_dirs[0], self._decision_dirs[1]
        if d0 == 0 or d1 == 0 or d0 * d1 >= 0:
            return False
        margin = self._current_margin_c(state)
        if margin is None or self._last_decision_margin_c is None:
            return True
        return abs(margin - self._last_decision_margin_c) <= NO_CHURN_MARGIN_DELTA_C

    def _clamp_setpoint_value(
        self,
        kind: str,
        zone: str,
        requested: float,
        state: dict[str, Any],
        *,
        allow_unoccupied_setback: bool = True,
    ) -> float:
        """
        Deterministic clamp before apply_action:
          - max ±1.5°C from last applied setpoint for this zone/kind
          - occupied hours: hard-clip into comfort band
          - always respect actuator min/max
          - no-churn: hold last applied when opposite oscillation
        """
        notes: list[str] = []
        occupied = self._is_occupied(state)
        last = self._last_applied_setpoint(kind, zone)
        value = float(requested)

        if self._no_churn_hold(state) and last is not None:
            notes.append(f"no-churn hold {kind}/{zone}={last:.1f}")
            value = last
        else:
            if last is not None:
                lo_step = last - MAX_SETPOINT_STEP_C
                hi_step = last + MAX_SETPOINT_STEP_C
                clamped = max(lo_step, min(hi_step, value))
                if clamped != value:
                    notes.append(
                        f"step-cap {kind}/{zone} {value:.1f}→{clamped:.1f} (Δ≤{MAX_SETPOINT_STEP_C})"
                    )
                value = clamped

            if occupied:
                band_lo, band_hi = config.COMFORT_TEMP_MIN_C, config.COMFORT_TEMP_MAX_C
                clipped = max(band_lo, min(band_hi, value))
                if clipped != value:
                    notes.append(
                        f"occupied-band clip {kind}/{zone} {value:.1f}→{clipped:.1f}"
                    )
                value = clipped
            elif (
                allow_unoccupied_setback
                and kind == "cooling"
                and value < 26.0
                and (last is None or last < 26.0)
            ):
                # Soft preference only — night policy may raise further later.
                pass

        if kind == "cooling":
            value = max(config.COOLING_SETPOINT_MIN_C, min(config.COOLING_SETPOINT_MAX_C, value))
        else:
            value = max(config.HEATING_SETPOINT_MIN_C, min(config.HEATING_SETPOINT_MAX_C, value))

        self._guardrail_notes.extend(notes)
        return round(value, 2)

    def _guardrail_tool_args(
        self, name: str, args: dict[str, Any], state: dict[str, Any]
    ) -> dict[str, Any]:
        """Rewrite setpoint tool args through deterministic clamps before apply."""
        if name not in ("set_cooling_setpoint", "set_heating_setpoint"):
            return args
        out = dict(args)
        zone = out.get("zone_name") or config.PRIMARY_ZONE
        try:
            raw = float(out.get("value_celsius"))
        except (TypeError, ValueError):
            return out
        kind = "cooling" if name == "set_cooling_setpoint" else "heating"
        out["value_celsius"] = self._clamp_setpoint_value(kind, zone, raw, state)
        return out

    def _record_decision_churn(self, state: dict[str, Any], decision: AgentDecision) -> None:
        """Track signed primary-zone cooling moves for the no-churn rule."""
        action = decision.action or {}
        cooling = action.get("cooling") or {}
        zone = config.PRIMARY_ZONE
        new_val = cooling.get(zone)
        if new_val is None and isinstance(cooling, dict) and cooling:
            new_val = next(iter(cooling.values()))
        prior = getattr(self, "_pre_decision_cooling", None)
        if prior is not None and new_val is not None:
            direction = float(new_val) - float(prior)
        else:
            direction = 0.0
        self._decision_dirs.append(direction)
        self._last_decision_margin_c = self._current_margin_c(state)

    def _sim_hour(self, state: dict[str, Any]) -> int:
        hour = (state.get("sim_time") or {}).get("hour")
        if hour is None and state.get("timestamp"):
            try:
                hour = int(str(state["timestamp"]).split()[1].split(":")[0])
            except Exception:
                hour = 12
        try:
            return int(hour)
        except Exception:
            return 12

    def _enforce_energy_policy(
        self, state: dict[str, Any], decision: AgentDecision
    ) -> AgentDecision:
        """
        Guardrails after the LLM acts:
        - Night: force cooling ≥ 26.5°C (baseline setback is ~30°C; don't over-cool).
        - Expand single-zone actions to all zones.
        - All writes go through clamp → apply.
        """
        hour = self._sim_hour(state)
        action = dict(decision.action or {})
        cooling = dict(action.get("cooling") or {})
        heating = dict(action.get("heating") or {})
        occupied = self._is_occupied(state)

        # Night setback — critical for beating baseline energy.
        if not occupied:
            target = 27.0
            changed = False
            for z in config.ZONES:
                cur = cooling.get(z)
                if cur is None or float(cur) < target:
                    cooling[z] = target
                    changed = True
            if changed:
                for z, val in list(cooling.items()):
                    clamped = self._clamp_setpoint_value(
                        "cooling", z, float(val), state, allow_unoccupied_setback=True
                    )
                    cooling[z] = clamped
                    execute_tool(
                        self.runner,
                        "set_cooling_setpoint",
                        {"zone_name": z, "value_celsius": clamped},
                    )
                decision.reasoning = (
                    (decision.reasoning or "") + f" | night setback→{target}°C"
                ).strip(" |")
        else:
            # Daytime: if only primary zone set, mirror to all zones.
            if cooling and config.PRIMARY_ZONE in cooling and len(cooling) == 1:
                primary_val = float(cooling[config.PRIMARY_ZONE])
                for z in config.ZONES:
                    clamped = self._clamp_setpoint_value("cooling", z, primary_val, state)
                    cooling[z] = clamped
                    execute_tool(
                        self.runner,
                        "set_cooling_setpoint",
                        {"zone_name": z, "value_celsius": clamped},
                    )
            if heating and config.PRIMARY_ZONE in heating and len(heating) == 1:
                primary_val = float(heating[config.PRIMARY_ZONE])
                for z in config.ZONES:
                    clamped = self._clamp_setpoint_value("heating", z, primary_val, state)
                    heating[z] = clamped
                    execute_tool(
                        self.runner,
                        "set_heating_setpoint",
                        {"zone_name": z, "value_celsius": clamped},
                    )

        if self._guardrail_notes:
            uniq = []
            for n in self._guardrail_notes:
                if n not in uniq:
                    uniq.append(n)
            decision.reasoning = (
                (decision.reasoning or "") + " | guardrails: " + "; ".join(uniq[:4])
            ).strip(" |")

        if cooling:
            action["cooling"] = cooling
        if heating:
            action["heating"] = heating
        decision.action = action or None
        decision.noop = not bool(action)
        return decision

    def _heuristic_decide(self, state: dict[str, Any]) -> AgentDecision:
        """Deterministic summer policy: raise cooling SP on ALL zones to save energy."""
        zt = state.get("zone_temp_c")
        hour = self._sim_hour(state)
        features = self.runner.compute_trend_features(3)
        margin = features.get("margin_to_nearer_boundary_c")
        favorable = bool(features.get("favorable_for_edge"))

        carbon = get_carbon_signal(state)
        if not self._is_occupied(state):
            cool_sp = min(config.COOLING_SETPOINT_MAX_C, 27.0)
            reason = f"Night setback cooling={cool_sp}°C"
        elif isinstance(zt, (int, float)) and zt > config.COMFORT_TEMP_MAX_C + 0.3:
            cool_sp = max(config.COOLING_SETPOINT_MIN_C, config.COMFORT_TEMP_MAX_C - 0.5)
            reason = f"Zone warm ({zt:.1f}°C) — cooling={cool_sp}°C"
        elif (
            isinstance(margin, (int, float))
            and margin >= EDGE_MARGIN_MIN_C
            and favorable
        ):
            cool_sp = config.COMFORT_TEMP_MAX_C
            if carbon.level == "high":
                cool_sp = min(config.COOLING_SETPOINT_MAX_C, cool_sp + 0.5)
            reason = f"Edge OK (margin={margin:.1f}, trend={features.get('outdoor_trend')})"
        else:
            # Conservative mid-band when margin/trend not favorable.
            cool_sp = 0.5 * (config.COMFORT_TEMP_MIN_C + config.COMFORT_TEMP_MAX_C)
            reason = (
                f"Hold mid-band cooling={cool_sp}°C "
                f"(margin={margin}, trend={features.get('outdoor_trend')})"
            )

        heat_sp = config.COMFORT_TEMP_MIN_C
        results = []
        tool_calls = []
        cooling_map: dict[str, float] = {}
        heating_map: dict[str, float] = {}
        zone_temps = state.get("zone_temps_c") or {}
        core_t = zone_temps.get(config.PRIMARY_ZONE, zt)

        for zone in config.ZONES:
            z_cool = cool_sp
            zt_z = zone_temps.get(zone)
            if (
                isinstance(core_t, (int, float))
                and isinstance(zt_z, (int, float))
                and zt_z > core_t + 0.8
            ):
                z_cool = max(config.COOLING_SETPOINT_MIN_C, cool_sp - 0.5)
            z_cool = self._clamp_setpoint_value("cooling", zone, z_cool, state)
            z_heat = self._clamp_setpoint_value("heating", zone, heat_sp, state)
            results.append(
                execute_tool(
                    self.runner,
                    "set_cooling_setpoint",
                    {"zone_name": zone, "value_celsius": z_cool},
                )
            )
            results.append(
                execute_tool(
                    self.runner,
                    "set_heating_setpoint",
                    {"zone_name": zone, "value_celsius": z_heat},
                )
            )
            tool_calls.append(
                {"name": "set_cooling_setpoint", "arguments": {"zone_name": zone, "value_celsius": z_cool}}
            )
            tool_calls.append(
                {"name": "set_heating_setpoint", "arguments": {"zone_name": zone, "value_celsius": z_heat}}
            )
            cooling_map[zone] = z_cool
            heating_map[zone] = z_heat

        return AgentDecision(
            action={"cooling": cooling_map, "heating": heating_map},
            reasoning=reason + f" multi-zone x{len(config.ZONES)}",
            tool_calls=tool_calls,
            tool_results=results,
            noop=False,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _compact_state(self, state: dict[str, Any]) -> dict[str, Any]:
        zone_temps = state.get("zone_temps_c") or {}
        return {
            "timestamp": state.get("timestamp"),
            "outdoor_temp_c": _round(state.get("outdoor_temp_c")),
            "zone_temp_c": _round(state.get("zone_temp_c")),
            "zone_temps_c": {z: _round(t) for z, t in zone_temps.items()},
            "electricity_demand_w": _round(state.get("electricity_demand_w"), 0),
            "cumulative_kwh": _round(state.get("cumulative_kwh"), 2),
        }

    def _build_user_message(self, state: dict[str, Any]) -> str:
        current = self._compact_state(state)
        history = list(self._history)
        recent = history[:-1] if len(history) > 1 else []

        trend_features = self.runner.compute_trend_features(3)
        lo, hi = config.COMFORT_TEMP_MIN_C, config.COMFORT_TEMP_MAX_C
        margin_c = trend_features.get("margin_to_nearer_boundary_c")
        occupied = self._is_occupied(state)

        comfort_status = "unknown"
        zt = current.get("zone_temp_c")
        if isinstance(zt, (int, float)):
            if zt < lo:
                comfort_status = "BELOW occupied comfort band"
            elif zt > hi:
                comfort_status = (
                    "ABOVE occupied comfort band"
                    if occupied
                    else "Above occupied band (unoccupied — setback OK)"
                )
            elif isinstance(margin_c, (int, float)) and margin_c >= EDGE_MARGIN_MIN_C:
                comfort_status = (
                    "INSIDE band with ≥1°C margin — edge OK only if outdoor trend favorable"
                )
            else:
                comfort_status = (
                    "INSIDE band but <1°C margin — hold conservative mid-band setpoint"
                )

        carbon = get_carbon_signal(state)
        band_edge_allowed = bool(
            isinstance(margin_c, (int, float))
            and margin_c >= EDGE_MARGIN_MIN_C
            and trend_features.get("favorable_for_edge")
        )

        payload = {
            "objective": "minimize energy (and carbon) while respecting occupied/unoccupied bands",
            "hour_mode": "occupied" if occupied else "unoccupied",
            "occupied_comfort_band_c": [lo, hi],
            "unoccupied_cooling_setback_c": [26.0, 27.0],
            "comfort_status": comfort_status,
            "trend_features": trend_features,
            "band_edge_allowed": band_edge_allowed,
            "recent_logged_readings": self.runner.get_recent_readings(3),
            "primary_zone": config.PRIMARY_ZONE,
            "zones": list(config.ZONES),
            "carbon": carbon.as_prompt_block(),
            "current": current,
            "recent_agent_readings": recent,
            "no_churn_hold_active": self._no_churn_hold(state),
        }
        return (
            "Current building state (JSON). Decide the next HVAC setpoints "
            "using the available tools. Apply setpoints to all zones when practical.\n"
            + json.dumps(payload, indent=2)
        )

    def _call_llm_and_execute(
        self, state: dict[str, Any], *, attempt: int
    ) -> AgentDecision:
        assert self._client is not None

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self._build_user_message(state)},
        ]
        if attempt > 0:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Previous response was missing a valid tool call. "
                        "You MUST call set_cooling_setpoint or set_heating_setpoint now."
                    ),
                }
            )

        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            temperature=0.2,
            max_tokens=400,
        )

        choice = response.choices[0].message
        reasoning = (choice.content or "").strip()
        raw_tool_calls = list(choice.tool_calls or [])

        if not raw_tool_calls:
            return AgentDecision(
                action=None,
                reasoning=reasoning or "Model returned no tool call",
                noop=True,
                error="no_tool_calls",
            )

        parsed_calls: list[dict[str, Any]] = []
        results: list[dict[str, Any]] = []
        aggregated_action: dict[str, Any] = {}

        for tc in raw_tool_calls:
            name = tc.function.name
            args_raw = tc.function.arguments
            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else dict(args_raw or {})
            except json.JSONDecodeError:
                args = {}
                result = {"ok": False, "error": f"Malformed arguments: {args_raw!r}"}
                parsed_calls.append({"name": name, "arguments": args_raw})
                results.append(result)
                continue

            # Deterministic clamp / no-churn BEFORE apply_action.
            if name in ("set_cooling_setpoint", "set_heating_setpoint"):
                args = self._guardrail_tool_args(name, args, state)

            parsed_calls.append({"name": name, "arguments": args})
            result = execute_tool(self.runner, name, args)
            results.append(result)

            # Fold setpoint writes into a single action dict for the event log.
            if name == "set_cooling_setpoint" and result.get("ok"):
                aggregated_action.setdefault("cooling", {})[result["zone_name"]] = result[
                    "value_celsius"
                ]
            elif name == "set_heating_setpoint" and result.get("ok"):
                aggregated_action.setdefault("heating", {})[result["zone_name"]] = result[
                    "value_celsius"
                ]

        any_ok = any(r.get("ok") for r in results)
        if not any_ok and not aggregated_action:
            return AgentDecision(
                action=None,
                reasoning=reasoning or "All tool executions failed",
                tool_calls=parsed_calls,
                tool_results=results,
                noop=True,
                error="tool_execution_failed",
            )

        if not reasoning:
            reasoning = _default_reasoning(aggregated_action, state)

        return AgentDecision(
            action=aggregated_action or None,
            reasoning=reasoning,
            tool_calls=parsed_calls,
            tool_results=results,
            noop=not bool(aggregated_action),
        )


def _round(value: Any, ndigits: int = 2) -> Any:
    if value is None:
        return None
    try:
        return round(float(value), ndigits)
    except (TypeError, ValueError):
        return value


def _default_reasoning(action: dict[str, Any], state: dict[str, Any]) -> str:
    zt = state.get("zone_temp_c")
    bits = [f"zone={_round(zt)}°C"]
    if "cooling" in action:
        bits.append(f"cooling→{action['cooling']}")
    if "heating" in action:
        bits.append(f"heating→{action['heating']}")
    return "Adjusted setpoints for comfort/energy: " + ", ".join(bits)
