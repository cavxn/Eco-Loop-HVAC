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

1. Keep zone air temperature inside the comfort band
   {config.COMFORT_TEMP_MIN_C}–{config.COMFORT_TEMP_MAX_C} °C.
2. Minimize cooling electricity.

Critical energy-saving rule for COOLING season:
- Raising the cooling setpoint SAVES energy. Lowering it WASTES energy.
- Default daytime cooling setpoint: {config.COMFORT_TEMP_MAX_C}°C (top of comfort).
- Night / early morning (hour < 7 or hour >= 19): raise cooling to 26–27°C to
  avoid unnecessary overnight cooling while allowing mild drift.
- Only drop cooling toward 23–24°C if zone temp is clearly ABOVE
  {config.COMFORT_TEMP_MAX_C}°C.
- Do NOT cool to 22–23°C — that wastes energy versus the baseline.
- Heating setpoint: keep near {config.COMFORT_TEMP_MIN_C}°C in summer (deadband).
- Always issue a tool call. Prefer set_cooling_setpoint on {config.PRIMARY_ZONE}.
- Multi-zone: after setting Core_ZN, also set Perimeter_ZN_1..4 to the SAME
  cooling setpoint unless a perimeter zone is >0.8°C hotter than Core (then
  give that zone 0.5°C lower cooling SP).
- When carbon_level is "high", bias toward the highest acceptable cooling SP.
- One short sentence of reasoning max.

Bounds:
- Cooling setpoints: [{config.COOLING_SETPOINT_MIN_C}, {config.COOLING_SETPOINT_MAX_C}] °C
- Heating setpoints: [{config.HEATING_SETPOINT_MIN_C}, {config.HEATING_SETPOINT_MAX_C}] °C
"""


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
        One agent turn: prompt → tool call(s) → execute → return action + reasoning.

        Retries once on malformed / empty tool responses, then falls back to a
        deterministic energy-saving heuristic (never blocks the sim for minutes).
        """
        state = state or self.runner.get_state()
        self._history.append(self._compact_state(state))

        last_error: str | None = None
        attempts = 1 + max(0, config.LLM_MAX_RETRIES)
        rate_limited = False

        for attempt in range(attempts):
            try:
                decision = self._call_llm_and_execute(state, attempt=attempt)
                if decision.tool_calls or decision.action:
                    return self._enforce_energy_policy(state, decision)
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
        return fallback

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
        """
        hour = self._sim_hour(state)
        action = dict(decision.action or {})
        cooling = dict(action.get("cooling") or {})
        heating = dict(action.get("heating") or {})

        # Night setback — critical for beating baseline energy.
        if hour < 7 or hour >= 19:
            target = 27.0
            changed = False
            for z in config.ZONES:
                cur = cooling.get(z)
                if cur is None or float(cur) < target:
                    cooling[z] = target
                    changed = True
            if changed:
                for z, val in cooling.items():
                    execute_tool(
                        self.runner,
                        "set_cooling_setpoint",
                        {"zone_name": z, "value_celsius": val},
                    )
                decision.reasoning = (
                    (decision.reasoning or "") + f" | night setback→{target}°C"
                ).strip(" |")
        else:
            # Daytime: if only primary zone set, mirror to all zones.
            if cooling and config.PRIMARY_ZONE in cooling and len(cooling) == 1:
                primary_val = float(cooling[config.PRIMARY_ZONE])
                for z in config.ZONES:
                    cooling[z] = primary_val
                    execute_tool(
                        self.runner,
                        "set_cooling_setpoint",
                        {"zone_name": z, "value_celsius": primary_val},
                    )
            if heating and config.PRIMARY_ZONE in heating and len(heating) == 1:
                primary_val = float(heating[config.PRIMARY_ZONE])
                for z in config.ZONES:
                    heating[z] = primary_val
                    execute_tool(
                        self.runner,
                        "set_heating_setpoint",
                        {"zone_name": z, "value_celsius": primary_val},
                    )

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
        hour = (state.get("sim_time") or {}).get("hour")
        if hour is None and state.get("timestamp"):
            try:
                hour = int(str(state["timestamp"]).split()[1].split(":")[0])
            except Exception:
                hour = 12

        carbon = get_carbon_signal(state)
        if isinstance(hour, (int, float)) and (hour < 7 or hour >= 19):
            cool_sp = min(config.COOLING_SETPOINT_MAX_C, 27.0)
            reason = f"Night setback cooling={cool_sp}°C"
        elif isinstance(zt, (int, float)) and zt > config.COMFORT_TEMP_MAX_C + 0.3:
            cool_sp = max(config.COOLING_SETPOINT_MIN_C, config.COMFORT_TEMP_MAX_C - 0.5)
            reason = f"Zone warm ({zt:.1f}°C) — cooling={cool_sp}°C"
        else:
            cool_sp = config.COMFORT_TEMP_MAX_C
            if carbon.level == "high":
                cool_sp = min(config.COOLING_SETPOINT_MAX_C, cool_sp + 0.5)
            reason = f"Hold cooling={cool_sp}°C (carbon={carbon.level})"

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
                    {"zone_name": zone, "value_celsius": heat_sp},
                )
            )
            tool_calls.append(
                {"name": "set_cooling_setpoint", "arguments": {"zone_name": zone, "value_celsius": z_cool}}
            )
            tool_calls.append(
                {"name": "set_heating_setpoint", "arguments": {"zone_name": zone, "value_celsius": heat_sp}}
            )
            cooling_map[zone] = z_cool
            heating_map[zone] = heat_sp

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
        # Exclude the just-appended current reading from "recent" to avoid dup.
        recent = history[:-1] if len(history) > 1 else []

        comfort_status = "unknown"
        zt = current.get("zone_temp_c")
        if isinstance(zt, (int, float)):
            if zt < config.COMFORT_TEMP_MIN_C:
                comfort_status = "BELOW comfort band — raise heating / lower cooling carefully"
            elif zt > config.COMFORT_TEMP_MAX_C:
                comfort_status = "ABOVE comfort band — lower cooling setpoint or increase cooling"
            else:
                comfort_status = "INSIDE comfort band — prioritize energy savings"

        carbon = get_carbon_signal(state)

        payload = {
            "objective": "minimize energy (and carbon) while respecting comfort band",
            "comfort_band_c": [config.COMFORT_TEMP_MIN_C, config.COMFORT_TEMP_MAX_C],
            "comfort_status": comfort_status,
            "primary_zone": config.PRIMARY_ZONE,
            "zones": list(config.ZONES),
            "carbon": carbon.as_prompt_block(),
            "current": current,
            "recent_readings": recent,
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
            noop=not aggregated_action,
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
