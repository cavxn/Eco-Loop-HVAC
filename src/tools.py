"""
OpenAI-compatible tool schemas + executors for live EnergyPlus setpoint control.

Tools mirror a lightweight MCP-style surface:
  - get_zone_state
  - set_cooling_setpoint
  - set_heating_setpoint
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Callable

from src import config

if TYPE_CHECKING:
    from src.energyplus_runner import EnergyPlusRunner


# ---------------------------------------------------------------------------
# OpenAI / Groq / Together function-calling schemas
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_zone_state",
            "description": (
                "Read the latest sensor snapshot for one zone (air temperature, "
                "outdoor temperature, electricity demand, cumulative kWh). "
                "Use when you need to inspect before changing setpoints."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "zone_name": {
                        "type": "string",
                        "description": (
                            "Zone to inspect. One of: "
                            + ", ".join(config.ZONES)
                        ),
                        "enum": list(config.ZONES),
                    },
                },
                "required": ["zone_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_cooling_setpoint",
            "description": (
                "Set the zone cooling setpoint in °C. Higher values save cooling "
                f"energy. Allowed range: {config.COOLING_SETPOINT_MIN_C}–"
                f"{config.COOLING_SETPOINT_MAX_C}°C. Keep zone air temp within "
                f"the comfort band {config.COMFORT_TEMP_MIN_C}–"
                f"{config.COMFORT_TEMP_MAX_C}°C."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "zone_name": {
                        "type": "string",
                        "description": "Target zone name.",
                        "enum": list(config.ZONES),
                    },
                    "value_celsius": {
                        "type": "number",
                        "description": "Cooling setpoint in degrees Celsius.",
                    },
                },
                "required": ["zone_name", "value_celsius"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_heating_setpoint",
            "description": (
                "Set the zone heating setpoint in °C. Lower values save heating "
                f"energy. Allowed range: {config.HEATING_SETPOINT_MIN_C}–"
                f"{config.HEATING_SETPOINT_MAX_C}°C. Keep zone air temp within "
                f"the comfort band {config.COMFORT_TEMP_MIN_C}–"
                f"{config.COMFORT_TEMP_MAX_C}°C."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "zone_name": {
                        "type": "string",
                        "description": "Target zone name.",
                        "enum": list(config.ZONES),
                    },
                    "value_celsius": {
                        "type": "number",
                        "description": "Heating setpoint in degrees Celsius.",
                    },
                },
                "required": ["zone_name", "value_celsius"],
            },
        },
    },
]


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def _validate_zone(zone_name: str) -> str:
    if zone_name not in config.ZONES:
        raise ValueError(
            f"Unknown zone '{zone_name}'. Expected one of {config.ZONES}."
        )
    return zone_name


# ---------------------------------------------------------------------------
# Executors (bound to a live EnergyPlusRunner)
# ---------------------------------------------------------------------------

def get_zone_state(runner: EnergyPlusRunner, zone_name: str) -> dict[str, Any]:
    """Return a compact state dict for the requested zone."""
    zone = _validate_zone(zone_name)
    state = runner.get_state()
    zone_temps = state.get("zone_temps_c") or {}
    return {
        "zone_name": zone,
        "zone_temp_c": zone_temps.get(zone, state.get("zone_temp_c")),
        "outdoor_temp_c": state.get("outdoor_temp_c"),
        "electricity_demand_w": state.get("electricity_demand_w"),
        "cumulative_kwh": state.get("cumulative_kwh"),
        "timestamp": state.get("timestamp"),
        "comfort_band_c": [config.COMFORT_TEMP_MIN_C, config.COMFORT_TEMP_MAX_C],
    }


def set_cooling_setpoint(
    runner: EnergyPlusRunner, zone_name: str, value_celsius: float
) -> dict[str, Any]:
    """Write a cooling setpoint into the live simulation via apply_action."""
    zone = _validate_zone(zone_name)
    value = _clamp(
        value_celsius, config.COOLING_SETPOINT_MIN_C, config.COOLING_SETPOINT_MAX_C
    )
    applied = runner.apply_action({"cooling": {zone: value}})
    return {
        "ok": True,
        "tool": "set_cooling_setpoint",
        "zone_name": zone,
        "value_celsius": value,
        "applied": applied,
    }


def set_heating_setpoint(
    runner: EnergyPlusRunner, zone_name: str, value_celsius: float
) -> dict[str, Any]:
    """Write a heating setpoint into the live simulation via apply_action."""
    zone = _validate_zone(zone_name)
    value = _clamp(
        value_celsius, config.HEATING_SETPOINT_MIN_C, config.HEATING_SETPOINT_MAX_C
    )
    applied = runner.apply_action({"heating": {zone: value}})
    return {
        "ok": True,
        "tool": "set_heating_setpoint",
        "zone_name": zone,
        "value_celsius": value,
        "applied": applied,
    }


TOOL_EXECUTORS: dict[str, Callable[..., dict[str, Any]]] = {
    "get_zone_state": get_zone_state,
    "set_cooling_setpoint": set_cooling_setpoint,
    "set_heating_setpoint": set_heating_setpoint,
}


def execute_tool(
    runner: EnergyPlusRunner,
    name: str,
    arguments: dict[str, Any] | str | None,
) -> dict[str, Any]:
    """
    Dispatch a tool call by name.

    `arguments` may be a dict or a JSON string (as returned by some providers).
    """
    if name not in TOOL_EXECUTORS:
        return {"ok": False, "error": f"Unknown tool: {name}"}

    if arguments is None:
        args: dict[str, Any] = {}
    elif isinstance(arguments, str):
        try:
            args = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError as exc:
            return {"ok": False, "error": f"Invalid tool arguments JSON: {exc}"}
    elif isinstance(arguments, dict):
        args = arguments
    else:
        return {"ok": False, "error": f"Unsupported arguments type: {type(arguments)}"}

    try:
        result = TOOL_EXECUTORS[name](runner, **args)
        return result
    except TypeError as exc:
        return {"ok": False, "error": f"Bad arguments for {name}: {exc}", "arguments": args}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc), "tool": name, "arguments": args}
