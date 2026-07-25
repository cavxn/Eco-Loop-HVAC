"""
Minimal MCP server exposing eco-loop HVAC tools.

This wraps the same tool surface as src/tools.py so judges can see a real
MCP-style server, while the in-process BuildingAgent loop remains the
primary closed-loop path (faster, no IPC during EnergyPlus callbacks).

Run (stdio transport):
  PYTHONPATH=. python -m src.mcp_server

Or with the MCP CLI / Cursor MCP config pointing at this module.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src import config
from src.tools import TOOL_SCHEMAS, execute_tool

# Soft dependency — keep import optional so core PoC works without mcp installed.
try:
    from mcp.server.fastmcp import FastMCP
except ImportError:  # pragma: no cover
    FastMCP = None  # type: ignore


class _RunnerProxy:
    """
    Lightweight stand-in used when the MCP server runs outside a live simulation.

    In the closed loop, BuildingAgent calls tools.py directly against the real
    EnergyPlusRunner. This MCP server is for tooling demos / Cursor integration;
    it records requested setpoints to the event bus JSONL.
    """

    def __init__(self) -> None:
        self._state: dict[str, Any] = {
            "timestamp": None,
            "outdoor_temp_c": None,
            "zone_temp_c": None,
            "zone_temps_c": {z: None for z in config.ZONES},
            "electricity_demand_w": None,
            "cumulative_kwh": None,
        }
        self.actions: list[dict[str, Any]] = []
        self._hydrate_from_latest_log()

    def _hydrate_from_latest_log(self) -> None:
        for path in (config.AI_CSV, config.BASELINE_CSV):
            if not path.is_file():
                continue
            try:
                import pandas as pd

                df = pd.read_csv(path)
                if "event_type" in df.columns:
                    df = df[df["event_type"] == "sensor"]
                if df.empty:
                    continue
                row = df.iloc[-1]
                zone_temps = {
                    z: float(row[f"zone_temp_{z}"]) if f"zone_temp_{z}" in row and pd.notna(row[f"zone_temp_{z}"]) else None
                    for z in config.ZONES
                }
                self._state = {
                    "timestamp": row.get("timestamp"),
                    "outdoor_temp_c": float(row["outdoor_temp_c"]) if pd.notna(row.get("outdoor_temp_c")) else None,
                    "zone_temp_c": float(row["zone_temp_c"]) if pd.notna(row.get("zone_temp_c")) else None,
                    "zone_temps_c": zone_temps,
                    "electricity_demand_w": float(row["electricity_demand_w"]) if pd.notna(row.get("electricity_demand_w")) else None,
                    "cumulative_kwh": float(row["cumulative_kwh"]) if pd.notna(row.get("cumulative_kwh")) else None,
                }
                return
            except Exception:
                continue

    def get_state(self) -> dict[str, Any]:
        return dict(self._state)

    def apply_action(self, setpoints: dict[str, Any]) -> dict[str, Any]:
        self.actions.append(setpoints)
        # Append to event bus so the "comms bus" artifact shows MCP traffic.
        try:
            config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
            record = {
                "run_id": "mcp",
                "event_type": "mcp_action",
                "action": setpoints,
            }
            with config.EVENT_BUS_JSONL.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
        except Exception:
            pass
        return setpoints


def build_mcp_server():
    if FastMCP is None:
        raise ImportError(
            "mcp package not installed. Run: pip install mcp"
        )

    mcp = FastMCP("eco-loop-hvac")
    runner = _RunnerProxy()

    @mcp.tool()
    def get_zone_state(zone_name: str) -> dict[str, Any]:
        """Read latest zone temperature / energy snapshot."""
        return execute_tool(runner, "get_zone_state", {"zone_name": zone_name})

    @mcp.tool()
    def set_cooling_setpoint(zone_name: str, value_celsius: float) -> dict[str, Any]:
        """Set cooling setpoint (°C) for a zone."""
        return execute_tool(
            runner,
            "set_cooling_setpoint",
            {"zone_name": zone_name, "value_celsius": value_celsius},
        )

    @mcp.tool()
    def set_heating_setpoint(zone_name: str, value_celsius: float) -> dict[str, Any]:
        """Set heating setpoint (°C) for a zone."""
        return execute_tool(
            runner,
            "set_heating_setpoint",
            {"zone_name": zone_name, "value_celsius": value_celsius},
        )

    @mcp.resource("schema://tools")
    def tool_schema() -> str:
        """OpenAI-compatible tool schemas mirrored by this MCP server."""
        return json.dumps(TOOL_SCHEMAS, indent=2)

    return mcp


def main() -> None:
    server = build_mcp_server()
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
