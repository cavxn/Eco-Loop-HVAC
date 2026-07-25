"""
EventLogger — thread-safe CSV + JSONL event log (the PoC "communication bus").

Every record captures simulation timestamp, sensor state, optional control action,
and optional LLM reasoning text. Full history lives here so prompts can stay compact.
"""

from __future__ import annotations

import csv
import json
import threading
from pathlib import Path
from typing import Any


# Stable CSV columns for dashboard joins (baseline vs AI).
CSV_FIELDS = [
    "event_type",
    "timestamp",
    "callback_index",
    "outdoor_temp_c",
    "zone_temp_c",
    "primary_zone",
    "electricity_demand_w",
    "interval_kwh",
    "cumulative_kwh",
    "action_json",
    "reasoning",
    # Per-zone temperatures (RefBldg small office)
    "zone_temp_Core_ZN",
    "zone_temp_Perimeter_ZN_1",
    "zone_temp_Perimeter_ZN_2",
    "zone_temp_Perimeter_ZN_3",
    "zone_temp_Perimeter_ZN_4",
]


def _json_safe(value: Any) -> Any:
    """Coerce values into JSON-serializable form."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


class EventLogger:
    """
    Append-only dual-format logger.

    - CSV: flat columns for pandas / dashboard
    - JSONL: full nested records (sensor_state, action, reasoning) for the event bus
    """

    def __init__(
        self,
        csv_path: str | Path,
        jsonl_path: str | Path,
        *,
        run_id: str = "run",
        mirror_bus_path: str | Path | None = None,
        overwrite: bool = True,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.jsonl_path = Path(jsonl_path)
        self.run_id = run_id
        self.mirror_bus_path = Path(mirror_bus_path) if mirror_bus_path else None
        self._lock = threading.Lock()
        self._closed = False
        self._count = 0

        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        if self.mirror_bus_path is not None:
            self.mirror_bus_path.parent.mkdir(parents=True, exist_ok=True)

        if overwrite:
            self.csv_path.write_text("")
            self.jsonl_path.write_text("")
            # Bus is append-only across runs unless overwrite requested for a fresh session.
            if self.mirror_bus_path is not None and not self.mirror_bus_path.exists():
                self.mirror_bus_path.write_text("")

        # Open CSV with header if empty / new.
        new_csv = (not self.csv_path.exists()) or self.csv_path.stat().st_size == 0
        self._csv_file = self.csv_path.open("a", newline="", encoding="utf-8")
        self._jsonl_file = self.jsonl_path.open("a", encoding="utf-8")
        self._bus_file = (
            self.mirror_bus_path.open("a", encoding="utf-8")
            if self.mirror_bus_path is not None
            else None
        )
        self._csv_writer = csv.DictWriter(
            self._csv_file, fieldnames=CSV_FIELDS, extrasaction="ignore"
        )
        if new_csv:
            self._csv_writer.writeheader()
            self._csv_file.flush()

    @property
    def record_count(self) -> int:
        return self._count

    def log(
        self,
        *,
        timestamp: str | None = None,
        sensor_state: dict[str, Any] | None = None,
        action: dict[str, Any] | None = None,
        reasoning: str | None = None,
        event_type: str = "sensor",
        **extra: Any,
    ) -> dict[str, Any]:
        """
        Append one event to CSV + JSONL (and optional event-bus mirror).

        Returns the JSONL record that was written.
        """
        if self._closed:
            raise RuntimeError("EventLogger is closed")

        sensor_state = dict(sensor_state or {})
        ts = timestamp or sensor_state.get("timestamp") or ""
        zone_temps = sensor_state.get("zone_temps_c") or {}

        record: dict[str, Any] = {
            "run_id": self.run_id,
            "event_type": event_type,
            "timestamp": ts,
            "callback_index": sensor_state.get("callback_index"),
            "sensor_state": _json_safe(sensor_state),
            "action": _json_safe(action) if action else None,
            "reasoning": reasoning,
            **{k: _json_safe(v) for k, v in extra.items()},
        }

        csv_row = {
            "event_type": event_type,
            "timestamp": ts,
            "callback_index": sensor_state.get("callback_index", ""),
            "outdoor_temp_c": sensor_state.get("outdoor_temp_c", ""),
            "zone_temp_c": sensor_state.get("zone_temp_c", ""),
            "primary_zone": sensor_state.get("primary_zone", ""),
            "electricity_demand_w": sensor_state.get("electricity_demand_w", ""),
            "interval_kwh": sensor_state.get("interval_kwh", ""),
            "cumulative_kwh": sensor_state.get("cumulative_kwh", ""),
            "action_json": json.dumps(_json_safe(action)) if action else "",
            "reasoning": (reasoning or "").replace("\n", " ").strip(),
            "zone_temp_Core_ZN": zone_temps.get("Core_ZN", ""),
            "zone_temp_Perimeter_ZN_1": zone_temps.get("Perimeter_ZN_1", ""),
            "zone_temp_Perimeter_ZN_2": zone_temps.get("Perimeter_ZN_2", ""),
            "zone_temp_Perimeter_ZN_3": zone_temps.get("Perimeter_ZN_3", ""),
            "zone_temp_Perimeter_ZN_4": zone_temps.get("Perimeter_ZN_4", ""),
        }

        line = json.dumps(record, ensure_ascii=False)

        with self._lock:
            self._csv_writer.writerow(csv_row)
            self._csv_file.flush()
            self._jsonl_file.write(line + "\n")
            self._jsonl_file.flush()
            if self._bus_file is not None:
                self._bus_file.write(line + "\n")
                self._bus_file.flush()
            self._count += 1

        return record

    def log_sensor(self, sensor_state: dict[str, Any]) -> dict[str, Any]:
        """Convenience: log a raw sensor snapshot before any AI decision."""
        return self.log(
            timestamp=sensor_state.get("timestamp"),
            sensor_state=sensor_state,
            action=None,
            reasoning=None,
            event_type="sensor",
        )

    def log_decision(
        self,
        sensor_state: dict[str, Any],
        action: dict[str, Any] | None,
        reasoning: str | None = None,
    ) -> dict[str, Any]:
        """Convenience: log an agent decision tied to the current sensor state."""
        return self.log(
            timestamp=sensor_state.get("timestamp"),
            sensor_state=sensor_state,
            action=action,
            reasoning=reasoning,
            event_type="decision",
        )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._csv_file.close()
            self._jsonl_file.close()
            if self._bus_file is not None:
                self._bus_file.close()
            self._closed = True

    def __enter__(self) -> EventLogger:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
