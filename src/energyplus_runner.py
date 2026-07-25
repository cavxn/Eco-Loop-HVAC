"""
EnergyPlusRunner — live pyenergyplus wrapper with mid-simulation read/write.

Registers two callbacks on a single run_energyplus() invocation:
  - begin_system_timestep_before_predictor — re-apply cached setpoints
  - end_zone_timestep_after_zone_reporting — read sensors, log, invoke on_timestep
    (where the LLM may call apply_action → set_actuator_value before E+ advances)
"""

from __future__ import annotations

import sys
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src import config
from src.data_logger import EventLogger


def _ensure_energyplus_on_path() -> Path:
    """Make the EnergyPlus install's pyenergyplus package importable."""
    root = config.ENERGYPLUS_ROOT
    if not root.is_dir():
        raise FileNotFoundError(
            f"EnergyPlus root not found at {root}. "
            "Set ENERGYPLUS_ROOT to your install directory."
        )
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


_ensure_energyplus_on_path()

from pyenergyplus.api import EnergyPlusAPI  # noqa: E402  (path must be set first)


# Zone Temperature Control actuators — standard EMS handles for thermostat override.
ACTUATOR_COMPONENT = "Zone Temperature Control"
ACTUATOR_COOLING = "Cooling Setpoint"
ACTUATOR_HEATING = "Heating Setpoint"

# Fallback: override the shared compact schedules used by all DualSP thermostats.
SCHEDULE_ACTUATOR_COMPONENT = "Schedule:Compact"
SCHEDULE_ACTUATOR_CONTROL = "Schedule Value"
COOLING_SCHEDULE_NAME = "CLGSETP_SCH"
HEATING_SCHEDULE_NAME = "HTGSETP_SCH"


@dataclass
class SensorHandles:
    outdoor_temp: int = -1
    electricity_demand_w: int = -1
    electricity_meter: int = -1
    cooling_meter: int = -1
    zone_temp: dict[str, int] = field(default_factory=dict)
    cooling_setpoint: dict[str, int] = field(default_factory=dict)
    heating_setpoint: dict[str, int] = field(default_factory=dict)
    # Shared schedule actuators (apply same setpoint to all zones)
    cooling_schedule: int = -1
    heating_schedule: int = -1
    ready: bool = False
    use_zone_actuators: bool = False
    use_schedule_actuators: bool = False


TimestepCallback = Callable[["EnergyPlusRunner"], None]


class EnergyPlusRunner:
    """
    Thin wrapper around EnergyPlusAPI for closed-loop HVAC setpoint control.

    Typical usage:
        runner = EnergyPlusRunner(on_timestep=my_callback)
        exit_code = runner.run(idf_path, epw_path)
    """

    def __init__(
        self,
        zones: list[str] | None = None,
        on_timestep: TimestepCallback | None = None,
        output_dir: str | Path | None = None,
        quiet: bool = True,
        logger: EventLogger | None = None,
    ) -> None:
        self.zones = list(zones or config.ZONES)
        self.on_timestep = on_timestep
        self.output_dir = Path(output_dir) if output_dir else config.OUTPUTS_DIR / "eplus"
        self.quiet = quiet
        self.logger = logger

        self.api = EnergyPlusAPI()
        self.exchange = self.api.exchange
        self.runtime = self.api.runtime

        self._ep_state: Any = None
        self._handles = SensorHandles()
        self._callback_count = 0
        self._got_handles = False
        self._pending_actions: dict[str, Any] = {}
        self._last_applied: dict[str, Any] = {}
        self._current_state: dict[str, Any] = {}
        self._last_meter_j = 0.0
        self._last_cooling_j = 0.0
        self._cumulative_kwh = 0.0
        self._cumulative_cooling_kwh = 0.0
        self._variables_requested = False
        self._last_logged_time_key: tuple[int, int, int, int, int] | None = None
        # Rolling window of logged sensor snapshots for predictive features.
        self._sensor_history: deque[dict[str, Any]] = deque(maxlen=3)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_state(self) -> dict[str, Any]:
        """Return the latest sensor snapshot (empty until first successful callback)."""
        return dict(self._current_state)

    def get_recent_readings(self, n: int = 3) -> list[dict[str, Any]]:
        """Return up to the last N logged sensor snapshots (oldest → newest)."""
        items = list(self._sensor_history)
        return items[-n:] if n < len(items) else items

    def compute_trend_features(self, n: int = 3) -> dict[str, Any]:
        """
        Lightweight predictive features from the last N logged readings:
          outdoor_temp_delta_c, zone_temp_delta_c, margin_to_nearer_boundary_c.
        """
        readings = self.get_recent_readings(n)
        lo, hi = config.COMFORT_TEMP_MIN_C, config.COMFORT_TEMP_MAX_C
        if not readings:
            return {
                "n_readings": 0,
                "outdoor_temp_delta_c": None,
                "zone_temp_delta_c": None,
                "margin_to_nearer_boundary_c": None,
                "outdoor_trend": "insufficient_history",
                "favorable_for_edge": False,
            }

        def _f(v: Any) -> float | None:
            return float(v) if isinstance(v, (int, float)) else None

        outdoors = [_f(r.get("outdoor_temp_c")) for r in readings]
        zones = [_f(r.get("zone_temp_c")) for r in readings]
        outdoors_ok = [t for t in outdoors if t is not None]
        zones_ok = [t for t in zones if t is not None]

        outdoor_delta = (
            outdoors_ok[-1] - outdoors_ok[0] if len(outdoors_ok) >= 2 else None
        )
        zone_delta = zones_ok[-1] - zones_ok[0] if len(zones_ok) >= 2 else None

        zt = zones_ok[-1] if zones_ok else None
        margin = min(zt - lo, hi - zt) if zt is not None else None

        # Classify outdoor trend for band-edge gating (cooling season).
        if outdoor_delta is None:
            outdoor_trend = "insufficient_history"
            favorable = False
        elif abs(outdoor_delta) < 0.5:
            outdoor_trend = "stable"
            favorable = True
        elif outdoor_delta > 0:
            outdoor_trend = "rising_sharp" if outdoor_delta >= 2.0 else "rising"
            favorable = False  # approaching a peak — hold mid-band
        else:
            sharp = outdoor_delta <= -2.0
            outdoor_trend = "falling_sharp" if sharp else "falling"
            favorable = not sharp

        return {
            "n_readings": len(readings),
            "outdoor_temp_delta_c": round(outdoor_delta, 3) if outdoor_delta is not None else None,
            "zone_temp_delta_c": round(zone_delta, 3) if zone_delta is not None else None,
            "margin_to_nearer_boundary_c": round(margin, 3) if margin is not None else None,
            "outdoor_trend": outdoor_trend,
            "favorable_for_edge": favorable,
            "comfort_band_c": [lo, hi],
        }

    def apply_action(self, setpoints: dict[str, Any]) -> dict[str, Any]:
        """
        Apply cooling/heating setpoints to the live simulation.

        Accepted shapes (any combination):
            {"cooling": {"Core_ZN": 24.5}, "heating": {"Core_ZN": 21.0}}
            {"cooling_setpoint": 24.5, "heating_setpoint": 21.0}   # all zones / schedules
            {"Core_ZN": {"cooling": 24.5, "heating": 21.0}}

        Returns the normalized action that was (or will be) written.
        """
        normalized = self._normalize_setpoints(setpoints)
        if not normalized:
            return {}

        # If we are inside a live callback with ready handles, write immediately.
        if self._ep_state is not None and self._handles.ready:
            self._write_actuators(normalized)
            self._last_applied = self._merge_actions(self._last_applied, normalized)
        else:
            # Queue for the next callback (e.g. agent called slightly early).
            self._pending_actions = self._merge_actions(self._pending_actions, normalized)
        return normalized

    def log_decision(
        self,
        action: dict[str, Any] | None,
        reasoning: str | None = None,
    ) -> dict[str, Any] | None:
        """Log an agent decision against the latest sensor snapshot (if a logger is attached)."""
        if self.logger is None:
            return None
        return self.logger.log_decision(self.get_state(), action, reasoning)

    def run(
        self,
        idf_path: str | Path,
        epw_path: str | Path,
        output_dir: str | Path | None = None,
    ) -> int:
        """Register the timestep callback and run EnergyPlus. Returns the exit code."""
        idf_path = Path(idf_path)
        epw_path = Path(epw_path)
        if output_dir is not None:
            self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if not idf_path.is_file():
            raise FileNotFoundError(f"IDF not found: {idf_path}")
        if not epw_path.is_file():
            raise FileNotFoundError(f"EPW not found: {epw_path}")

        self._reset_run_state()
        self._ep_state = self.api.state_manager.new_state()

        # Request variables before the run so handles resolve later.
        self._request_variables(self._ep_state)

        # Actuate before the predictor; sense/log/decide after HVAC so energy is real.
        self.runtime.callback_begin_system_timestep_before_predictor(
            self._ep_state, self._on_begin_system_timestep
        )
        self.runtime.callback_end_zone_timestep_after_zone_reporting(
            self._ep_state, self._on_end_system_timestep
        )

        args = [
            "-w",
            str(epw_path),
            "-d",
            str(self.output_dir),
            str(idf_path),
        ]
        if self.quiet:
            args.insert(0, "-x")  # expand objects without extra prompts if needed

        # Prefer progress silence via console flag when available.
        try:
            self.runtime.set_console_output_status(self._ep_state, not self.quiet)
        except Exception:
            pass

        try:
            exit_code = self.runtime.run_energyplus(self._ep_state, args)
        finally:
            if self.logger is not None:
                self.logger.close()
        return int(exit_code)

    @property
    def callback_count(self) -> int:
        return self._callback_count

    # ------------------------------------------------------------------
    # Callback + handles
    # ------------------------------------------------------------------

    def _reset_run_state(self) -> None:
        self._handles = SensorHandles()
        self._callback_count = 0
        self._got_handles = False
        self._pending_actions = {}
        self._last_applied = {}
        self._current_state = {}
        self._last_meter_j = 0.0
        self._last_cooling_j = 0.0
        self._cumulative_kwh = 0.0
        self._cumulative_cooling_kwh = 0.0
        self._variables_requested = False
        self._last_logged_time_key = None
        self._sensor_history = deque(maxlen=3)

    def _request_variables(self, state: Any) -> None:
        """Ask EnergyPlus to keep these variables available to the API."""
        self.exchange.request_variable(state, "Site Outdoor Air Drybulb Temperature", "Environment")
        self.exchange.request_variable(
            state, "Facility Total Building Electricity Demand Rate", "Whole Building"
        )
        self.exchange.request_variable(
            state, "Facility Total HVAC Electricity Demand Rate", "Whole Building"
        )
        for zone in self.zones:
            self.exchange.request_variable(state, "Zone Mean Air Temperature", zone)
        self._variables_requested = True

    def _on_begin_system_timestep(self, state: Any) -> None:
        """Apply (and re-apply) setpoints before the predictor each timestep."""
        self._ep_state = state

        if self.exchange.warmup_flag(state):
            return
        if not self.exchange.api_data_fully_ready(state):
            return

        if not self._got_handles:
            ok = self._cache_handles(state)
            if not ok:
                return
            self._got_handles = True

        # New writes from the agent take priority, then persist last action.
        if self._pending_actions:
            self._last_applied = self._merge_actions(self._last_applied, self._pending_actions)
            self._pending_actions = {}

        if self._last_applied:
            self._write_actuators(self._last_applied)

    def _on_end_system_timestep(self, state: Any) -> None:
        """Read sensors after HVAC reporting — demand/meters reflect this timestep."""
        self._ep_state = state

        if self.exchange.warmup_flag(state):
            return
        if not self.exchange.api_data_fully_ready(state):
            return

        if not self._got_handles:
            ok = self._cache_handles(state)
            if not ok:
                return
            self._got_handles = True

        # Debounce to one sample per 10-minute system timestep bucket.
        # end_system_timestep_after_hvac_reporting can fire multiple times per step.
        minute_bucket = int(self.exchange.minutes(state)) // 10
        time_key = (
            int(self.exchange.year(state)),
            int(self.exchange.month(state)),
            int(self.exchange.day_of_month(state)),
            int(self.exchange.hour(state)),
            minute_bucket,
        )
        if time_key == self._last_logged_time_key:
            return
        self._last_logged_time_key = time_key

        self._current_state = self._read_sensors(state)
        self._callback_count += 1
        self._sensor_history.append(
            {
                "timestamp": self._current_state.get("timestamp"),
                "outdoor_temp_c": self._current_state.get("outdoor_temp_c"),
                "zone_temp_c": self._current_state.get("zone_temp_c"),
                "callback_index": self._callback_count,
            }
        )

        # Always log raw sensors before any AI / external control decision.
        if self.logger is not None:
            try:
                self.logger.log_sensor(self._current_state)
            except Exception as exc:  # noqa: BLE001 — logging must not kill the sim
                print(f"[EnergyPlusRunner] logger error (ignored): {exc}", file=sys.stderr)

        if self.on_timestep is not None:
            try:
                self.on_timestep(self)
            except Exception as exc:  # noqa: BLE001 — never crash the sim from a callback
                print(f"[EnergyPlusRunner] on_timestep error (ignored): {exc}", file=sys.stderr)

    @staticmethod
    def _merge_actions(base: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {
            "cooling": dict(base.get("cooling") or {}),
            "heating": dict(base.get("heating") or {}),
        }
        if "cooling" in new:
            out["cooling"].update(new["cooling"])
        if "heating" in new:
            out["heating"].update(new["heating"])
        if not out["cooling"]:
            out.pop("cooling")
        if not out["heating"]:
            out.pop("heating")
        return out

    def _cache_handles(self, state: Any) -> bool:
        """Resolve and cache sensor/actuator handles once. Returns False if critical sensors missing."""
        ex = self.exchange
        h = SensorHandles()

        h.outdoor_temp = ex.get_variable_handle(
            state, "Site Outdoor Air Drybulb Temperature", "Environment"
        )
        # Prefer HVAC-specific demand; fall back to whole-building.
        hvac_demand = ex.get_variable_handle(
            state, "Facility Total HVAC Electricity Demand Rate", "Whole Building"
        )
        building_demand = ex.get_variable_handle(
            state, "Facility Total Building Electricity Demand Rate", "Whole Building"
        )
        h.electricity_demand_w = hvac_demand if hvac_demand != -1 else building_demand
        # Cumulative meters (J) — primary source of truth for energy totals.
        # Some builds expose meters without the ":Facility" suffix via API; try both.
        h.electricity_meter = ex.get_meter_handle(state, "Electricity:Facility")
        if h.electricity_meter == -1:
            h.electricity_meter = ex.get_meter_handle(state, "ElectricityNet:Facility")
        h.cooling_meter = ex.get_meter_handle(state, "Cooling:Electricity")
        # Also try HVAC electricity demand as a better instantaneous signal.
        if hvac_demand == -1:
            # Some IDFs only expose building-level demand.
            pass
        print(
            f"[EnergyPlusRunner] Energy handles: demand={h.electricity_demand_w} "
            f"facility_meter={h.electricity_meter} cooling_meter={h.cooling_meter}"
        )

        for zone in self.zones:
            h.zone_temp[zone] = ex.get_variable_handle(state, "Zone Mean Air Temperature", zone)

        # Prefer per-zone thermostat actuators.
        zone_ok = True
        for zone in self.zones:
            cool = ex.get_actuator_handle(state, ACTUATOR_COMPONENT, ACTUATOR_COOLING, zone)
            heat = ex.get_actuator_handle(state, ACTUATOR_COMPONENT, ACTUATOR_HEATING, zone)
            h.cooling_setpoint[zone] = cool
            h.heating_setpoint[zone] = heat
            if cool == -1 or heat == -1:
                zone_ok = False

        h.use_zone_actuators = zone_ok and all(
            h.cooling_setpoint[z] != -1 and h.heating_setpoint[z] != -1 for z in self.zones
        )

        # Fallback: actuate the shared HTG/CLG schedules.
        if not h.use_zone_actuators:
            h.cooling_schedule = ex.get_actuator_handle(
                state, SCHEDULE_ACTUATOR_COMPONENT, SCHEDULE_ACTUATOR_CONTROL, COOLING_SCHEDULE_NAME
            )
            h.heating_schedule = ex.get_actuator_handle(
                state, SCHEDULE_ACTUATOR_COMPONENT, SCHEDULE_ACTUATOR_CONTROL, HEATING_SCHEDULE_NAME
            )
            h.use_schedule_actuators = h.cooling_schedule != -1 and h.heating_schedule != -1

        critical_ok = h.outdoor_temp != -1 and all(h.zone_temp[z] != -1 for z in self.zones)
        if not critical_ok:
            print(
                "[EnergyPlusRunner] Failed to resolve critical sensor handles; "
                f"outdoor={h.outdoor_temp}, zones={h.zone_temp}",
                file=sys.stderr,
            )
            return False

        if not h.use_zone_actuators and not h.use_schedule_actuators:
            print(
                "[EnergyPlusRunner] WARNING: no setpoint actuators resolved — "
                "apply_action will be a no-op until handles are available.",
                file=sys.stderr,
            )

        h.ready = True
        self._handles = h
        mode = "zone" if h.use_zone_actuators else ("schedule" if h.use_schedule_actuators else "none")
        print(f"[EnergyPlusRunner] Handles cached (actuator mode={mode}).")
        return True

    def _safe_variable(self, state: Any, handle: int) -> float | None:
        if handle is None or handle < 0:
            return None
        try:
            return float(self.exchange.get_variable_value(state, handle))
        except Exception:
            return None

    def _safe_meter(self, state: Any, handle: int) -> float | None:
        if handle is None or handle < 0:
            return None
        try:
            return float(self.exchange.get_meter_value(state, handle))
        except Exception:
            return None

    def _read_sensors(self, state: Any) -> dict[str, Any]:
        h = self._handles
        ex = self.exchange

        outdoor = self._safe_variable(state, h.outdoor_temp)
        demand_w = self._safe_variable(state, h.electricity_demand_w)
        meter_j = self._safe_meter(state, h.electricity_meter)
        cooling_j = self._safe_meter(state, h.cooling_meter)

        # Instantaneous demand (W) integrated over the 10-min timestep is the most
        # reliable mid-run energy signal. Output:Meter values are often period-
        # based (hourly) and reset, so we only use them as a cross-check.
        timestep_hours = 1.0 / 6.0
        interval_kwh = 0.0
        if demand_w is not None:
            interval_kwh = max(0.0, (demand_w / 1000.0) * timestep_hours)
            self._cumulative_kwh += interval_kwh
        elif meter_j is not None:
            if meter_j < self._last_meter_j:
                delta_j = meter_j  # reporting-period reset
            else:
                delta_j = meter_j - self._last_meter_j
            self._last_meter_j = meter_j
            interval_kwh = max(0.0, delta_j) / 3.6e6
            self._cumulative_kwh += interval_kwh

        interval_cooling_kwh = 0.0
        if cooling_j is not None:
            if cooling_j < self._last_cooling_j:
                delta_c = cooling_j
            else:
                delta_c = cooling_j - self._last_cooling_j
            self._last_cooling_j = cooling_j
            interval_cooling_kwh = max(0.0, delta_c) / 3.6e6
            self._cumulative_cooling_kwh += interval_cooling_kwh

        zone_temps: dict[str, float | None] = {}
        for zone, handle in h.zone_temp.items():
            zone_temps[zone] = self._safe_variable(state, handle)

        primary = config.PRIMARY_ZONE if config.PRIMARY_ZONE in zone_temps else self.zones[0]
        sim_time = {
            "year": ex.year(state),
            "month": ex.month(state),
            "day": ex.day_of_month(state),
            "hour": ex.hour(state),
            "minute": ex.minutes(state),
        }

        return {
            "sim_time": sim_time,
            "timestamp": (
                f"{sim_time['year']:04d}-{sim_time['month']:02d}-{sim_time['day']:02d}"
                f" {sim_time['hour']:02d}:{sim_time['minute']:02d}"
            ),
            "outdoor_temp_c": outdoor,
            "zone_temps_c": zone_temps,
            "zone_temp_c": zone_temps.get(primary),
            "primary_zone": primary,
            "electricity_demand_w": demand_w,
            "interval_kwh": interval_kwh,
            "cumulative_kwh": self._cumulative_kwh,
            "interval_cooling_kwh": interval_cooling_kwh,
            "cumulative_cooling_kwh": self._cumulative_cooling_kwh,
            "callback_index": self._callback_count,
            "active_setpoints": dict(self._last_applied) if self._last_applied else None,
        }

    # ------------------------------------------------------------------
    # Actuators
    # ------------------------------------------------------------------

    def _normalize_setpoints(self, setpoints: dict[str, Any]) -> dict[str, Any]:
        """Normalize heterogeneous action dicts into {cooling: {zone: v}, heating: {zone: v}}."""
        cooling: dict[str, float] = {}
        heating: dict[str, float] = {}

        if not setpoints:
            return {}

        # Flat all-zone form
        if "cooling_setpoint" in setpoints or "cooling" in setpoints and isinstance(
            setpoints.get("cooling"), (int, float)
        ):
            val = setpoints.get("cooling_setpoint", setpoints.get("cooling"))
            for z in self.zones:
                cooling[z] = float(val)
        if "heating_setpoint" in setpoints or "heating" in setpoints and isinstance(
            setpoints.get("heating"), (int, float)
        ):
            val = setpoints.get("heating_setpoint", setpoints.get("heating"))
            for z in self.zones:
                heating[z] = float(val)

        # Nested by kind: {"cooling": {"Core_ZN": 24.5}, ...}
        cool_map = setpoints.get("cooling")
        if isinstance(cool_map, dict):
            for z, v in cool_map.items():
                cooling[z] = float(v)
        heat_map = setpoints.get("heating")
        if isinstance(heat_map, dict):
            for z, v in heat_map.items():
                heating[z] = float(v)

        # Nested by zone: {"Core_ZN": {"cooling": 24.5, "heating": 21.0}}
        for key, val in setpoints.items():
            if key in self.zones and isinstance(val, dict):
                if "cooling" in val:
                    cooling[key] = float(val["cooling"])
                if "heating" in val:
                    heating[key] = float(val["heating"])
                if "cooling_setpoint" in val:
                    cooling[key] = float(val["cooling_setpoint"])
                if "heating_setpoint" in val:
                    heating[key] = float(val["heating_setpoint"])

        # Clamp to safe bounds
        for z, v in list(cooling.items()):
            cooling[z] = min(
                config.COOLING_SETPOINT_MAX_C, max(config.COOLING_SETPOINT_MIN_C, v)
            )
        for z, v in list(heating.items()):
            heating[z] = min(
                config.HEATING_SETPOINT_MAX_C, max(config.HEATING_SETPOINT_MIN_C, v)
            )

        out: dict[str, Any] = {}
        if cooling:
            out["cooling"] = cooling
        if heating:
            out["heating"] = heating
        return out

    def _write_actuators(self, normalized: dict[str, Any]) -> None:
        state = self._ep_state
        h = self._handles
        if state is None or not h.ready:
            self._pending_actions.update(normalized)
            return

        cooling = normalized.get("cooling", {})
        heating = normalized.get("heating", {})

        if h.use_zone_actuators:
            for zone, value in cooling.items():
                handle = h.cooling_setpoint.get(zone, -1)
                if handle != -1:
                    self.exchange.set_actuator_value(state, handle, float(value))
            for zone, value in heating.items():
                handle = h.heating_setpoint.get(zone, -1)
                if handle != -1:
                    self.exchange.set_actuator_value(state, handle, float(value))
            return

        if h.use_schedule_actuators:
            # Shared schedules — use primary zone's value, or any provided value.
            if cooling:
                primary = cooling.get(config.PRIMARY_ZONE) or next(iter(cooling.values()))
                self.exchange.set_actuator_value(state, h.cooling_schedule, float(primary))
            if heating:
                primary = heating.get(config.PRIMARY_ZONE) or next(iter(heating.values()))
                self.exchange.set_actuator_value(state, h.heating_schedule, float(primary))


if __name__ == "__main__":
    # Smoke test: run with EventLogger and print progress.
    samples: list[dict[str, Any]] = []

    def _capture(runner: EnergyPlusRunner) -> None:
        state = runner.get_state()
        if runner.callback_count <= 5 or runner.callback_count % 50 == 0:
            print(
                f"[{runner.callback_count}] {state.get('timestamp')} "
                f"Tout={state.get('outdoor_temp_c')} "
                f"Tzone={state.get('zone_temp_c')} "
                f"kWh={state.get('cumulative_kwh'):.3f}"
            )
        samples.append(state)
        if runner.callback_count == 10:
            applied = runner.apply_action({"cooling_setpoint": 25.0, "heating_setpoint": 20.0})
            runner.log_decision(applied, reasoning="smoke-test setpoint bump")
            print(f"Applied action: {applied}")

    logger = EventLogger(
        csv_path=config.LOGS_DIR / "smoke_run.csv",
        jsonl_path=config.LOGS_DIR / "smoke_run.jsonl",
        run_id="smoke",
        mirror_bus_path=config.EVENT_BUS_JSONL,
        overwrite=True,
    )
    r = EnergyPlusRunner(on_timestep=_capture, quiet=True, logger=logger)
    code = r.run(config.IDF_PATH, config.EPW_PATH, output_dir=config.OUTPUTS_DIR / "smoke")
    print(
        f"Exit code={code}, callbacks={r.callback_count}, "
        f"samples={len(samples)}, log_records={logger.record_count}"
    )
    print(f"CSV -> {logger.csv_path}")
    if samples:
        print("Last state:", samples[-1])
