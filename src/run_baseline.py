"""
Run the uncontrolled baseline simulation and log every timestep to CSV/JSONL.

No LLM / setpoint overrides — the IDF's native thermostat schedules drive HVAC.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `python src/run_baseline.py` as well as `python -m src.run_baseline`.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src import config
from src.data_logger import EventLogger
from src.energyplus_runner import EnergyPlusRunner


def main() -> int:
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    output_dir = config.OUTPUTS_DIR / "baseline"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("BASELINE RUN (no AI control)")
    print(f"  IDF : {config.IDF_PATH}")
    print(f"  EPW : {config.EPW_PATH}")
    print(f"  CSV : {config.BASELINE_CSV}")
    print("=" * 60)

    logger = EventLogger(
        csv_path=config.BASELINE_CSV,
        jsonl_path=config.BASELINE_JSONL,
        run_id="baseline",
        mirror_bus_path=config.EVENT_BUS_JSONL,
        overwrite=True,
    )

    def on_timestep(runner: EnergyPlusRunner) -> None:
        state = runner.get_state()
        n = runner.callback_count
        if n <= 3 or n % 72 == 0:  # ~every 12 simulated hours
            print(
                f"  [{n:4d}] {state.get('timestamp')}  "
                f"Tout={state.get('outdoor_temp_c'):.1f}°C  "
                f"Tzone={state.get('zone_temp_c'):.2f}°C  "
                f"kWh={state.get('cumulative_kwh'):.2f}"
            )

    runner = EnergyPlusRunner(on_timestep=on_timestep, quiet=True, logger=logger)
    exit_code = runner.run(
        config.IDF_PATH,
        config.EPW_PATH,
        output_dir=output_dir,
    )

    ok = _validate_baseline_csv(config.BASELINE_CSV, expected_callbacks=runner.callback_count)
    print("-" * 60)
    print(f"EnergyPlus exit code : {exit_code}")
    print(f"Callbacks            : {runner.callback_count}")
    print(f"Log records          : {logger.record_count}")
    print(f"CSV written          : {config.BASELINE_CSV} ({'OK' if ok else 'FAILED'})")
    print("-" * 60)
    return 0 if exit_code == 0 and ok else 1


def _validate_baseline_csv(csv_path: Path, expected_callbacks: int) -> bool:
    """Confirm the CSV exists and contains sensible temperature / energy values."""
    import pandas as pd

    if not csv_path.is_file() or csv_path.stat().st_size == 0:
        print(f"ERROR: CSV missing or empty: {csv_path}")
        return False

    df = pd.read_csv(csv_path)
    sensors = df[df["event_type"] == "sensor"] if "event_type" in df.columns else df

    if sensors.empty:
        print("ERROR: no sensor rows in CSV")
        return False

    if expected_callbacks and len(sensors) < max(1, expected_callbacks // 2):
        print(
            f"ERROR: too few sensor rows ({len(sensors)}) vs callbacks ({expected_callbacks})"
        )
        return False

    required = ["timestamp", "outdoor_temp_c", "zone_temp_c", "cumulative_kwh"]
    for col in required:
        if col not in sensors.columns:
            print(f"ERROR: missing column {col}")
            return False

    outdoor = sensors["outdoor_temp_c"].astype(float)
    zone = sensors["zone_temp_c"].astype(float)
    kwh = sensors["cumulative_kwh"].astype(float)

    checks = [
        (outdoor.notna().all(), "outdoor_temp has NaNs"),
        (zone.notna().all(), "zone_temp has NaNs"),
        # July Chicago: outdoor roughly 10–40°C
        (outdoor.between(-10, 45).all(), f"outdoor_temp out of range [{outdoor.min()}, {outdoor.max()}]"),
        # Conditioned office: roughly 18–30°C under default schedules
        (zone.between(15, 35).all(), f"zone_temp out of range [{zone.min()}, {zone.max()}]"),
        (float(kwh.iloc[-1]) > 0, "cumulative_kwh never increased"),
        (
            (kwh.diff().fillna(0) >= -1e-6).all(),
            "cumulative_kwh decreased",
        ),
    ]

    ok = True
    for passed, msg in checks:
        if not passed:
            print(f"ERROR: {msg}")
            ok = False

    if ok:
        print("Validation summary:")
        print(f"  rows              : {len(sensors)}")
        print(f"  outdoor °C        : {outdoor.min():.1f} .. {outdoor.max():.1f}")
        print(f"  zone °C           : {zone.min():.1f} .. {zone.max():.1f}")
        print(f"  total kWh         : {kwh.iloc[-1]:.2f}")
        print(f"  first timestamp   : {sensors['timestamp'].iloc[0]}")
        print(f"  last timestamp    : {sensors['timestamp'].iloc[-1]}")

    return ok


if __name__ == "__main__":
    raise SystemExit(main())
