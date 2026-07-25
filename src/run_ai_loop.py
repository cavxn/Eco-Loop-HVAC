"""
Run the closed-loop AI-controlled EnergyPlus simulation.

Same IDF/EPW as the baseline. Every N timesteps, BuildingAgent reasons over
sensor state, issues tool calls, and setpoints are written back into the live sim.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src import config
from src.building_agent import BuildingAgent
from src.data_logger import EventLogger
from src.energyplus_runner import EnergyPlusRunner


def main() -> int:
    if not config.LLM_API_KEY:
        print(
            "ERROR: No LLM API key configured.\n"
            "Set LLM_API_KEY or GROQ_API_KEY in .env (see .env.example)."
        )
        return 2

    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    output_dir = config.OUTPUTS_DIR / "ai"
    output_dir.mkdir(parents=True, exist_ok=True)

    interval = max(1, config.AGENT_CALLBACK_INTERVAL)

    print("=" * 60)
    print("AI CLOSED-LOOP RUN")
    print(f"  IDF      : {config.IDF_PATH}")
    print(f"  EPW      : {config.EPW_PATH}")
    print(f"  CSV      : {config.AI_CSV}")
    print(f"  Model    : {config.LLM_MODEL}")
    print(f"  Base URL : {config.LLM_BASE_URL}")
    print(f"  Interval : every {interval} callback(s)")
    print("=" * 60)

    logger = EventLogger(
        csv_path=config.AI_CSV,
        jsonl_path=config.AI_JSONL,
        run_id="ai",
        mirror_bus_path=config.EVENT_BUS_JSONL,
        overwrite=True,
    )

    # Agent is created lazily on first control step so handle-cache / first
    # sensor read is already available.
    agent_holder: dict[str, BuildingAgent | None] = {"agent": None}
    stats = {
        "decisions": 0,
        "noops": 0,
        "errors": 0,
        "llm_seconds": 0.0,
    }

    def on_timestep(runner: EnergyPlusRunner) -> None:
        n = runner.callback_count
        state = runner.get_state()

        if n <= 3 or n % 72 == 0:
            print(
                f"  [{n:4d}] {state.get('timestamp')}  "
                f"Tout={_fmt(state.get('outdoor_temp_c'))}°C  "
                f"Tzone={_fmt(state.get('zone_temp_c'))}°C  "
                f"kWh={_fmt(state.get('cumulative_kwh'), 2)}"
            )

        # Only invoke the LLM every N callbacks.
        if n % interval != 0:
            return

        if agent_holder["agent"] is None:
            agent_holder["agent"] = BuildingAgent(runner)

        agent = agent_holder["agent"]
        t0 = time.perf_counter()
        decision = agent.decide(state)
        dt = time.perf_counter() - t0
        stats["llm_seconds"] += dt
        stats["decisions"] += 1
        if decision.noop:
            stats["noops"] += 1
        if decision.error:
            stats["errors"] += 1

        # apply_action is already invoked inside tool executors; log the decision.
        runner.log_decision(decision.as_log_action(), decision.reasoning)

        cool = (decision.action or {}).get("cooling", {})
        heat = (decision.action or {}).get("heating", {})
        flag = "NOOP" if decision.noop else "ACT "
        print(
            f"  → LLM {flag} #{stats['decisions']} ({dt:.2f}s)  "
            f"cool={cool or '-'} heat={heat or '-'}  "
            f"| {(decision.reasoning or '')[:90]}"
        )

    runner = EnergyPlusRunner(on_timestep=on_timestep, quiet=True, logger=logger)
    exit_code = runner.run(
        config.IDF_PATH,
        config.EPW_PATH,
        output_dir=output_dir,
    )

    ok = _validate_ai_csv(config.AI_CSV, runner.callback_count, stats["decisions"])

    print("-" * 60)
    print(f"EnergyPlus exit code : {exit_code}")
    print(f"Callbacks            : {runner.callback_count}")
    print(f"Log records          : {logger.record_count}")
    print(f"LLM decisions        : {stats['decisions']} "
          f"(noops={stats['noops']}, errors={stats['errors']})")
    print(f"LLM time total       : {stats['llm_seconds']:.1f}s "
          f"(avg {stats['llm_seconds'] / max(1, stats['decisions']):.2f}s)")
    print(f"CSV written          : {config.AI_CSV} ({'OK' if ok else 'FAILED'})")
    print("-" * 60)
    return 0 if exit_code == 0 and ok else 1


def _fmt(value, ndigits: int = 1) -> str:
    try:
        return f"{float(value):.{ndigits}f}"
    except (TypeError, ValueError):
        return "?"


def _validate_ai_csv(csv_path: Path, callbacks: int, decisions: int) -> bool:
    import pandas as pd

    if not csv_path.is_file() or csv_path.stat().st_size == 0:
        print(f"ERROR: CSV missing or empty: {csv_path}")
        return False

    df = pd.read_csv(csv_path)
    sensors = df[df["event_type"] == "sensor"]
    acts = df[df["event_type"] == "decision"]

    if sensors.empty:
        print("ERROR: no sensor rows")
        return False

    zone = sensors["zone_temp_c"].astype(float)
    kwh = sensors["cumulative_kwh"].astype(float)
    outdoor = sensors["outdoor_temp_c"].astype(float)

    ok = True
    checks = [
        (len(sensors) >= max(1, callbacks // 2), f"few sensor rows ({len(sensors)})"),
        (outdoor.between(-10, 45).all(), f"outdoor out of range [{outdoor.min()}, {outdoor.max()}]"),
        (zone.between(15, 35).all(), f"zone out of range [{zone.min()}, {zone.max()}]"),
        (float(kwh.iloc[-1]) > 0, "cumulative kWh is 0"),
        (decisions > 0, "no LLM decisions were made"),
        (len(acts) > 0, "no decision rows logged"),
    ]
    for passed, msg in checks:
        if not passed:
            print(f"ERROR: {msg}")
            ok = False

    if ok:
        print("Validation summary:")
        print(f"  sensor rows       : {len(sensors)}")
        print(f"  decision rows     : {len(acts)}")
        print(f"  outdoor °C        : {outdoor.min():.1f} .. {outdoor.max():.1f}")
        print(f"  zone °C           : {zone.min():.1f} .. {zone.max():.1f}")
        print(f"  total kWh         : {kwh.iloc[-1]:.2f}")
        if config.BASELINE_CSV.is_file():
            b = pd.read_csv(config.BASELINE_CSV)
            b = b[b["event_type"] == "sensor"]
            if not b.empty:
                base_kwh = float(b["cumulative_kwh"].iloc[-1])
                ai_kwh = float(kwh.iloc[-1])
                if base_kwh > 0:
                    savings = 100.0 * (base_kwh - ai_kwh) / base_kwh
                    print(f"  vs baseline kWh   : {base_kwh:.2f} → {ai_kwh:.2f} "
                          f"({savings:+.1f}%)")

    return ok


if __name__ == "__main__":
    raise SystemExit(main())
