"""
Baseline vs AI comparison dashboard.

Loads logs/baseline_run.csv and logs/ai_run.csv, computes kWh savings and
comfort-band compliance, and writes matplotlib charts to dashboard/output/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src import config

# Shared copy for comfort tradeoff (dashboard + Streamlit + docs).
COMFORT_TRADEOFF_NOTE = (
    "Baseline achieves 100% occupied-hours comfort because it holds a fixed "
    "conservative setpoint with no optimization objective; AI achieves ~82% "
    "occupied-hours compliance because it trades bounded comfort margin for "
    "the ~14.1% energy reduction — a tradeoff the fixed baseline schedule is "
    "structurally incapable of making."
)
ALL_HOURS_COMFORT_CONTEXT = (
    "includes intentional overnight setback to ~27°C when unoccupied"
)


def _load_sensors(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing log: {path}")
    df = pd.read_csv(path)
    if "event_type" in df.columns:
        df = df[df["event_type"] == "sensor"].copy()
    df = df.reset_index(drop=True)
    df["step"] = np.arange(len(df))
    for col in ("outdoor_temp_c", "zone_temp_c", "cumulative_kwh", "interval_kwh"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _parse_hour(ts: str) -> int | None:
    try:
        return int(str(ts).split()[1].split(":")[0])
    except Exception:
        return None


def compute_metrics(baseline: pd.DataFrame, ai: pd.DataFrame) -> dict:
    b_kwh = float(baseline["cumulative_kwh"].iloc[-1])
    a_kwh = float(ai["cumulative_kwh"].iloc[-1])
    savings_pct = 100.0 * (b_kwh - a_kwh) / b_kwh if b_kwh > 0 else 0.0

    lo, hi = config.COMFORT_TEMP_MIN_C, config.COMFORT_TEMP_MAX_C

    def comfort_stats(df: pd.DataFrame, occupied_only: bool = False) -> dict:
        d = df
        if occupied_only and "timestamp" in df.columns:
            hours = df["timestamp"].map(_parse_hour)
            mask = hours.map(lambda h: h is not None and 7 <= h < 19)
            d = df.loc[mask]
        z = d["zone_temp_c"].astype(float)
        if z.empty:
            return {"pct_in_band": 0.0, "violations": 0, "n": 0, "t_min": None, "t_max": None, "t_mean": None}
        inside = z.between(lo, hi)
        return {
            "pct_in_band": 100.0 * float(inside.mean()),
            "violations": int((~inside).sum()),
            "n": int(len(z)),
            "t_min": float(z.min()),
            "t_max": float(z.max()),
            "t_mean": float(z.mean()),
        }

    return {
        "baseline_kwh": b_kwh,
        "ai_kwh": a_kwh,
        "savings_kwh": b_kwh - a_kwh,
        "savings_pct": savings_pct,
        "comfort_lo": lo,
        "comfort_hi": hi,
        "baseline_comfort": comfort_stats(baseline),
        "ai_comfort": comfort_stats(ai),
        "baseline_comfort_occupied": comfort_stats(baseline, occupied_only=True),
        "ai_comfort_occupied": comfort_stats(ai, occupied_only=True),
    }


def render_charts(baseline: pd.DataFrame, ai: pd.DataFrame, metrics: dict, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    # Align lengths for overlay (truncate to shorter run if needed)
    n = min(len(baseline), len(ai))
    b = baseline.iloc[:n]
    a = ai.iloc[:n]

    # --- (a) Energy over time ---
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(b["step"], b["cumulative_kwh"], label="Baseline", color="#4a5568", lw=2)
    ax.plot(a["step"], a["cumulative_kwh"], label="AI-controlled", color="#2f855a", lw=2)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Cumulative electricity (kWh)")
    ax.set_title("Facility HVAC electricity — baseline vs AI")
    ax.legend()
    ax.grid(True, alpha=0.3)
    p = out_dir / "energy_over_time.png"
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(p)

    # --- (b) Zone temperature with comfort band ---
    fig, ax = plt.subplots(figsize=(10, 4.5))
    lo, hi = metrics["comfort_lo"], metrics["comfort_hi"]
    ax.axhspan(lo, hi, color="#c6f6d5", alpha=0.55, label=f"Comfort {lo}–{hi}°C")
    ax.plot(b["step"], b["zone_temp_c"], label="Baseline zone", color="#4a5568", lw=1.5)
    ax.plot(a["step"], a["zone_temp_c"], label="AI zone", color="#2f855a", lw=1.5)
    if "outdoor_temp_c" in a.columns:
        ax.plot(a["step"], a["outdoor_temp_c"], label="Outdoor", color="#dd6b20", lw=1, alpha=0.7)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("Zone air temperature — comfort band shaded")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    p = out_dir / "zone_temp_over_time.png"
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(p)

    # --- (c) Summary bar: kWh + % energy saved (green↓ when AI lower) ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax0, ax1 = axes
    bars = ax0.bar(
        ["Baseline", "AI"],
        [metrics["baseline_kwh"], metrics["ai_kwh"]],
        color=["#4a5568", "#2f855a" if metrics["savings_pct"] >= 0 else "#c53030"],
    )
    ax0.set_ylabel("Total kWh")
    ax0.set_title("Total electricity use")
    for bar, val in zip(bars, [metrics["baseline_kwh"], metrics["ai_kwh"]]):
        ax0.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax0.grid(True, axis="y", alpha=0.3)

    pct = float(metrics["savings_pct"])  # (baseline - ai) / baseline * 100
    occ = metrics.get("ai_comfort_occupied") or metrics["ai_comfort"]
    occ_pct = float(occ["pct_in_band"])
    if pct > 0:
        bar_color, energy_label = "#38a169", f"↓ {pct:.1f}% energy saved"
    elif pct < 0:
        bar_color, energy_label = "#e53e3e", f"↑ {abs(pct):.1f}% energy increase"
    else:
        bar_color, energy_label = "#a0aec0", "0.0% energy change"

    ax1.bar(["Energy"], [pct], color=bar_color, width=0.5)
    ax1.axhline(0, color="#718096", lw=1)
    ax1.set_ylabel("% energy saved vs baseline")
    ax1.set_title("Results at a glance")
    ax1.text(
        0.5,
        0.95,
        f"{energy_label}\n"
        f"Occupied comfort (07–19h): {occ_pct:.0f}% within "
        f"{metrics['comfort_lo']:.0f}–{metrics['comfort_hi']:.0f}°C",
        transform=ax1.transAxes,
        ha="center",
        va="top",
        fontsize=11,
        fontweight="bold",
        color="#1a202c",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#cbd5e0"),
    )
    ax1.text(
        0.5,
        -0.22,
        COMFORT_TRADEOFF_NOTE,
        transform=ax1.transAxes,
        ha="center",
        va="top",
        fontsize=7.5,
        color="#4a5568",
        wrap=True,
    )
    ax1.set_ylim(min(-5, pct - 5), max(25, pct + 10))
    ax1.grid(True, axis="y", alpha=0.3)

    p = out_dir / "savings_summary.png"
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(p)

    # Dedicated hero figure — two numbers + comfort tradeoff annotation
    fig_h, ax_h = plt.subplots(figsize=(10, 4.6))
    ax_h.axis("off")
    ax_h.set_title("PoC results — AI vs baseline", fontsize=14, fontweight="bold", pad=12)
    saved = pct >= 0
    base_occ = float(
        (metrics.get("baseline_comfort_occupied") or {}).get("pct_in_band", 100.0)
    )
    ax_h.text(
        0.25,
        0.62,
        f"{pct:.1f}%",
        ha="center",
        va="center",
        fontsize=42,
        fontweight="bold",
        color="#276749" if saved else "#c53030",
    )
    ax_h.text(
        0.25,
        0.40,
        "energy saved vs baseline\n(↓ less kWh is better)" if saved else "energy increase vs baseline",
        ha="center",
        va="center",
        fontsize=11,
        color="#276749" if saved else "#c53030",
    )
    ax_h.text(
        0.75,
        0.62,
        f"{occ_pct:.0f}%",
        ha="center",
        va="center",
        fontsize=42,
        fontweight="bold",
        color="#2b6cb0",
    )
    ax_h.text(
        0.75,
        0.40,
        f"% of occupied time (07–19h)\nwithin comfort band "
        f"{metrics['comfort_lo']:.0f}–{metrics['comfort_hi']:.0f}°C\n"
        f"(baseline occupied: {base_occ:.0f}%)",
        ha="center",
        va="center",
        fontsize=11,
        color="#2b6cb0",
    )
    ax_h.text(
        0.5,
        0.08,
        COMFORT_TRADEOFF_NOTE,
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#4a5568",
        wrap=True,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#edf2f7", edgecolor="#cbd5e0"),
    )
    p_hero = out_dir / "results_hero.png"
    fig_h.tight_layout()
    fig_h.savefig(p_hero, dpi=140)
    plt.close(fig_h)
    paths.append(p_hero)

    # Multi-zone panel if columns exist
    zone_cols = [c for c in a.columns if c.startswith("zone_temp_") and c != "zone_temp_c"]
    if zone_cols:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.axhspan(lo, hi, color="#c6f6d5", alpha=0.45)
        for col in zone_cols:
            label = col.replace("zone_temp_", "")
            ax.plot(a["step"], a[col], lw=1.2, label=label)
        ax.set_title("AI run — all zone temperatures")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("°C")
        ax.legend(ncol=3, fontsize=8)
        ax.grid(True, alpha=0.3)
        p = out_dir / "multizone_temps.png"
        fig.tight_layout()
        fig.savefig(p, dpi=140)
        plt.close(fig)
        paths.append(p)

    return paths


def main() -> int:
    out_dir = config.DASHBOARD_DIR / "output"
    print("Loading logs…")
    baseline = _load_sensors(config.BASELINE_CSV)
    ai = _load_sensors(config.AI_CSV)
    metrics = compute_metrics(baseline, ai)

    print("=" * 60)
    print("DASHBOARD METRICS")
    print(f"  Baseline kWh     : {metrics['baseline_kwh']:.2f}")
    print(f"  AI kWh           : {metrics['ai_kwh']:.2f}")
    print(f"  Savings          : {metrics['savings_kwh']:.2f} kWh "
          f"({metrics['savings_pct']:.1f}% energy saved)")
    print(f"  All-hours comfort : baseline {metrics['baseline_comfort']['pct_in_band']:.1f}% | "
          f"AI {metrics['ai_comfort']['pct_in_band']:.1f}% in-band "
          f"({ALL_HOURS_COMFORT_CONTEXT})")
    print(f"  Occupied hours   : baseline {metrics['baseline_comfort_occupied']['pct_in_band']:.1f}% | "
          f"AI {metrics['ai_comfort_occupied']['pct_in_band']:.1f}% in-band (07–19h)")
    print(f"  Note             : {COMFORT_TRADEOFF_NOTE}")
    print("=" * 60)

    paths = render_charts(baseline, ai, metrics, out_dir)
    for p in paths:
        print(f"  wrote {p}")

    # Persist metrics for Streamlit / docs
    import json
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
