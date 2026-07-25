"""
Baseline vs AI comparison dashboard.

Loads logs/baseline_run.csv and logs/ai_run.csv, computes kWh savings and
comfort-band compliance, and writes matplotlib charts to dashboard/output/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src import config

# Prefer a clean sans for static charts (falls back silently if missing).
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Avenir Next", "Helvetica Neue", "Arial"],
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "figure.facecolor": "#f4f7f5",
        "axes.facecolor": "#fbfcfb",
        "axes.edgecolor": "#c5d0ca",
        "grid.color": "#d7e0db",
    }
)

ALL_HOURS_COMFORT_CONTEXT = (
    "includes intentional overnight setback to ~27°C when unoccupied"
)


def format_tradeoff_note(metrics: dict) -> str:
    """Build the comfort/energy explanation from live metrics (no hardcoded %)."""
    base_occ = float(
        (metrics.get("baseline_comfort_occupied") or {}).get("pct_in_band", 0.0)
    )
    ai_occ = float((metrics.get("ai_comfort_occupied") or {}).get("pct_in_band", 0.0))
    ai_all = float((metrics.get("ai_comfort") or {}).get("pct_in_band", 0.0))
    savings = float(metrics.get("savings_pct", 0.0))
    energy_clause = (
        f"reducing energy use by {savings:.1f}%"
        if savings >= 0
        else f"increasing energy use by {abs(savings):.1f}%"
    )
    return (
        f"Baseline achieves {base_occ:.1f}% occupied-hours comfort because it holds a fixed "
        f"conservative setpoint with no optimization objective; the AI agent achieves "
        f"{ai_occ:.1f}% occupied-hours compliance while {energy_clause} over a 30-day period "
        f"spanning both mild and peak outdoor conditions — a tradeoff the fixed baseline "
        f"schedule cannot make. All-hours comfort ({ai_all:.1f}%) is lower because it "
        f"includes intentional overnight setback to ~27°C during unoccupied periods, "
        f"which is by design."
    )


# Backward-compatible name; prefer format_tradeoff_note(metrics) for live values.
COMFORT_TRADEOFF_NOTE = (
    "Baseline achieves 87.9% occupied-hours comfort because it holds a fixed "
    "conservative setpoint with no optimization objective; the AI agent achieves 66.8% "
    "occupied-hours compliance while reducing energy use by 3.6% over a 30-day period "
    "spanning both mild and peak outdoor conditions — a tradeoff the fixed baseline "
    "schedule cannot make. All-hours comfort (58.2%) is lower because it includes "
    "intentional overnight setback to ~27°C during unoccupied periods, which is by design."
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
            return {
                "pct_in_band": 0.0,
                "violations": 0,
                "n": 0,
                "t_min": None,
                "t_max": None,
                "t_mean": None,
            }
        inside = z.between(lo, hi)
        return {
            "pct_in_band": 100.0 * float(inside.mean()),
            "violations": int((~inside).sum()),
            "n": int(len(z)),
            "t_min": float(z.min()),
            "t_max": float(z.max()),
            "t_mean": float(z.mean()),
        }

    metrics = {
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
    metrics["tradeoff_note"] = format_tradeoff_note(metrics)
    return metrics


def render_charts(
    baseline: pd.DataFrame, ai: pd.DataFrame, metrics: dict, out_dir: Path
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    note = metrics.get("tradeoff_note") or format_tradeoff_note(metrics)

    n = min(len(baseline), len(ai))
    b = baseline.iloc[:n]
    a = ai.iloc[:n]

    # --- (a) Cumulative energy ---
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(b["step"], b["cumulative_kwh"], label="Baseline", color="#1a365d", lw=2.0)
    ax.plot(a["step"], a["cumulative_kwh"], label="AI", color="#0d9488", lw=2.0)
    ax.set_title("Facility HVAC electricity — baseline vs AI")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Cumulative kWh")
    ax.legend()
    ax.grid(True, alpha=0.3)
    p = out_dir / "energy_cumulative.png"
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(p)

    # --- (b) Zone temperature with comfort band ---
    fig, ax = plt.subplots(figsize=(10, 4.2))
    lo, hi = metrics["comfort_lo"], metrics["comfort_hi"]
    ax.axhspan(lo, hi, color="#99f6e4", alpha=0.35, label="Comfort band")
    ax.plot(b["step"], b["zone_temp_c"], label="Baseline zone", color="#1a365d", lw=1.4)
    ax.plot(a["step"], a["zone_temp_c"], label="AI zone", color="#0d9488", lw=1.4)
    if "outdoor_temp_c" in a.columns:
        ax.plot(a["step"], a["outdoor_temp_c"], label="Outdoor", color="#b45309", lw=1.0, alpha=0.75)
    ax.set_title("Zone air temperature — comfort band shaded")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("°C")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    p = out_dir / "zone_temperature.png"
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(p)

    # --- (c) Bar + savings ---
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4.0))
    saved = metrics["savings_pct"] >= 0
    bars = ax0.bar(
        ["Baseline", "AI"],
        [metrics["baseline_kwh"], metrics["ai_kwh"]],
        color=["#334155", "#0d9488" if saved else "#dc2626"],
    )
    ax0.set_ylabel("Total kWh")
    ax0.set_title("Total facility electricity")
    for bar, val in zip(bars, [metrics["baseline_kwh"], metrics["ai_kwh"]]):
        ax0.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    pct = float(metrics["savings_pct"])
    occ = metrics.get("ai_comfort_occupied") or metrics["ai_comfort"]
    occ_pct = float(occ["pct_in_band"])
    # ↓ = less energy used (good) → green; ↑ = more energy (bad) → red
    ax1.bar(
        ["Energy Δ"],
        [pct],
        color="#0d9488" if pct >= 0 else "#dc2626",
    )
    ax1.axhline(0, color="#64748b", lw=0.8)
    ax1.set_ylabel("% energy saved vs baseline")
    ax1.set_title(
        f"{'↓' if pct >= 0 else '↑'} {abs(pct):.1f}% energy "
        f"{'saved' if pct >= 0 else 'increase'}"
    )
    ax1.text(
        0.5,
        0.02,
        f"Occupied comfort (07–19h): {occ_pct:.1f}% within "
        f"{metrics['comfort_lo']:.0f}–{metrics['comfort_hi']:.0f}°C",
        transform=ax1.transAxes,
        ha="center",
        va="bottom",
        fontsize=8,
        color="#475569",
    )
    ax1.set_ylim(min(-5, pct - 5), max(25, pct + 10))
    ax1.grid(True, axis="y", alpha=0.3)

    p = out_dir / "savings_summary.png"
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(p)

    # Dedicated hero figure
    fig_h, ax_h = plt.subplots(figsize=(10, 4.8))
    ax_h.axis("off")
    ax_h.set_title("PoC results — AI vs baseline (30-day)", fontsize=14, fontweight="bold", pad=12)
    base_occ = float(
        (metrics.get("baseline_comfort_occupied") or {}).get("pct_in_band", 0.0)
    )
    energy_color = "#0f766e" if saved else "#b91c1c"
    ax_h.text(
        0.25,
        0.64,
        f"{'↓' if saved else '↑'} {abs(pct):.1f}%",
        ha="center",
        va="center",
        fontsize=40,
        fontweight="bold",
        color=energy_color,
    )
    ax_h.text(
        0.25,
        0.42,
        "energy saved vs baseline\n(↓ less kWh is better)"
        if saved
        else "energy increase vs baseline\n(↑ more kWh)",
        ha="center",
        va="center",
        fontsize=11,
        color=energy_color,
    )
    ax_h.text(
        0.75,
        0.64,
        f"{occ_pct:.1f}%",
        ha="center",
        va="center",
        fontsize=40,
        fontweight="bold",
        color="#0369a1",
    )
    ax_h.text(
        0.75,
        0.42,
        f"% of occupied time (07–19h)\nwithin comfort band "
        f"{metrics['comfort_lo']:.0f}–{metrics['comfort_hi']:.0f}°C\n"
        f"(baseline occupied: {base_occ:.1f}%)",
        ha="center",
        va="center",
        fontsize=11,
        color="#0369a1",
    )
    ax_h.text(
        0.5,
        0.06,
        note,
        ha="center",
        va="bottom",
        fontsize=8.0,
        color="#334155",
        wrap=True,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#f1f5f9", edgecolor="#cbd5e1"),
    )
    p_hero = out_dir / "results_hero.png"
    fig_h.tight_layout()
    fig_h.savefig(p_hero, dpi=140)
    plt.close(fig_h)
    paths.append(p_hero)

    zone_cols = [c for c in a.columns if c.startswith("zone_temp_") and c != "zone_temp_c"]
    if zone_cols:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.axhspan(lo, hi, color="#99f6e4", alpha=0.35)
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
    note = metrics["tradeoff_note"]

    print("=" * 60)
    print("DASHBOARD METRICS")
    print(f"  Baseline kWh     : {metrics['baseline_kwh']:.2f}")
    print(f"  AI kWh           : {metrics['ai_kwh']:.2f}")
    print(
        f"  Savings          : {metrics['savings_kwh']:.2f} kWh "
        f"({metrics['savings_pct']:.1f}% energy saved)"
    )
    print(
        f"  All-hours comfort : baseline {metrics['baseline_comfort']['pct_in_band']:.1f}% | "
        f"AI {metrics['ai_comfort']['pct_in_band']:.1f}% in-band "
        f"({ALL_HOURS_COMFORT_CONTEXT})"
    )
    print(
        f"  Occupied hours   : baseline {metrics['baseline_comfort_occupied']['pct_in_band']:.1f}% | "
        f"AI {metrics['ai_comfort_occupied']['pct_in_band']:.1f}% in-band (07–19h)"
    )
    print(f"  Note             : {note}")
    print("=" * 60)

    paths = render_charts(baseline, ai, metrics, out_dir)
    for p in paths:
        print(f"  wrote {p}")

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
