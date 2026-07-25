"""
Grid carbon-intensity signal for carbon-aware HVAC prompting.

Uses a deterministic diurnal mock for Chicago (kgCO2/kWh) so the PoC never
depends on an external API mid-simulation. Optionally overlays a live fetch
when CARBON_API_URL is set.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class CarbonSignal:
    intensity_kg_per_kwh: float
    level: str  # low | medium | high
    source: str
    hour: int | None = None

    def as_prompt_block(self) -> dict[str, Any]:
        return {
            "grid_carbon_intensity_kg_per_kwh": round(self.intensity_kg_per_kwh, 3),
            "carbon_level": self.level,
            "carbon_source": self.source,
            "guidance": _guidance(self.level),
        }


def _guidance(level: str) -> str:
    if level == "high":
        return (
            "Grid is carbon-intensive NOW — prioritize deeper energy cuts "
            "(raise cooling setpoint toward top of comfort band / night setback)."
        )
    if level == "low":
        return (
            "Grid is relatively clean — still save energy, but comfort can take "
            "slight priority within the band."
        )
    return "Grid carbon is moderate — balance comfort and energy as usual."


def _classify(intensity: float) -> str:
    if intensity >= 0.45:
        return "high"
    if intensity <= 0.30:
        return "low"
    return "medium"


def mock_chicago_carbon(hour: int | None = None, month: int | None = None) -> CarbonSignal:
    """
    Synthetic PJM/Chicago-like intensity curve (kgCO2/kWh).

    Higher mid-afternoon (gas peakers), lower overnight / high-renewable shoulder.
    """
    if hour is None:
        hour = datetime.now().hour
    if month is None:
        month = datetime.now().month

    # Base summer intensity ~0.38, winter a bit higher.
    seasonal = 0.40 if month in (12, 1, 2) else 0.36
    # Diurnal swing ±0.12 peaking near hour 16.
    diurnal = 0.12 * math.sin((hour - 10) / 24.0 * 2 * math.pi)
    intensity = max(0.18, min(0.65, seasonal + diurnal))
    return CarbonSignal(
        intensity_kg_per_kwh=intensity,
        level=_classify(intensity),
        source="mock_chicago_diurnal",
        hour=hour,
    )


def fetch_live_carbon() -> CarbonSignal | None:
    """Optional live fetch — disabled unless CARBON_API_URL is configured."""
    url = os.getenv("CARBON_API_URL", "").strip()
    if not url:
        return None
    try:
        import json
        import urllib.request

        with urllib.request.urlopen(url, timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        # Expect {"carbon_intensity": <float kg/kWh>} or Electricity Maps-like
        intensity = float(
            data.get("carbon_intensity")
            or data.get("carbonIntensity")
            or data.get("data", {}).get("carbonIntensity", 0)
        )
        if intensity > 5:  # likely gCO2/kWh
            intensity = intensity / 1000.0
        return CarbonSignal(
            intensity_kg_per_kwh=intensity,
            level=_classify(intensity),
            source="live_api",
        )
    except Exception:
        return None


def get_carbon_signal(state: dict[str, Any] | None = None) -> CarbonSignal:
    """Prefer live API when configured; otherwise mock from sim clock."""
    live = fetch_live_carbon()
    if live is not None:
        return live

    hour = None
    month = None
    if state:
        sim = state.get("sim_time") or {}
        hour = sim.get("hour")
        month = sim.get("month")
        if hour is None and state.get("timestamp"):
            try:
                hour = int(str(state["timestamp"]).split()[1].split(":")[0])
            except Exception:
                hour = None
    return mock_chicago_carbon(hour=hour, month=month)
