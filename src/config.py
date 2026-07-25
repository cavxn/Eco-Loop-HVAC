"""Configuration for the eco-loop EnergyPlus + LLM HVAC PoC."""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# Project roots
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"

# EnergyPlus install (override with ENERGYPLUS_ROOT)
DEFAULT_EPLUS_ROOT = Path.home() / "EnergyPlus" / "EnergyPlus-25.2.0-cf7368216c-Darwin-macOS13-arm64"
ENERGYPLUS_ROOT = Path(os.getenv("ENERGYPLUS_ROOT", str(DEFAULT_EPLUS_ROOT)))

# Building model + weather
IDF_PATH = Path(os.getenv("IDF_PATH", str(MODELS_DIR / "baseline.idf")))
EPW_PATH = Path(os.getenv("EPW_PATH", str(MODELS_DIR / "weather" / "chicago.epw")))

# Conditioned zones in RefBldgSmallOfficeNew2004_Chicago
ZONES = [
    "Core_ZN",
    "Perimeter_ZN_1",
    "Perimeter_ZN_2",
    "Perimeter_ZN_3",
    "Perimeter_ZN_4",
]
PRIMARY_ZONE = "Core_ZN"

# Comfort band (°C) — ASHRAE-ish office occupied range for the PoC
COMFORT_TEMP_MIN_C = float(os.getenv("COMFORT_TEMP_MIN_C", "21.0"))
COMFORT_TEMP_MAX_C = float(os.getenv("COMFORT_TEMP_MAX_C", "25.0"))

# Allowed actuator setpoint bounds (°C)
COOLING_SETPOINT_MIN_C = 22.0
COOLING_SETPOINT_MAX_C = 28.0
HEATING_SETPOINT_MIN_C = 18.0
HEATING_SETPOINT_MAX_C = 22.0

# Call the LLM every N system-timestep callbacks (Timestep=6 → 10 min steps;
# N=24 ≈ every 4 simulated hours — denser control on multi-week runs)
AGENT_CALLBACK_INTERVAL = int(os.getenv("AGENT_CALLBACK_INTERVAL", "24"))

# LLM (OpenAI-compatible: Groq / Together / Fireworks)
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("GROQ_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
LLM_TIMEOUT_S = float(os.getenv("LLM_TIMEOUT_S", "30"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))

# Logging
BASELINE_CSV = LOGS_DIR / "baseline_run.csv"
BASELINE_JSONL = LOGS_DIR / "baseline_run.jsonl"
AI_CSV = LOGS_DIR / "ai_run.csv"
AI_JSONL = LOGS_DIR / "ai_run.jsonl"
EVENT_BUS_JSONL = LOGS_DIR / "event_bus.jsonl"
