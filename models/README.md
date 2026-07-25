# Building models

## `baseline.idf`

Stock **DOE Reference Building — Small Office New 2004 (Chicago)**
(`RefBldgSmallOfficeNew2004_Chicago.idf` from EnergyPlus ExampleFiles),
with these **light PoC edits only**:

1. `SimulationControl` — run weather-file periods `YES`, sizing-period sims `NO`
2. `RunPeriod` — shortened to **15–17 July** (three summer days) instead of annual
3. Extra / timestep-oriented `Output:Variable` lines for live API sensing

Geometry, HVAC, schedules, and thermostat objects are otherwise the reference file.

## `weather/chicago.epw`

`USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw` from the EnergyPlus WeatherData
folder. The EPW is **not embedded in the IDF**; runners pass it on the EnergyPlus
CLI (`-w models/weather/chicago.epw`).

## Runtime-modified IDFs?

**None.** This PoC does **not** rewrite or save alternate `.idf` files during the
AI run. Heating/cooling setpoints are injected into the **live** simulation via
EnergyPlus’s Python **Actuator API** (`Zone Temperature Control` /
`set_actuator_value`) from `EnergyPlusRunner.apply_action()`. Reviewers looking
for “modified IDF artifacts” should inspect `logs/ai_run.csv` (decision rows) and
the actuator path in `src/energyplus_runner.py` instead.
