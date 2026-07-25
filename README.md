# Eco-Loop Building Agents

Live closed-loop Physical AI PoC: an open-source LLM (via Groq tool-calling)
controls HVAC setpoints inside a running **EnergyPlus** simulation and proves
electricity savings vs baseline without breaking thermal comfort.

## Quick start

```bash
# 1. EnergyPlus 25.x installed; set path in .env (see .env.example)
cp .env.example .env   # add GROQ / LLM_API_KEY

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Baseline (no AI)
PYTHONPATH=. python -m src.run_baseline

# 3. AI closed loop
PYTHONPATH=. python -m src.run_ai_loop

# 4. Static charts
PYTHONPATH=. python dashboard/dashboard.py
# → dashboard/output/*.png

# 5. Live Streamlit dashboard (optional)
streamlit run dashboard/streamlit_app.py

# 6. MCP server demo (optional stretch)
pip install mcp
PYTHONPATH=. python -m src.mcp_server
```

## Layout

```
src/energyplus_runner.py   # pyenergyplus live read/write
src/building_agent.py      # LLM tool-calling brain
src/tools.py               # OpenAI-compatible tool schemas + exec
src/mcp_server.py          # Minimal MCP wrapper (stretch)
src/carbon.py              # Grid carbon signal (stretch)
src/data_logger.py         # CSV/JSONL event bus
src/run_baseline.py / run_ai_loop.py
dashboard/dashboard.py     # matplotlib comparison charts
dashboard/streamlit_app.py # live UI (stretch)
docs/ARCHITECTURE.md
models/baseline.idf        # RefBldg Small Office, Jul 15–17
```

## Scoring focus

| Rubric | How we hit it |
|--------|----------------|
| System integration (30%) | Callbacks never crash; LLM retry + heuristic fallback |
| Energy efficiency (25%) | Raise cooling SP / night setback vs baseline 24°C |
| Thermal comfort (20%) | Band enforced in prompt + actuator clamps |
| Agentic tool-calling (15%) | 3 tools, MCP-compatible schemas + optional MCP server |
| Docs (10%) | `docs/ARCHITECTURE.md` |

## Config

See `.env.example` for `ENERGYPLUS_ROOT`, `LLM_*`, `AGENT_CALLBACK_INTERVAL`,
comfort bounds, optional `CARBON_API_URL`.
