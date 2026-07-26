# 🏢 Eco-Loop Building Agents

> **Live Closed-Loop Physical AI for Intelligent HVAC Optimization**

Eco-Loop is a **Physical AI proof-of-concept** that demonstrates how an open-source Large Language Model (LLM) can autonomously control HVAC systems inside a live **EnergyPlus** building simulation using **tool calling**. The AI continuously monitors building conditions, reasons over sensor data, updates HVAC setpoints through EnergyPlus actuators, and reduces energy consumption while maintaining occupant thermal comfort.

---

## 🎥 Demo Video

**Watch the project demonstration:**

https://www.youtube.com/watch?v=NGmv6PbWYJI

---

## ✨ Features

- 🤖 LLM-powered autonomous HVAC control
- 🔄 Real-time closed-loop EnergyPlus integration
- 🛠 OpenAI-compatible Tool Calling (Groq)
- 🌡 Live HVAC actuator updates
- 🏢 Multi-zone building optimization
- 📊 Interactive Streamlit dashboard
- 📈 Baseline vs AI energy comparison
- 🛡 Predictive guardrails with heuristic fallback
- 📝 CSV & JSONL decision logging
- 🌍 Optional carbon-aware optimization
- 🔌 Optional MCP server integration

---

# 📊 Results

| Metric | Baseline | AI Controller |
|---------|---------:|-------------:|
| Total Electricity | **1759.3 kWh** | **1696.8 kWh** |
| Energy Savings | — | **3.6%** |
| Occupied Comfort | 87.9% | **66.8%** |
| Simulation Period | 30 Days | 30 Days |

The AI reduces electricity consumption by intelligently adjusting cooling setpoints and night setback strategies while maintaining acceptable occupied thermal comfort.

---

# 🏗 System Architecture

```
EnergyPlus
      │
      ▼
Live Sensor Data
      │
      ▼
Building Agent (LLM)
      │
Tool Calling (Groq)
      │
      ▼
Predictive Guardrails
      │
      ▼
HVAC Actuators
      │
      ▼
Updated Building State
      │
      ▼
Dashboard & Logger
      │
      └───────────► Repeat
```

---

# 🚀 Quick Start

## 1. Clone the repository

```bash
git clone https://github.com/cavxn/Eco-Loop-HVAC.git

cd Eco-Loop-HVAC
```

---

## 2. Create Environment

```bash
python3 -m venv .venv

source .venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Configure Environment

Copy the example environment file.

```bash
cp .env.example .env
```

Update:

```text
ENERGYPLUS_ROOT=
GROQ_API_KEY=
LLM_MODEL=llama-3.1-8b-instant
```

Install **EnergyPlus 25.x** before running the project.

---

## 5. Run Baseline Simulation

```bash
PYTHONPATH=. python -m src.run_baseline
```

---

## 6. Run AI Closed Loop

```bash
PYTHONPATH=. python -m src.run_ai_loop
```

---

## 7. Generate Static Dashboard

```bash
PYTHONPATH=. python dashboard/dashboard.py
```

Generated charts are saved in

```
dashboard/output/
```

---

## 8. Launch Live Dashboard

```bash
streamlit run dashboard/streamlit_app.py
```

---

## 9. Optional MCP Server

```bash
pip install mcp

PYTHONPATH=. python -m src.mcp_server
```

---

# 📁 Repository Structure

```
Eco-Loop-HVAC
│
├── dashboard/
│   ├── dashboard.py
│   ├── streamlit_app.py
│   └── output/
│
├── docs/
│   └── ARCHITECTURE.md
│
├── models/
│   └── baseline.idf
│
├── src/
│   ├── building_agent.py
│   ├── carbon.py
│   ├── config.py
│   ├── data_logger.py
│   ├── energyplus_runner.py
│   ├── mcp_server.py
│   ├── run_ai_loop.py
│   ├── run_baseline.py
│   └── tools.py
│
├── requirements.txt
├── .env.example
└── README.md
```

---

# ⚙️ Workflow

1. EnergyPlus starts the simulation.
2. Live sensor data is collected.
3. The Building Agent analyzes the building state.
4. The LLM invokes HVAC control tools using structured Tool Calling.
5. Predictive guardrails validate every AI decision.
6. Approved setpoints are applied to EnergyPlus actuators.
7. Updated sensor values are logged and visualized.
8. The feedback loop repeats throughout the simulation.

---

# 🧠 Technologies

### Programming

- Python 3.12

### Building Simulation

- EnergyPlus 25.2
- pyenergyplus

### Artificial Intelligence

- Llama 3.1 8B
- Groq API
- OpenAI Tool Calling

### Dashboard

- Streamlit
- Matplotlib

### Data

- CSV
- JSONL

### Configuration

- python-dotenv

---

# 📈 Evaluation Highlights

| SIH Criterion | Implementation |
|---------------|----------------|
| System Integration | Stable callback-based control with automatic retry and heuristic fallback |
| Energy Efficiency | 3.6% reduction in electricity consumption over a 30-day simulation |
| Thermal Comfort | Occupied comfort maintained through intelligent HVAC optimization |
| Agentic AI | LLM Tool Calling with structured schemas and optional MCP compatibility |
| Documentation | Complete architecture document and implementation details |

---

# ⚙️ Configuration

Project settings are configured through `.env`.

Key parameters include:

- `ENERGYPLUS_ROOT`
- `GROQ_API_KEY`
- `LLM_MODEL`
- `AGENT_CALLBACK_INTERVAL`
- Comfort temperature bounds
- Optional `CARBON_API_URL`

See `.env.example` for details.

---

# 📄 Documentation

Additional documentation is available in:

```
docs/ARCHITECTURE.md
```

---

# 🔮 Future Work

- Real Building Management System (BMS) integration
- Dynamic electricity pricing optimization
- Renewable energy scheduling
- Reinforcement Learning controllers
- Edge deployment
- Digital Twin integration

---

# 📜 License

This project is licensed under the MIT License.

---

## 👨‍💻 Team

**Eco-Loop Building Agents**

Developed as a Physical AI proof-of-concept for **Smart India Hackathon (SIH) 2026**.
