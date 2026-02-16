# 😠 MAD Architecture Search

**Multi-Agent Discovery System for Neural Architecture Search**

A research framework that uses multiple specialized AI agents to discover and test novel neural network architectures through autonomous experimentation.

## 🎯 What is MAD?

MAD (Multi-Agent Discovery) is an automated research system that:
- **Discovers** computational tricks and architectural patterns from papers
- **Proposes** novel neural architecture experiments
- **Implements** and tests these architectures automatically
- **Learns** from results to guide future experiments

The system runs continuously, coordinating multiple agents to explore the neural architecture design space.

## 🏗️ Architecture

```
┌─────────────────┐
│  Research Agent │  Reads papers, extracts tricks
└────────┬────────┘
         │
┌────────▼────────┐
│ Proposal Agent  │  Designs experiments
└────────┬────────┘
         │
┌────────▼────────┐
│Experiment Agent │  Implements & runs experiments
└────────┬────────┘
         │
┌────────▼────────┐
│   Log Agent     │  Analyzes results, updates knowledge
└─────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 16+ (for dashboard)
- `uv` for Python dependency management

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 1. Run the Backend

```bash
cd backend

# The system will start all agents
uv run python runner.py
```

This starts:
- Research agent (reads papers)
- Experiment agent (runs experiments)
- Log agent (tracks progress)
- Work coordinator

### 2. Run the Dashboard

**Terminal 1 - Start SSE Server:**
```bash
cd dashboard/server
uv run python sse_server.py
```

**Terminal 2 - Start Frontend:**
```bash
cd dashboard/frontend
# TODO: Add proper frontend setup
# For now, the frontend components are here
```

**Terminal 3 - Access Dashboard:**
```bash
# Dashboard available at http://localhost:5173
# Or integrate into your own React app by importing components
```

## 📁 Project Structure

```
mad-architecture-search/
├── backend/
│   ├── agents/              # AI agents (research, experiment, log, etc.)
│   ├── tasks/               # Benchmark tasks for testing architectures
│   ├── code/                # Implemented experiments (numbered)
│   ├── experiments/         # Experiment logs and results
│   ├── proposals/           # Proposed experiments (markdown)
│   ├── tricks/              # Discovered computational tricks
│   ├── papers/              # Research papers (gitignored)
│   └── runner.py            # Main orchestrator
│
├── dashboard/
│   ├── server/              # SSE server for real-time updates
│   └── frontend/            # React dashboard components
│
└── docs/                    # Documentation
```

## 🧪 How It Works

### 1. Research Phase
The research agent reads papers from `backend/papers/` and extracts:
- Novel architectural tricks
- Mathematical innovations
- Implementation patterns

These are stored in `backend/tricks/` as structured markdown files.

### 2. Proposal Phase
The proposal agent:
- Combines tricks in novel ways
- Designs experiments to test hypotheses
- Writes detailed proposals in `backend/proposals/`

### 3. Implementation Phase
The experiment agent:
- Reads approved proposals
- Generates code for the architecture
- Creates training scripts and configs
- Saves to `backend/code/XXX/`

### 4. Execution Phase
The system:
- Trains the model on benchmark tasks
- Logs metrics and results
- Stores outputs in `backend/experiments/`

### 5. Learning Phase
The log agent:
- Analyzes experimental results
- Updates trick ratings based on performance
- Provides feedback for future experiments

## 🎛️ Dashboard Features

The real-time dashboard shows:

- **Agent Status**: What each agent is currently doing
- **Experiments**: Live training metrics and results
- **Proposals**: Queue of proposed experiments
- **Tricks**: Library of discovered patterns
- **Research Log**: Papers read and insights extracted
- **Code Viewer**: Browse generated experiment code

## 🔧 Configuration

Key configuration files:
- `backend/pyproject.toml` - Python dependencies
- `backend/runner.py` - Agent coordination settings
- `dashboard/server/sse_server.py` - Dashboard backend config

## 📊 Benchmark Tasks

The system tests architectures on:
- **Selective Copy**: Token copying with markers
- **In-Context Recall**: Memorization tasks
- **Addition**: Basic arithmetic
- **State Tracking**: Sequence state modeling
- **S5**: Structured state space sequences

Tasks are in `backend/tasks/` - each has its own data generation and evaluation logic.

## 🤝 Contributing

This is a research project exploring autonomous scientific discovery.

Areas for contribution:
- New benchmark tasks
- Additional agent types
- Dashboard improvements
- Architecture search strategies

## 📝 License

[Add your license here]

## 🙏 Credits

Built with:
- [Anthropic Claude](https://www.anthropic.com/) - For agent reasoning
- [Modal](https://modal.com/) - For experiment execution
- [PyTorch](https://pytorch.org/) - For neural network training

## 📖 Learn More

See `docs/` for:
- Detailed agent documentation
- Architecture design guide
- API reference
- Troubleshooting

---

**Status**: Active Research Project
**Last Updated**: February 2026
