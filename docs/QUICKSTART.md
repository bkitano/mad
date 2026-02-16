# Quick Start Guide

## Installation

### 1. Install Dependencies

```bash
# Install uv (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Node.js (if not already installed)
# Visit https://nodejs.org/
```

### 2. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/mad-architecture-search.git
cd mad-architecture-search
```

## Running the System

### Option A: Full System (All Agents)

```bash
cd backend
uv run python runner.py
```

This starts all agents in coordination:
- Research agent
- Experiment agent
- Log agent
- Work tracker

### Option B: Individual Agents

Run specific agents separately:

```bash
# Research agent (reads papers, extracts tricks)
uv run python -m agents.research_agent

# Experiment agent (runs experiments)
uv run python -m agents.experiment_agent

# Log agent (analyzes results)
uv run python -m agents.log_agent
```

## Dashboard

### Start the Dashboard Server

```bash
cd dashboard/server
uv run python sse_server.py
```

The SSE server provides real-time updates at `http://localhost:8000`

### Integrate Frontend Components

The React components in `dashboard/frontend/` can be imported into your app:

```tsx
import { AgentStatus } from './dashboard/frontend/AgentStatus'
import { ProposalsView } from './dashboard/frontend/ProposalsView'
import { ExperimentCard } from './dashboard/frontend/ExperimentCard'

function Dashboard() {
  return (
    <div>
      <AgentStatus />
      <ProposalsView />
    </div>
  )
}
```

Or build a standalone dashboard - see `dashboard/frontend/README.md` for details.

## First Steps

1. **Add papers**: Put PDF papers in `backend/papers/`
2. **Run research**: `uv run python -m agents.research_agent`
3. **Check tricks**: View extracted tricks in `backend/tricks/`
4. **Generate proposals**: Proposals appear in `backend/proposals/`
5. **Run experiments**: `uv run python -m agents.experiment_agent`
6. **View results**: Check `backend/experiments/` for logs

## Configuration

### Python Environment

The project uses `uv` for dependency management. Always prefix Python commands with `uv run`:

```bash
uv run python script.py  # ✓ Correct
python script.py         # ✗ Wrong
```

### Agent Settings

Edit agent behavior in their respective files:
- `backend/agents/research_agent.py`
- `backend/agents/experiment_agent.py`
- `backend/agents/log_agent.py`

## Next Steps

- Read the [Architecture Guide](ARCHITECTURE.md)
- Learn about [Adding New Tasks](TASKS.md)
- See [Agent Development](AGENTS.md)
- Check [Troubleshooting](TROUBLESHOOTING.md)

## Getting Help

- Check existing experiments in `backend/experiments/`
- Review tricks in `backend/tricks/`
- Read agent logs in `backend/*.log`
- Open an issue on GitHub
