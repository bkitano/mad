# Claude Instructions for MAD Architecture Search

## Python Environment

**CRITICAL**: This project uses `uv` for Python dependency management.

### Always use `uv run python` for ALL Python commands

- ✅ CORRECT: `uv run python -m agents.experiment_agent`
- ✅ CORRECT: `uv run python runner.py`
- ✅ CORRECT: `uv run python train.py`
- ❌ WRONG: `python -m agents.experiment_agent`
- ❌ WRONG: `python3 runner.py`
- ❌ WRONG: `python train.py`

### Why?

The `uv run` command ensures:
1. All dependencies from `pyproject.toml` are available
2. The correct Python environment is activated
3. Required packages like `claude_agent_sdk` are importable
4. Consistent behavior across different terminals/shells

### Background Processes

When running Python scripts in the background:
```bash
nohup uv run python -m agents.experiment_agent --proposal 006-name > /tmp/exp_006.log 2>&1 &
```

### Running Tests or Scripts

Always prefix with `uv run`:
```bash
uv run python -m pytest tests/
uv run python scripts/analyze.py
uv run python -m agents.trick_search
```

## Project Structure

- `agents/` - Agent implementations (trick search, research, experiment, log)
- `proposals/` - Experiment proposals
- `code/XXX/` - Implemented experiments
- `experiments/` - Experiment logs and results
- `tricks/` - Discovered computational tricks
- `notes/` - Research notes and logs

## Agent Status Tracking

All agents (trick_search, research, experiment, log, scaler) now write their status to:
- `experiments/agent_status.json` - Real-time agent status for dashboard

Status tracking includes:
- Current status: running, waiting, idle, error
- Iteration number
- Next run timestamp
- Additional details per agent type

## Experiment Failure Logging

The experiment agent now ALWAYS creates logs when experiments fail:
- `experiments/experiment-log-XXX.md` - Detailed failure log with stack traces
- `experiments/XXX_results.md` - Results file documenting the failure

This ensures every experiment attempt is documented, even failures.
