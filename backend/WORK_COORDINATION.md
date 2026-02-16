# Work Coordination System

This document explains how multiple experiment agents coordinate to avoid duplicate work.

## Overview

The **Work Tracker** maintains a shared JSON file (`experiments/active_work.json`) that tracks which proposals are currently being worked on by which agents. This prevents multiple agents from implementing the same proposal simultaneously.

## How It Works

### 1. Work Lifecycle

```
┌─────────┐     ┌──────────┐     ┌───────────┐     ┌─────────┐
│ Claim   │ --> │ Heartbeat│ --> │ Heartbeat │ --> │ Release │
│ Proposal│     │ (5 min)  │     │ (5 min)   │     │         │
└─────────┘     └──────────┘     └───────────┘     └─────────┘
```

**Claim**: Agent claims a proposal before starting work
- Writes agent ID, timestamp, and status to `active_work.json`
- Prevents other agents from claiming the same proposal

**Heartbeat**: Agent sends periodic updates every ~5 minutes
- Updates `last_heartbeat` timestamp
- Updates status (e.g., "implementing", "training")
- Signals that work is still in progress

**Release**: Agent releases claim when done
- Removes from active work
- Moves to history with final status ("completed", "failed", "abandoned")

### 2. Stale Work Detection

If an agent crashes or hangs, its work becomes **stale** after 30 minutes without a heartbeat.

The work tracker automatically:
- Detects stale work (no heartbeat in >30 min)
- Removes it from active work
- Makes the proposal available for other agents

### 3. Proposal Selection

When selecting a proposal to work on, agents:

1. Call `select_best_proposal()` which:
   - Reads log agent's prioritizations (impact scores, recommended order)
   - Filters out already implemented experiments
   - **Filters out proposals currently claimed by active agents**
   - Ranks by composite score (impact × 10 + order bonus + priority)

2. Try to claim the selected proposal:
   - If successful → start work
   - If already claimed → skip or select next best

## File Structure

```
experiments/
├── active_work.json     # Tracks current work (shared state)
└── experiment-log.md    # Detailed work journal
```

### `active_work.json` Format

```json
{
  "active_work": {
    "006-monarch-gated-state-transition": {
      "agent_id": "agent-01",
      "proposal_id": "006-monarch-gated-state-transition",
      "started_at": "2026-02-15T11:30:00",
      "last_heartbeat": "2026-02-15T11:35:00",
      "status": "implementing"
    }
  },
  "history": [
    {
      "agent_id": "agent-02",
      "proposal_id": "004-oscillatory-dplr-ssm",
      "started_at": "2026-02-15T10:00:00",
      "completed_at": "2026-02-15T10:45:00",
      "status": "completed"
    }
  ]
}
```

## Usage

### Running the System

The system is run via `runner.py`, which runs all agents including 5 parallel experiment agents:

```bash
# Run the full MAD system (default: 5 parallel experiment agents)
uv run python runner.py

# Run with custom number of experiment agents
uv run python runner.py --num-experiment-agents 3

# Run one iteration of all agents (for testing)
uv run python runner.py --once

# Run one experiment iteration with 5 parallel agents
uv run python runner.py --once-experiment

# Run with custom intervals
uv run python runner.py --experiment-interval 30 --num-experiment-agents 5
```

### Managing Work Tracker

```bash
# List active work
uv run python -m agents.work_tracker list

# Clean stale entries
uv run python -m agents.work_tracker clean

# Manually claim a proposal (for testing)
uv run python -m agents.work_tracker claim 006-monarch-gated-state-transition

# Manually release a proposal
uv run python -m agents.work_tracker release 006-monarch-gated-state-transition
```

### List Proposals with Priority Info

```bash
# Show all proposals with log agent scores
uv run python -m agents.experiment_agent --list-proposals
```

Output format:
```
🔧 001-column-sparse: ✅ MVE | Priority: high | Est: $0.40 | Impact: 5/10 | Order: #3
   006-monarch-gated: ✅ MVE | Priority: high | Est: $0.40 🎯 | Impact: 9/10 | Order: #1
```

Icons:
- 🔧 = Already implemented
- 🎯 = High-impact (mentioned in log agent's top picks)
- ✅ = Has MVE section

## Integration Points

### Experiment Agent (`experiment_agent.py`)

**Before work**:
- Calls `select_best_proposal()` → filters claimed proposals
- Calls `claim_proposal(proposal_id, agent_id)` → registers work

**During work**:
- Starts async heartbeat task → updates every 5 min
- Sends heartbeat with status updates

**After work**:
- Calls `release_proposal(proposal_id, agent_id, status)` → releases claim
- Status can be: "completed", "failed", "abandoned"

### Work Tracker (`work_tracker.py`)

**Core functions**:
- `claim_proposal(proposal_id, agent_id)` → Claim work
- `heartbeat(proposal_id, agent_id, status)` → Update heartbeat
- `release_proposal(proposal_id, agent_id, status)` → Release claim
- `get_claimed_proposals()` → List claimed proposal IDs
- `get_active_work()` → Get all active work (auto-cleans stale)

**File locking**:
- Uses `fcntl.flock()` for atomic read/write
- Prevents race conditions when multiple agents access file

## Best Practices

### For Agent Developers

1. **Always use try/finally to release claims**:
   ```python
   if claim_proposal(proposal_id, agent_id):
       try:
           # Do work
           heartbeat(proposal_id, agent_id)
       finally:
           release_proposal(proposal_id, agent_id)
   ```

2. **Start heartbeat task before long-running work**:
   ```python
   async def periodic_heartbeat():
       while True:
           await asyncio.sleep(300)  # 5 min
           heartbeat(proposal_id, agent_id)

   heartbeat_task = asyncio.create_task(periodic_heartbeat())
   ```

3. **Cancel heartbeat task in finally block**:
   ```python
   finally:
       if heartbeat_task:
           heartbeat_task.cancel()
   ```

### For System Operators

1. **Monitor active work**:
   ```bash
   watch -n 60 "uv run python -m agents.work_tracker list"
   ```

2. **Clean stale work periodically** (optional, auto-cleaned on read):
   ```bash
   uv run python -m agents.work_tracker clean
   ```

3. **Check for stuck agents**:
   - Look for work with no recent heartbeat (>15 min)
   - Check if status is changing or stuck

## Troubleshooting

### Problem: Agent crashed, proposal still claimed

**Solution**: Wait 30 minutes for auto-cleanup, or manually release:
```bash
uv run python -m agents.work_tracker release <proposal_id>
```

### Problem: Multiple agents working on same proposal

**Cause**: Race condition during claiming (very rare)

**Solution**:
1. Stop all agents
2. Clean active work: `rm experiments/active_work.json`
3. Restart with proper coordination

### Problem: Proposal stuck in "implementing" for hours

**Check**:
1. Is agent still running? (`ps aux | grep experiment_agent`)
2. Is heartbeat updating? (`cat experiments/active_work.json | jq '.active_work'`)

**Solution**:
- If agent dead → auto-cleanup after 30 min
- If agent alive but stuck → manually kill and release:
  ```bash
  pkill -f "experiment_agent.*006"
  uv run python -m agents.work_tracker release 006-monarch-gated-state-transition
  ```

## Technical Details

### File Locking Mechanism

Uses POSIX file locks (`fcntl.flock`) to ensure atomic operations:
- **LOCK_EX**: Exclusive lock for writes
- **LOCK_UN**: Release lock
- Blocks until lock available (serializes access)

### Stale Work Detection

```python
HEARTBEAT_TIMEOUT_MINUTES = 30
cutoff = now - timedelta(minutes=30)

for proposal_id, work_info in active_work.items():
    last_heartbeat = datetime.fromisoformat(work_info['last_heartbeat'])
    if last_heartbeat < cutoff:
        # Mark as stale, move to history
```

### Composite Scoring for Selection

```python
composite_score = (
    impact_score * 10        # 0-100 from log agent
    + (10 - min(order, 10))  # 0-10 from recommended order
    + base_priority * 3      # 0-9 from proposal metadata
    + (5 if high_impact else 0)  # +5 bonus for 🎯
)
```

Prioritizes:
1. High impact scores from log agent
2. Top positions in recommended order
3. High base priority from proposals
4. Mentions in "High-Impact Proposals" section

## Future Enhancements

Possible improvements:
- Add agent health monitoring (ping/pong)
- Add work queue with priorities
- Add distributed locking for multi-machine setups
- Add Prometheus metrics for monitoring
- Add automatic retry on failure
- Add work estimation and load balancing
