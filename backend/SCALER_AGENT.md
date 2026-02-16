# Experiment Scaler Agent

**Purpose**: Automatically scales up successful MVEs (Minimum Viable Experiments) to validate approaches on real benchmarks.

## Overview

The Scaler Agent monitors completed experiments and implements scaled-up versions when MVEs show positive results. It ensures scaled experiments stay within practical resource limits while testing core hypotheses on real-world tasks.

**Key Feature**: Multiple scaler agents can run in parallel, automatically coordinating work to avoid duplication. Each agent claims an experiment before starting and releases it when done.

## Constraints

**Hard limits enforced:**
- **Maximum 2 A100 GPUs** per scaled experiment
- **Maximum 6 hours runtime** (12 A100-hours ≈ $20-30 cost)
- Must fit in GPU memory (80GB per A100)
- Must use real benchmarks (WikiText-103, LRA, MQAR, etc.)

**Typical scaled experiment:**
- Model size: 10M-50M parameters
- Training: Efficient mixed precision with gradient accumulation
- Datasets: Standard benchmarks at appropriate scale
- Logging: W&B or TensorBoard for tracking

## When Experiments Get Scaled

The agent scans `experiments/` for completed MVEs with:
- ✅ **Positive verdict**: "PROCEED", "SUCCESS", "Scale up", etc.
- ✅ **Success criteria met**: MVE validated core hypothesis
- ✅ **No critical blockers**: Implementation works as expected
- ❌ **Not already scaled**: `code/{num}_scaled/` doesn't exist
- ❌ **Not currently claimed**: No other agent is working on it

## Parallel Work Coordination

Multiple scaler agents can run simultaneously without conflicts:

1. **Claiming**: Each agent claims an experiment before starting work
2. **Heartbeat**: Agents send heartbeats every ~10 messages to show they're active
3. **Release**: Agents release claims when done (success or failure)
4. **Stale Detection**: Claims without heartbeats for >30 minutes are auto-released
5. **Work Tracker**: Shared state in `experiments/active_work.json`

This allows running 3-5 parallel scalers to process multiple experiments simultaneously.

## What the Agent Does

For each scalable MVE:

1. **Analyzes** the MVE results and implementation
2. **Reviews** the original proposal's full-scale plans
3. **Designs** a scaled experiment within resource constraints:
   - Model architecture (size, layers, dimensions)
   - Dataset selection and training protocol
   - Hardware requirements and cost estimate
4. **Implements** the scaled experiment:
   - Creates `code/{num}_scaled/` directory
   - Writes training scripts, configs, data loaders
   - Adds documentation and success criteria
   - Includes early stopping and cost tracking

## Running the Scaler Agent

### Standalone Test (Single Agent)
```bash
cd /home/bkitano/Desktop/vault/projects/mad-architecture-search
uv run python runner.py --once-scaler --num-scaler-agents 1
```

### Parallel Test (Multiple Agents)
```bash
uv run python runner.py --once-scaler --num-scaler-agents 3
```

### In Autonomous Loop
The scaler runs automatically every 3 hours with 3 parallel agents:
```bash
uv run python runner.py  # Includes scaler agent
```

### Customize Parallelism and Interval
```bash
# Run 5 parallel scalers every 2 hours
uv run python runner.py --scaler-interval 120 --num-scaler-agents 5
```

### Work Coordination

View active scaling work:
```bash
uv run python -m agents.work_tracker list
```

Clean stale work entries:
```bash
uv run python -m agents.work_tracker clean
```

## Directory Structure

Scaled experiments are created in:
```
code/{num}_scaled/
├── README.md           # Experiment plan, baselines, success criteria
├── config.yaml         # Model and training configuration
├── train.py            # Main training script with multi-GPU support
├── models/             # Scaled model implementation
│   └── scaled_model.py
├── data/               # Data loading for real benchmarks
│   └── dataloader.py
└── scripts/
    ├── run_2gpu.sh     # Multi-GPU training script
    └── eval.sh         # Evaluation script
```

## Current Status

**Experiments ready for scaling:**
- ✅ **Experiment 008**: Found and ready to scale

The agent will automatically implement scaled versions when run.

## Integration with Other Agents

- **Experiment Agent**: Creates MVEs that the scaler can promote
- **Log Agent**: Documents scaling activities and results
- **Research Agent**: Original proposals inform scaling decisions
- **Human Feedback**: GPU efficiency criteria guide scaling design

## Safety Features

1. **Cost estimation**: Every scaled experiment includes projected GPU-hours
2. **Early stopping**: Scripts include logic to halt if results are poor
3. **Memory checks**: Validates model fits in 2×A100 memory before running
4. **Baseline comparisons**: Always includes established model comparisons
5. **GPU efficiency**: Follows human_feedback.md criteria for hardware utilization

## Example Scaling Plan

**MVE**: 1-layer, 16-dim state, 5K params, synthetic task
**Scaled**: 6-layer, 256-dim state, 25M params, WikiText-103
**Resources**: 2×A100, 4 hours, ~$15
**Success**: Matches or beats Mamba-2 baseline on perplexity
