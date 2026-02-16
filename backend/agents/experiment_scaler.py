"""
Experiment Scaler Agent

An agent that takes completed MVEs with "PROCEED" verdicts and implements
scaled-up versions that validate the approach on real benchmarks.

The Experiment Scaler:
1. Scans experiments/ for completed MVEs with positive results
2. Reads the original MVE implementation
3. Designs a scaled-up experiment (larger model, real datasets)
4. Implements the scaled experiment with Modal deployment
5. Stays within resource limits: up to 2 A100 GPUs, 6 hours max

Constraints:
- Default to 1 A100 or T4 for efficiency, use 2 A100s only if needed
- Maximum 6 hours runtime
- Must use real benchmarks (WikiText-103, LRA, etc.)
- Target ~10M-50M parameters (practical scale for validation)
- All training runs on Modal (never locally)

Usage:
    # One-shot scaling cycle
    python -m agents.experiment_scaler

    # Programmatic usage
    from agents.experiment_scaler import run_scaler_cycle
    async for msg in run_scaler_cycle():
        print(msg)
"""

import asyncio
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator, Optional, List, Dict

from claude_agent_sdk import query, ClaudeAgentOptions

from agents.work_tracker import (
    claim_experiment,
    release_experiment,
    heartbeat_experiment,
    is_experiment_claimed,
    get_claimed_experiments,
)


PROJECT_ROOT = Path(__file__).parent.parent
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
CODE_DIR = PROJECT_ROOT / "code"
PROPOSALS_DIR = PROJECT_ROOT / "proposals"
HUMAN_FEEDBACK_FILE = PROJECT_ROOT / "human_feedback.md"


# =============================================================================
# System Prompt
# =============================================================================

SCALER_AGENT_SYSTEM_PROMPT = """You are the Experiment Scaler Agent, an expert in taking successful
proof-of-concept experiments and scaling them to validate real-world performance.

Your goal is to take completed MVEs (Minimum Viable Experiments) that showed positive
results and implement scaled-up versions that test the approach on actual benchmarks.

## Your Knowledge Base

- **Experiments folder**: {experiments_dir} - Contains results files with verdicts
- **Code folder**: {code_dir} - Contains MVE implementations
- **Proposals folder**: {proposals_dir} - Original proposals with full-scale plans
- **Human feedback**: {human_feedback_file} - Constraints and preferences

## When to Scale an Experiment

Scale an MVE if the results file indicates:
- Verdict: "PROCEED", "SUCCESS", "Scale up", etc.
- Success criteria were met
- No critical blockers were discovered
- The approach shows promise for real benchmarks

## Scaling Constraints (CRITICAL)

**Hard limits:**
- Maximum 2 A100 GPUs per experiment (start with 1 A100 or T4 if possible)
- Maximum 6 hours runtime (12 A100-hours = ~$24)
- Must fit in GPU memory (80GB per A100, 16GB per T4)

**GPU Selection:**
- T4: Small models (< 10M params), quick validation runs
- A100: Medium models (10M-50M params), standard for scaled experiments
- 2×A100: Large models (> 50M params) or very long sequences

**This means:**
- Model size: 10M-50M parameters typically
- Batch size: Optimize for throughput within memory
- Dataset: Use standard benchmarks (WikiText-103, LRA, MQAR, etc.)
- Training: Use efficient techniques (mixed precision, gradient accumulation)

## Your Task

For each scalable MVE you find:

1. **Analyze the MVE**: Read the results, understand what worked, what the bottleneck was
2. **Review the proposal**: Check if it has a full-scale experiment plan
3. **Design the scaled experiment**:
   - GPU selection: T4/A100/2×A100 based on model size
   - Model size that fits in selected GPU memory
   - Appropriate dataset and training duration
   - Metrics that validate the core hypothesis
   - Cost estimate (must be < $30)
4. **Implement the experiment**:
   - Create new code directory (code/XXX_scaled/)
   - Create modal_config.py with appropriate GPU configuration
   - Write training script with proper configs
   - Add data loading for real benchmarks
   - Include thorough documentation
   - Add cost tracking and early stopping
5. **Deploy to Modal**:
   - Run `modal run --detach modal_config.py --config config.yaml`
   - Log the Modal job ID for tracking
   - Don't wait for completion - job runs asynchronously

## Scaling Design Principles

**Good scaling decisions:**
- Test on 1-2 real benchmarks (not 5 benchmarks)
- Use small variants of standard datasets (WikiText-103 not WikiText-2)
- Implement efficient baselines for comparison
- Add comprehensive logging (W&B, TensorBoard)
- Include ablations if cheap enough

**Bad scaling decisions:**
- Jumping straight to 1B parameters
- Training for multiple days
- Testing on many datasets simultaneously
- Ignoring memory constraints
- Not planning for early stopping if results are poor

## Implementation Checklist

For each scaled experiment, create:

```
code/{{num}}_scaled/
├── README.md           # Experiment plan, baselines, success criteria
├── config.yaml         # Model/training config
├── modal_config.py     # Modal deployment (REQUIRED)
├── train.py            # Main training script
├── models/             # Model implementation
│   └── scaled_model.py
└── data/               # Data loading
    └── dataloader.py
```

## Modal Deployment (REQUIRED)

**ALL scaled experiments MUST run on Modal**, not locally:

1. **Create modal_config.py**: Use code/001/train/modal_config.py as template
   - Set appropriate GPU type: T4 (cheap), A100 (standard), or 2×A100 (large)
   - Configure N_GPUS (1 or 2) based on model size
   - Set timeout to expected runtime (default 6 hours = 21600 seconds)
   - Include all dependencies and mount volumes

2. **Deploy with detach**: Always run:
   ```bash
   modal run --detach modal_config.py --config config.yaml
   ```
   The `--detach` flag runs asynchronously - don't wait for completion

3. **Multi-GPU setup**: For 2-GPU experiments:
   - Set N_GPUS=2 in modal_config.py
   - Modal handles distributed setup automatically via accelerate
   - No need for separate run_2gpu.sh scripts

4. **Monitor**: Log Modal job ID/URL and check via `modal app logs <app-id>`

## Output Format

When you propose a scaled experiment, write:

1. **Experiment summary**: Which MVE, what was learned, why scale now
2. **Scaling plan**:
   - Model architecture changes (size, layers, dimensions)
   - Dataset and training protocol
   - Hardware requirements (GPUs, memory, time)
   - Cost estimate
3. **Success criteria**: What metrics must be achieved
4. **Implementation**: Actually write the code, configs, and documentation

## Important Notes

- **GPU efficiency matters**: Use the human_feedback.md GPU efficiency criteria
- **Cost tracking**: Add logging to track actual GPU-hours spent
- **Early stopping**: If results are clearly worse than baseline after 1 hour, stop
- **Compare to baselines**: Always compare to existing models (Mamba, attention, etc.)
- **Document everything**: Future experiments will reference this work

Focus on experiments that are most likely to yield publishable results or inform
architectural decisions for production models.
"""


# =============================================================================
# Helper Functions
# =============================================================================

def find_scalable_experiments() -> List[Dict[str, any]]:
    """
    Scan experiments/ for completed MVEs with positive results.

    Returns:
        List of dicts with experiment metadata
    """
    scalable = []

    if not EXPERIMENTS_DIR.exists():
        return scalable

    for results_file in EXPERIMENTS_DIR.glob("*_results.md"):
        try:
            content = results_file.read_text()

            # Extract experiment number
            exp_num = results_file.stem.split('_')[0]

            # Check for positive verdict
            content_lower = content.lower()

            # Look for proceed/success signals
            has_proceed = any(keyword in content_lower for keyword in [
                'proceed', 'success', 'scale up', 'scale to',
                'ready for scaling', 'validated', '✅'
            ])

            # Look for failure signals
            has_failure = any(keyword in content_lower for keyword in [
                'failed', 'kill', 'abandon', 'debug', '❌', 'not working'
            ])

            if has_proceed and not has_failure:
                # Check if already scaled
                scaled_dir = CODE_DIR / f"{exp_num}_scaled"
                if scaled_dir.exists():
                    continue  # Already scaled

                scalable.append({
                    'exp_num': exp_num,
                    'results_file': results_file,
                    'results_content': content,
                    'mve_dir': CODE_DIR / exp_num,
                })

        except Exception as e:
            print(f"Error processing {results_file}: {e}")
            continue

    return scalable


def get_proposal_for_experiment(exp_num: str) -> Optional[str]:
    """
    Find the proposal corresponding to an experiment.

    Args:
        exp_num: Experiment number (e.g., "002")

    Returns:
        Proposal content or None
    """
    # Check MVE README for proposal reference
    mve_dir = CODE_DIR / exp_num
    readme = mve_dir / "README.md"

    if not readme.exists():
        return None

    readme_content = readme.read_text()

    # Extract proposal reference
    for line in readme_content.split('\n')[:20]:
        if 'proposal' in line.lower() and '.md' in line.lower():
            # Extract proposal filename
            import re
            match = re.search(r'\d+-[a-z-]+\.md', line)
            if match:
                proposal_file = PROPOSALS_DIR / match.group(0)
                if proposal_file.exists():
                    return proposal_file.read_text()

    return None


# =============================================================================
# Main Agent Function
# =============================================================================

async def run_scaler_cycle(agent_id: Optional[str] = None) -> AsyncIterator[str]:
    """
    Run one cycle of the experiment scaler:
    1. Find completed MVEs with positive results
    2. Claim one unclaimed experiment
    3. Design and implement scaled version

    Args:
        agent_id: Optional agent ID for coordination

    Yields:
        Status messages
    """
    if agent_id is None:
        agent_id = f"scaler-{uuid.uuid4().hex[:8]}"

    print("=" * 80)
    print(f"[SCALER {agent_id}] Starting scaling cycle")
    print("=" * 80)

    # Find scalable experiments
    scalable_experiments = find_scalable_experiments()

    if not scalable_experiments:
        yield f"[{agent_id}] No experiments ready for scaling found."
        return

    # Filter out already claimed experiments
    claimed = get_claimed_experiments()
    available_experiments = [
        exp for exp in scalable_experiments
        if exp['exp_num'] not in claimed and not is_experiment_claimed(exp['exp_num'])
    ]

    if not available_experiments:
        yield f"[{agent_id}] All scalable experiments are already claimed by other agents."
        return

    print(f"[SCALER {agent_id}] Found {len(available_experiments)} available experiments:")
    for exp in available_experiments:
        print(f"  - Experiment {exp['exp_num']}")

    # Read human feedback
    human_feedback = ""
    if HUMAN_FEEDBACK_FILE.exists():
        human_feedback = HUMAN_FEEDBACK_FILE.read_text()

    # Process ONLY THE FIRST available experiment (one per agent)
    exp = available_experiments[0]
    exp_num = exp['exp_num']

    # Try to claim the experiment
    if not claim_experiment(exp_num, agent_id):
        yield f"[{agent_id}] Failed to claim experiment {exp_num} (claimed by another agent)"
        return

    print(f"\n[SCALER {agent_id}] ✓ Claimed experiment {exp_num}")
    print("-" * 80)

    try:

        # Send initial heartbeat
        heartbeat_experiment(exp_num, agent_id, "analyzing_mve")

        # Read MVE code structure
        mve_dir = exp['mve_dir']
        mve_files = {}

        if mve_dir.exists():
            # Read key files
            for filename in ['README.md', 'config.yaml', 'train.py']:
                filepath = mve_dir / filename
                if filepath.exists():
                    mve_files[filename] = filepath.read_text()

            # Read model code
            models_dir = mve_dir / 'models'
            if models_dir.exists():
                for py_file in models_dir.glob('*.py'):
                    mve_files[f"models/{py_file.name}"] = py_file.read_text()

        # Get proposal
        proposal_content = get_proposal_for_experiment(exp_num)

        # Heartbeat: designing
        heartbeat_experiment(exp_num, agent_id, "designing_scaled_experiment")

        # Build context for scaling
        context = f"""# Experiment {exp_num} Scaling Request

## Results Summary

{exp['results_content']}

## MVE Implementation

"""

        for filename, content in mve_files.items():
            context += f"### {filename}\n\n```\n{content[:2000]}\n```\n\n"

        if proposal_content:
            context += f"## Original Proposal\n\n```\n{proposal_content[:3000]}\n```\n\n"

        context += f"""
## Human Feedback / Constraints

{human_feedback}

## Your Task

Design and implement a scaled-up version of this MVE that:
1. Validates the approach on real benchmarks
2. Stays within 2 A100 GPUs × 6 hours (12 GPU-hours, ~$20-30)
3. Tests the core hypothesis at a practical scale
4. Includes proper baselines and ablations

Implement the complete scaled experiment in code/{exp_num}_scaled/ with all
necessary files (README, config, training script, model code, data loaders).

Make sure to:
- Specify exact model size (parameters count)
- Choose appropriate dataset size
- Estimate memory usage and runtime
- Add early stopping if results are poor
- Include W&B or TensorBoard logging
"""

        print(f"[SCALER {agent_id}] Designing scaled experiment for {exp_num}...")

        # Heartbeat: implementing
        heartbeat_experiment(exp_num, agent_id, "implementing_scaled_code")

        # Call Claude to design and implement scaled experiment
        message_count = 0
        async for message in query(
            prompt=context,
            options=ClaudeAgentOptions(
                model="sonnet",  # Use sonnet for cost efficiency
                system_prompt=SCALER_AGENT_SYSTEM_PROMPT.format(
                    experiments_dir=EXPERIMENTS_DIR,
                    code_dir=CODE_DIR,
                    proposals_dir=PROPOSALS_DIR,
                    human_feedback_file=HUMAN_FEEDBACK_FILE,
                ),
                allowed_tools=["Read", "Write", "Glob", "Bash", "Edit"],
                permission_mode="acceptEdits",
                cwd=str(PROJECT_ROOT),
            )
        ):
            message_count += 1
            # Send heartbeat every 10 messages
            if message_count % 10 == 0:
                heartbeat_experiment(exp_num, agent_id, "implementing_scaled_code")

            # Extract and yield text from message
            if hasattr(message, 'content'):
                for block in getattr(message, 'content', []):
                    if hasattr(block, 'text'):
                        yield block.text

        print(f"[SCALER {agent_id}] ✓ Completed scaling design for experiment {exp_num}")

        # Release with success
        release_experiment(exp_num, agent_id, "completed")

    except Exception as e:
        print(f"[SCALER {agent_id}] ✗ Error scaling experiment {exp_num}: {e}")
        release_experiment(exp_num, agent_id, "failed")
        raise

    finally:
        # Ensure claim is released even if something goes wrong
        if is_experiment_claimed(exp_num):
            release_experiment(exp_num, agent_id, "completed")

    print("\n" + "=" * 80)
    print(f"[SCALER {agent_id}] Scaling cycle complete")
    print("=" * 80)


# =============================================================================
# CLI Entry Point
# =============================================================================

async def main():
    """CLI entry point for running the scaler agent once."""
    print("MAD Architecture Search - Experiment Scaler Agent")
    print()

    async for message in run_scaler_cycle():
        print(message, end='', flush=True)


if __name__ == "__main__":
    asyncio.run(main())
