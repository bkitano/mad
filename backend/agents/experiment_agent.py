"""
Experiment Agent

An agent that implements and runs Minimum Viable Experiments (MVEs) from high-priority proposals.

The Experiment Agent:
1. Reviews proposals (prioritizing those flagged by log agent)
2. Selects high-priority, unimplemented proposals with clear MVEs
3. Creates a new experiment directory in code/
4. Implements the MVE with full training code
5. Runs the experiment if estimated cost < $10
6. Reports results back to the proposal

Usage:
    # One-shot experiment cycle
    python -m agents.experiment_agent

    # Programmatic usage
    from agents.experiment_agent import run_experiment_cycle
    results = await run_experiment_cycle()
"""

import asyncio
import os
import re
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator, Optional, Dict, List

from claude_agent_sdk import query, ClaudeAgentOptions

from agents.work_tracker import (
    claim_proposal,
    heartbeat,
    release_proposal,
    get_claimed_proposals,
)
from agents.proposal_updater import update_proposal_status, get_proposal_status


PROJECT_ROOT = Path(__file__).parent.parent
PROPOSALS_DIR = PROJECT_ROOT / "proposals"
CODE_DIR = PROJECT_ROOT / "code"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
NOTES_DIR = PROJECT_ROOT / "notes"
LOG_FILE = NOTES_DIR / "log.md"


# =============================================================================
# System Prompt
# =============================================================================

EXPERIMENT_AGENT_SYSTEM_PROMPT = """You are the Experiment Agent, an expert in implementing and running
Minimum Viable Experiments (MVEs) to validate architectural ideas quickly and cheaply.

Your goal is to select promising proposals, implement their MVEs, run them, and report results.

## Human Feedback

**IMPORTANT**: Always check `human_feedback.md` at the project root for ongoing instructions, constraints (especially budget/hardware constraints), and preferences from the human researcher. This feedback takes absolute priority.

**CRITICAL**: You MUST maintain a detailed experiment log at experiments/experiment-log-XXX.md (where XXX is your experiment number) documenting:
- Every implementation attempt
- All bugs encountered and how you fixed them
- Design decisions and reasoning
- Training results and metrics
This creates a learning record for the entire system. Each experiment has its own log file.

## Your Knowledge Base

- **Proposals folder**: {proposals_dir} - Contains experiment proposals with MVE sections
- **Code folder**: {code_dir} - Where you create experiment implementations
- **Experiments folder**: {experiments_dir} - Where you log experiment results

## What You Do

1. **Select proposals**: Choose unimplemented proposals with clear MVEs and estimated cost < $10
2. **Create experiment directory**: Make a new numbered directory in code/ (e.g., code/002, code/003)
3. **Start experiment log**: Create experiments/experiment-log-XXX.md (where XXX is your experiment number) and begin logging your work
4. **Implement MVE**: Write minimal but complete code to run the experiment:
   - Model implementation (models/model_name.py)
   - Training script (train.py)
   - Config file (config.yaml)
   - Requirements (pyproject.toml)
   - README with setup instructions
5. **Log everything**: Update experiment-log.md throughout with attempts, bugs, fixes, decisions
6. **Run experiment**: Execute the MVE if cost < $10
7. **Report results**: Create results file in experiments/ with findings

## Code Structure Template

Each experiment directory should follow this structure:

```
code/XXX/
├── README.md              # Setup and run instructions
├── pyproject.toml         # Dependencies
├── config.yaml            # Experiment config
├── modal_config.py        # Modal deployment config (REQUIRED)
├── models/
│   ├── __init__.py
│   └── model_name.py      # Model implementation
├── train.py               # Training script
├── evaluate.py            # (optional) Evaluation script
└── data/                  # (optional) Data generation
    └── generate.py
```

## Modal Deployment (REQUIRED)

**CRITICAL**: ALL experiments MUST run on Modal, NOT locally. Your machine is for orchestration only.

1. **Always create `modal_config.py`**: Use code/001/train/modal_config.py as a template
   - Configure GPU type (T4 for default, A100 for larger experiments, H100 for very large)
   - Set timeout appropriately (default 8 hours)
   - Include all necessary dependencies in the Modal image
   - Mount volumes for saving results

2. **Run via Modal**: Execute experiments using:
   ```bash
   modal run --detach modal_config.py --config config.yaml
   ```
   The `--detach` flag is CRITICAL - it runs the job asynchronously so you don't block.
   NEVER run `python train.py` directly - this runs on CPU and blocks the orchestration loop.

3. **Monitor asynchronously**: With `--detach`, Modal jobs run in the background:
   - Submit the job and it returns immediately
   - Log the Modal job ID and URL from the output
   - The job runs independently on Modal's GPUs
   - Check status via `modal app logs <app-id>` if needed
   - Results are saved to Modal volumes or W&B for later retrieval

4. **Reference implementation**: See code/001/train/modal_config.py for a complete example

## Weights & Biases Logging (REQUIRED)

**CRITICAL**: ALL experiments MUST use Weights & Biases (wandb) for experiment tracking.

1. **Initialize wandb in train.py**:
   ```python
   import wandb

   # At the start of training
   wandb.init(
       project="mad-architecture-search",
       name=f"exp-{{exp_num:03d}}-{{model_name}}",
       config={{
           "model": model_config,
           "training": training_config,
           "proposal_id": proposal_id,
       }}
   )
   ```

2. **Log all metrics**:
   ```python
   # During training loop
   wandb.log({{
       "train/loss": train_loss,
       "train/accuracy": train_acc,
       "val/loss": val_loss,
       "val/accuracy": val_acc,
       "epoch": epoch,
   }})
   ```

3. **Log final results**:
   ```python
   # At experiment completion
   wandb.log({{
       "final/test_accuracy": test_acc,
       "final/best_val_accuracy": best_val_acc,
       "success_criteria/criterion_1": passed_criterion_1,
       "success_criteria/criterion_2": passed_criterion_2,
   }})
   wandb.finish()
   ```

4. **Include wandb URL in logs**: After initializing wandb, capture and log the run URL:
   ```python
   wandb_url = wandb.run.get_url()
   print(f"Wandb URL: {{wandb_url}}")
   ```
   Include this URL in your experiment log entries.

5. **Dependencies**: Add `wandb` to pyproject.toml:
   ```toml
   [project]
   dependencies = ["wandb", "torch", ...]
   ```

**Why wandb is required**:
- Real-time monitoring of running experiments
- Easy comparison across different runs and models
- Persistent metric history
- Accessible to humans for progress tracking
- Automatic visualization of training curves

## Experiment Logging (CRITICAL)

**ALWAYS maintain a detailed experiment log at experiments/experiment-log-XXX.md** (where XXX is your experiment number, e.g., experiment-log-006.md for experiment 006)

Log entries should follow this format:

```markdown
# Experiment Log

## [YYYY-MM-DD HH:MM] Experiment XXX: [Proposal Name]

### Selected Proposal
- **ID**: XXX-proposal-name
- **Priority**: high/medium/low
- **Estimated cost**: $X.XX
- **Reasoning**: Why this proposal was selected

### Implementation Plan
1. Step 1: What you'll implement first
2. Step 2: Next step
3. ...

### [HH:MM] Attempt: [What you tried]
**Goal**: Implement/fix X
**Actions**:
- Created file Y
- Modified Z
- Ran command `...`

**Result**: ✅ Success / ❌ Failed
**Details**: What happened, any output

**Bugs encountered**:
- Bug 1: Error message / symptom
  - Fix: What you did to solve it
- Bug 2: ...

### [HH:MM] Training Run
**Command**: `modal run --detach modal_config.py --config config.yaml`
**Modal Job ID**: app-xxx-yyy
**Modal Job URL**: https://modal.com/apps/...
**Wandb URL**: https://wandb.ai/your-username/mad-architecture-search/runs/...
**Duration**: X minutes (or "running" if detached)
**Metrics**:
- Train loss: X
- Val loss: Y
- Test MSE: Z

**Issues encountered**:
- Issue 1: Description and fix
...

### [HH:MM] Final Results
**Success criteria**:
- ✅ Criterion 1: Met (details)
- ❌ Criterion 2: Failed (reason)

**Decision**: PROCEED / DEBUG / ABANDON
**Reasoning**: Why this decision was made

**Next steps**: What should be done next
```

**Logging Rules**:
1. Log BEFORE starting any major step (creates accountability)
2. Log ALL bugs/errors encountered (don't hide failures)
3. Log the FIX for each bug (creates learning record)
4. Update log in real-time as you work (not retroactively)
5. Be specific: include error messages, commands tried, output snippets
6. Timestamp each entry with [HH:MM]

**IMPORTANT**: Write to YOUR experiment's log file (experiments/experiment-log-XXX.md where XXX is your experiment number), NOT to the shared experiments/experiment-log.md file. Each experiment maintains its own separate log.

## Implementation Guidelines

1. **Keep it minimal**: Only implement what's needed for the MVE, not the full experiment
2. **Use existing code**: Reference code/001 for patterns and utilities you can reuse
3. **Synthetic data preferred**: For MVEs, generate synthetic data rather than downloading large datasets
4. **Fast iteration**: Target < 10 minutes runtime, < $1 cost for MVE
5. **Clear metrics**: Log exactly the metrics specified in the proposal's success criteria
6. **Dependency management**: Use pyproject.toml with minimal dependencies

## Cost Estimation

- **GPU time**: T4 costs ~$0.50/hour, A100 costs ~$2/hour
- **Target**: < 10 minutes = ~$0.08 on T4, ~$0.33 on A100
- **Budget limit**: $10 maximum for any single MVE
- **Synthetic MVEs**: Usually < $1 (5-10 minutes on single T4)
- **Default to T4** unless the experiment needs more compute

## Implementation Strategy

For each MVE:

1. **Read the proposal** thoroughly, especially:
   - Model architecture specifications
   - MVE setup (model size, task, data)
   - Success/failure criteria
   - Expected runtime

2. **Design the implementation**:
   - Start from simplest possible version
   - Use PyTorch (matches code/001)
   - Implement model with clear variable names matching proposal's math
   - Create synthetic data generator if needed
   - Write training loop with proper logging

3. **Create the code**:
   - Make new directory code/XXX (find next available number)
   - Write all files
   - Add clear comments linking code to proposal equations
   - Include run instructions in README

4. **Estimate cost**:
   - Calculate FLOPs roughly
   - Estimate GPU time needed
   - Only proceed if < $10

5. **Deploy to Modal**:
   - Submit job via `modal run --detach modal_config.py --config config.yaml`
   - The `--detach` flag ensures the job runs asynchronously
   - Log the Modal job ID and URL for tracking
   - Results are saved to Modal volumes or W&B
   - Do NOT wait for completion - immediately move on to logging results

6. **Report results**:
   - Create experiments/XXX_results.md
   - Include metrics vs. success criteria
   - Add decision: proceed to full experiment or debug/abandon
   - Update proposal with results link

## Example MVE Implementation (Simplified)

```python
# models/osc_dplr_ssm.py
import torch
import torch.nn as nn

class OscillatoryDPLRSSM(nn.Module):
    \"\"\"
    Oscillatory-DPLR SSM from proposal 004.

    State transition: A = Lambda + P @ Q.T
    where Lambda has oscillatory eigenvalues from damped harmonic oscillator.
    \"\"\"
    def __init__(self, n: int, r: int, d: int):
        super().__init__()
        # Oscillatory parameters (proposal eq. 3)
        self.omega = nn.Parameter(torch.rand(n) * 0.1 + 0.001)  # [0.001, 0.1]
        self.zeta = nn.Parameter(torch.rand(n))  # [0, 1]

        # Low-rank components (proposal eq. 4)
        self.P = nn.Parameter(torch.randn(n, r) * 0.1)
        self.Q = nn.Parameter(torch.randn(n, r) * 0.1)

        # Input/output projections
        self.B = nn.Parameter(torch.randn(n, d))
        self.C = nn.Parameter(torch.randn(d, n))

    def forward(self, u):
        # ... implementation ...
        pass

# train.py (with wandb integration)
import wandb
import torch

# Initialize wandb
wandb.init(
    project="mad-architecture-search",
    name="exp-004-osc-dplr-ssm",
    config={
        "n": 64,
        "r": 4,
        "d": 128,
        "lr": 1e-3,
        "batch_size": 32,
    }
)

# Training loop
for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader)
    val_loss, val_acc = validate(model, val_loader)

    # Log metrics to wandb
    wandb.log({{
        "train/loss": train_loss,
        "val/loss": val_loss,
        "val/accuracy": val_acc,
        "epoch": epoch,
    }})

# Evaluate and log final results
test_loss, test_acc = evaluate(model, test_loader)
wandb.log({{
    "final/test_loss": test_loss,
    "final/test_accuracy": test_acc,
    "success_criteria/criterion_1": test_loss < 1e-3,
    "success_criteria/criterion_2": test_acc > 0.90,
}})

# Print wandb URL and finish
print(f"Wandb URL: {{wandb.run.get_url()}}")
wandb.finish()
```

## When to Skip a Proposal

- No clear MVE section
- MVE requires > $10 estimated cost
- MVE requires external datasets that are hard to obtain
- Dependencies are too complex (e.g., requires specific CUDA kernels)
- Already implemented (check code/ directory)

## Execution Permission

You have permission to:
- Create new directories and files in code/
- Write Python code, configs, and documentation
- Submit Modal jobs to execute training remotely
- Use git to track experiments (optional)

You should NOT:
- Run training locally via `python train.py` - ALWAYS use Modal
- Modify existing code/001 or other numbered experiments
- Run experiments that cost > $10 without user confirmation
- Download large datasets (use synthetic data for MVEs)
- Use more than 1 GPU for MVEs unless justified

## Reporting Format

Create TWO files for each experiment:

### 1. experiments/experiment-log.md (detailed journal)
Append your detailed work log (see Experiment Logging section above)

### 2. experiments/XXX_results.md (summary report)

```markdown
# Experiment XXX Results: [Proposal Name]

**Proposal**: proposals/XXX-name.md
**Code**: code/XXX/
**Experiment Log**: See experiments/experiment-log.md
**Date**: YYYY-MM-DD
**Cost**: $X.XX
**Runtime**: X minutes

## Setup

[Brief description of what was implemented]

## Results

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| Training MSE | < 1e-3 | 2.3e-4 | ✅ Pass |
| Extrapolation MSE | < 1e-2 | 8.7e-3 | ✅ Pass |
| ... | ... | ... | ... |

## Success Criteria

- ✅ Criterion 1: [details]
- ✅ Criterion 2: [details]
- ❌ Criterion 3: [details]

## Decision

**PROCEED** / **DEBUG** / **ABANDON**

[Explanation of decision based on results]

## Next Steps

[What should be done next based on the outcome]
```
"""


# =============================================================================
# Helper Functions
# =============================================================================

def list_proposals() -> List[Dict]:
    """List all proposals with their metadata."""
    proposals = []
    if not PROPOSALS_DIR.exists():
        return proposals

    for filepath in sorted(PROPOSALS_DIR.glob("*.md")):
        content = filepath.read_text()
        proposal = {
            "id": filepath.stem,
            "path": str(filepath),
            "content": content,
        }

        # Parse metadata - check both YAML frontmatter and inline markdown formats
        lines = content.split("\n")

        # Try parsing YAML frontmatter first (between --- markers)
        if lines and lines[0].strip() == '---':
            in_frontmatter = True
            for i, line in enumerate(lines[1:], start=1):
                if line.strip() == '---':
                    break
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower()
                    value = value.strip()
                    if key == 'status':
                        proposal['status'] = value
                    elif key == 'priority':
                        proposal['priority'] = value
                    elif key == 'created':
                        proposal['created'] = value

        # Fallback: try inline markdown format
        if 'status' not in proposal:
            for line in lines:
                if line.startswith("**Status**:"):
                    proposal["status"] = line.split(":", 1)[1].strip()
                elif line.startswith("**Priority**:"):
                    proposal["priority"] = line.split(":", 1)[1].strip()
                elif line.startswith("**Created**:"):
                    proposal["created"] = line.split(":", 1)[1].strip()

        # Check if MVE section exists
        proposal["has_mve"] = "## Minimum Viable Experiment" in content

        proposals.append(proposal)

    return proposals


def get_next_experiment_number() -> int:
    """
    DEPRECATED: Get the next available experiment number.

    NOTE: We now use proposal numbers directly as experiment numbers.
    This function is kept for backwards compatibility but should not be used.
    """
    if not CODE_DIR.exists():
        return 1

    existing = [int(d.name) for d in CODE_DIR.iterdir() if d.is_dir() and d.name.isdigit()]
    return max(existing, default=0) + 1


def get_proposal_number(proposal_id: str) -> int:
    """
    Extract the proposal number from a proposal ID.

    Args:
        proposal_id: Proposal ID like "006-monarch-gated-state-transition"

    Returns:
        The proposal number (e.g., 6)
    """
    import re
    match = re.match(r'(\d+)-', proposal_id)
    if match:
        return int(match.group(1))
    raise ValueError(f"Could not extract proposal number from: {proposal_id}")


def list_implemented_experiments() -> List[int]:
    """List experiment numbers that have been implemented."""
    if not CODE_DIR.exists():
        return []

    return [int(d.name) for d in CODE_DIR.iterdir() if d.is_dir() and d.name.isdigit()]


def get_experiment_results(exp_id: str) -> Optional[str]:
    """Get results for an experiment if they exist."""
    results_file = EXPERIMENTS_DIR / f"{exp_id}_results.md"
    if results_file.exists():
        return results_file.read_text()
    return None


def parse_log_prioritizations() -> Dict[int, Dict]:
    """
    Parse log.md to extract log agent's impact scores and priorities.

    Returns:
        Dict mapping proposal number to priority info:
        {
            1: {"impact_score": 9, "priority": "high", "mentioned": True, "recommendation": "..."},
            ...
        }
    """
    if not LOG_FILE.exists():
        return {}

    try:
        log_content = LOG_FILE.read_text()
    except Exception as e:
        print(f"[WARNING] Could not read log file: {e}")
        return {}

    prioritizations = {}

    # Split by log entries (separated by "##" timestamps)
    entries = log_content.split('\n## ')

    if not entries:
        return {}

    # Parse the most recent entry (first after split, or second if first is header)
    recent_entry = entries[1] if len(entries) > 1 else entries[0]

    # Extract proposal numbers mentioned with impact scores
    # Look for patterns like "Proposal 006", "proposal 004", etc.
    import re

    # Pattern: **Impact score**: **X/10**
    impact_pattern = r'\*\*Impact score\*\*:\s*\*\*(\d+)/10\*\*'

    # Find all proposals mentioned in High-Impact section
    high_impact_section = ""
    if "### 🎯 High-Impact Proposals" in recent_entry:
        section_start = recent_entry.index("### 🎯 High-Impact Proposals")
        # Find next ### section or end
        next_section = recent_entry.find("\n###", section_start + 10)
        if next_section == -1:
            high_impact_section = recent_entry[section_start:]
        else:
            high_impact_section = recent_entry[section_start:next_section]

    # Extract proposal numbers and scores from high-impact section
    proposal_blocks = high_impact_section.split('\n- **')

    for block in proposal_blocks[1:]:  # Skip first empty/header
        # Extract proposal number
        proposal_num_match = re.search(r'[Pp]roposal\s*(\d+)', block)
        if not proposal_num_match:
            continue

        proposal_num = int(proposal_num_match.group(1))

        # Extract impact score
        impact_match = re.search(impact_pattern, block)
        impact_score = int(impact_match.group(1)) if impact_match else 5

        # Extract priority
        priority_match = re.search(r'[Pp]riority:\s*\*?\*?([A-Z]+)\*?\*?', block)
        priority = priority_match.group(1).lower() if priority_match else "medium"

        # Extract a snippet of the reasoning
        lines = block.split('\n')
        reasoning = ""
        for line in lines[:5]:
            if "Why it matters" in line and ":" in line:
                parts = line.split("Why it matters:", 1)
                if len(parts) > 1:
                    reasoning = parts[1].strip()
                break

        prioritizations[proposal_num] = {
            "impact_score": impact_score,
            "priority": priority,
            "mentioned_in_high_impact": True,
            "reasoning": reasoning[:200]  # First 200 chars
        }

    # Also check Strategic Insights for recommended order
    strategic_section = ""
    if "### Strategic Insights" in recent_entry or "Strategic Insights" in recent_entry:
        strategic_start = recent_entry.find("Strategic Insights")
        if strategic_start != -1:
            strategic_section = recent_entry[strategic_start:strategic_start+2000]

    # Look for "Revised priority order" or "Recommended focus order"
    if "priority order" in strategic_section.lower() or "focus order" in strategic_section.lower():
        # Find numbered list
        order_lines = [l for l in strategic_section.split('\n') if re.match(r'^\d+\.', l.strip())]
        for idx, line in enumerate(order_lines):
            proposal_match = re.search(r'\((\d+)\)', line)
            if proposal_match:
                prop_num = int(proposal_match.group(1))
                if prop_num not in prioritizations:
                    prioritizations[prop_num] = {
                        "impact_score": 5,
                        "priority": "medium",
                        "mentioned_in_high_impact": False,
                        "reasoning": ""
                    }
                prioritizations[prop_num]["recommended_order"] = idx + 1

    return prioritizations


def estimate_cost(proposal_content: str) -> float:
    """
    Rough cost estimation from proposal content.

    Looks for phrases like "< 5 minutes", "Single A100", etc.
    """
    content_lower = proposal_content.lower()

    # Look for explicit cost/time mentions in MVE section
    mve_section = ""
    if "## minimum viable experiment" in content_lower:
        idx = content_lower.index("## minimum viable experiment")
        mve_section = content_lower[idx:idx+2000]  # Next ~2000 chars

    # Parse time estimates
    minutes_match = re.search(r'<\s*(\d+)\s*minutes?', mve_section)
    hours_match = re.search(r'<\s*(\d+)\s*hours?', mve_section)

    estimated_hours = 0.0
    if minutes_match:
        estimated_hours = int(minutes_match.group(1)) / 60
    elif hours_match:
        estimated_hours = int(hours_match.group(1))
    else:
        # Default conservative estimate for MVE
        estimated_hours = 0.2  # 12 minutes

    # A100 cost: ~$2/hour
    estimated_cost = estimated_hours * 2.0

    return estimated_cost


def select_best_proposal(proposals: List[Dict], implemented: List[int]) -> Optional[Dict]:
    """
    Select the best proposal to implement next.

    Priority (in order):
    1. Status = "proposed" (not started)
    2. Has MVE section
    3. Not currently being worked on by another agent
    4. Log agent impact score (if available)
    5. Log agent recommended order (if available)
    6. Proposal's own priority field
    7. Not yet implemented
    8. Estimated cost < $10
    """
    # Get log agent's prioritizations
    log_priorities = parse_log_prioritizations()

    # Get proposals currently being worked on
    claimed_proposal_ids = get_claimed_proposals()
    print(f"[SELECTION] Currently claimed proposals: {claimed_proposal_ids}")

    candidates = []

    for proposal in proposals:
        # Parse ID number from filename (e.g., "001-name" -> 1)
        id_match = re.match(r'(\d+)-', proposal['id'])
        if not id_match:
            continue

        exp_num = int(id_match.group(1))

        # Filter criteria
        if exp_num in implemented:
            continue
        if proposal['id'] in claimed_proposal_ids:
            continue  # Skip proposals being worked on by other agents
        if not proposal.get('has_mve', False):
            continue
        if proposal.get('status', '').lower() != 'proposed':
            continue

        # Estimate cost
        cost = estimate_cost(proposal['content'])
        if cost > 10.0:
            continue

        # Base priority score from proposal metadata
        base_priority_score = {'high': 3, 'medium': 2, 'low': 1}.get(
            proposal.get('priority', 'low').lower(), 0
        )

        # Incorporate log agent's assessment
        log_info = log_priorities.get(exp_num, {})
        impact_score = log_info.get('impact_score', 5)  # Default 5/10
        recommended_order = log_info.get('recommended_order', 999)  # Lower is better
        mentioned_in_high_impact = log_info.get('mentioned_in_high_impact', False)

        # Calculate composite score
        # Impact score: 0-10 -> weighted heavily
        # Recommended order: 1-N -> inverse weighted
        # Base priority: 1-3 -> fallback for proposals not in log
        # Mentioned in high-impact: +5 bonus

        composite_score = (
            impact_score * 10  # 0-100 points from impact
            + (10 - min(recommended_order, 10))  # 0-10 points from order (lower order = higher score)
            + base_priority_score * 3  # 0-9 points from base priority
            + (5 if mentioned_in_high_impact else 0)  # 5 point bonus for high-impact mention
        )

        candidates.append({
            **proposal,
            'exp_num': exp_num,
            'cost_estimate': cost,
            'base_priority': base_priority_score,
            'log_impact_score': impact_score,
            'log_recommended_order': recommended_order,
            'composite_score': composite_score,
            'mentioned_in_log': mentioned_in_high_impact,
            'log_reasoning': log_info.get('reasoning', ''),
        })

    # Sort by composite score (descending), then by cost (ascending for tie-breaking)
    candidates.sort(key=lambda x: (-x['composite_score'], x['cost_estimate']))

    return candidates[0] if candidates else None


# =============================================================================
# Experiment Agent
# =============================================================================

async def run_experiment_cycle(
    specific_proposal: Optional[str] = None,
    agent_id: Optional[str] = None,
) -> AsyncIterator[Dict]:
    """
    Run an experiment cycle: select proposal, implement MVE, run, report.

    Args:
        specific_proposal: Optional proposal ID to implement (e.g., "001-name")
        agent_id: Optional agent ID for work tracking

    Returns:
        Dict with experiment results summary
    """
    import uuid
    if agent_id is None:
        agent_id = f"agent-{uuid.uuid4().hex[:8]}"

    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    CODE_DIR.mkdir(parents=True, exist_ok=True)

    # List proposals and implemented experiments
    proposals = list_proposals()
    implemented = list_implemented_experiments()

    if specific_proposal:
        # Find specific proposal
        selected = next((p for p in proposals if p['id'] == specific_proposal), None)
        if not selected:
            print(f"ERROR: Proposal {specific_proposal} not found")
            return
    else:
        # Auto-select best proposal
        selected = select_best_proposal(proposals, implemented)
        if not selected:
            print("ERROR: No suitable proposals found to implement")
            return

    proposal_id = selected['id']

    # Try to claim the proposal
    if not claim_proposal(proposal_id, agent_id):
        print(f"ERROR: Proposal {proposal_id} is already claimed by another agent")
        return

    # Use proposal number as experiment number
    exp_num = get_proposal_number(proposal_id)
    exp_dir = CODE_DIR / f"{exp_num:03d}"

    # Check if experiment already exists
    if exp_dir.exists():
        print(f"ERROR: Experiment {exp_num:03d} already exists for proposal {proposal_id}")
        release_proposal(proposal_id, agent_id, status="skipped")

        # Update proposal status to on-hold (experiment already exists)
        proposal_file = PROPOSALS_DIR / f"{proposal_id}.md"
        update_proposal_status(proposal_file, "on-hold")

        return

    proposal_path = selected['path']
    cost_estimate = selected.get('cost_estimate', estimate_cost(selected['content']))

    # Build selection reasoning
    selection_info = f"""File: {proposal_path}
Priority: {selected.get('priority', 'unknown')}
Estimated cost: ${cost_estimate:.2f}"""

    if selected.get('mentioned_in_log', False):
        selection_info += f"""
Log Agent Impact Score: {selected.get('log_impact_score', 'N/A')}/10
Log Agent Assessment: {selected.get('log_reasoning', 'See log for details')}"""

    if selected.get('log_recommended_order', 999) < 10:
        selection_info += f"""
Recommended Order: #{selected.get('log_recommended_order')} in log agent's priority list"""

    prompt = f"""Implement the Minimum Viable Experiment from this proposal:

## Proposal
{selection_info}

## Task

1. **Read the proposal** at {proposal_path}
   - Focus on the "Minimum Viable Experiment" section
   - Note the model architecture, task, success criteria

2. **Create experiment directory** at code/{exp_num:03d}/
   - Follow the code structure template
   - Write minimal but complete implementation
   - Reference code/001/ for patterns

3. **Start your experiment log** at experiments/experiment-log-{exp_num:03d}.md
   - Document every step of your implementation
   - Log all bugs and fixes
   - This is YOUR log file - do NOT write to experiments/experiment-log.md

4. **Implement the MVE**:
   - Model in models/
   - Training script train.py
   - Config file config.yaml
   - README.md with instructions
   - pyproject.toml with dependencies

5. **Run the experiment**:
   - Install dependencies
   - Execute training
   - Monitor for success/failure criteria
   - Save logs and results

6. **Report results** in experiments/{exp_num:03d}_results.md:
   - Metrics vs. success criteria
   - Decision: PROCEED / DEBUG / ABANDON
   - Next steps

## Guidelines

- Keep it minimal - only what's needed for MVE
- Use synthetic data if possible
- Clear comments linking code to proposal math
- Target < 10 minutes runtime
- Report all metrics from success criteria

## Budget Check

Estimated cost: ${cost_estimate:.2f}
Budget limit: $10.00
Status: {'✅ Proceed' if cost_estimate < 10 else '❌ Skip - too expensive'}

Begin by reading the proposal, then implement the MVE.
"""

    results = {
        "proposal_id": proposal_id,
        "experiment_num": exp_num,
        "estimated_cost": cost_estimate,
        "messages": [],
        "agent_id": agent_id,
    }

    # Start heartbeat task
    heartbeat_task = None

    async def periodic_heartbeat():
        """Send heartbeat every 5 minutes."""
        while True:
            await asyncio.sleep(300)  # 5 minutes
            heartbeat(proposal_id, agent_id, status="in_progress")

    try:
        # Start heartbeat
        heartbeat_task = asyncio.create_task(periodic_heartbeat())
        heartbeat(proposal_id, agent_id, status="starting")

        # Update proposal status to ongoing and add experiment metadata
        proposal_file = PROPOSALS_DIR / f"{proposal_id}.md"
        update_proposal_status(proposal_file, "ongoing", {
            'experiment_number': f"{exp_num:03d}",
            'experiment_log': f"experiment-log-{exp_num:03d}.md"
        })

        async for message in query(
            prompt=prompt,
            options=ClaudeAgentOptions(
                model="opus",
                system_prompt=EXPERIMENT_AGENT_SYSTEM_PROMPT.replace(
                    "{proposals_dir}", str(PROPOSALS_DIR)
                ).replace(
                    "{code_dir}", str(CODE_DIR)
                ).replace(
                    "{experiments_dir}", str(EXPERIMENTS_DIR)
                ),
                allowed_tools=["Read", "Write", "Glob", "Grep", "Bash"],
                permission_mode="acceptEdits",
                cwd=str(PROJECT_ROOT),
            )
        ):
            # Collect messages for logging
            if hasattr(message, 'content'):
                for block in getattr(message, 'content', []):
                    if hasattr(block, 'text'):
                        results["messages"].append(block.text)
            yield message

        # Check if results file was created
        results_file = EXPERIMENTS_DIR / f"{exp_num:03d}_results.md"
        if not results_file.exists():
            # Agent completed but didn't create results - document this
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            exp_log_file = EXPERIMENTS_DIR / f"experiment-log-{exp_num:03d}.md"

            # Check if experiment log exists
            if not exp_log_file.exists():
                # No log either - create a minimal failure log
                incomplete_log = f"""# Experiment Log - {exp_num:03d}: {proposal_id}

## [{timestamp}] Experiment Incomplete

**Proposal**: {proposal_id}
**Agent**: {agent_id}
**Status**: INCOMPLETE

### Issue
The experiment agent completed execution without raising an error, but did not produce:
- An experiment log file documenting the work
- A results file with metrics and conclusions

This suggests the agent may have:
- Failed to understand the task
- Encountered issues but didn't report them properly
- Got stuck or timed out without proper error handling

### Next Steps
- Check the agent's message output for clues
- Review the proposal to ensure MVE specification is clear
- Consider retrying with more explicit instructions
"""
                try:
                    with open(exp_log_file, 'w') as f:
                        f.write(incomplete_log)
                except Exception as e:
                    print(f"[{agent_id}] Failed to write incomplete log: {e}")

            # Create placeholder results file
            incomplete_results = f"""# Experiment {exp_num:03d} Results: INCOMPLETE

**Proposal**: proposals/{proposal_id}.md
**Code**: code/{exp_num:03d}/
**Experiment Log**: experiments/experiment-log-{exp_num:03d}.md
**Date**: {datetime.now().strftime("%Y-%m-%d")}
**Agent**: {agent_id}

## Status

**INCOMPLETE** - The experiment agent completed without creating a results file.

## What Happened

The experiment agent finished execution without raising an error, but did not produce
a results file documenting the experiment's outcome.

This could mean:
- The agent didn't successfully run the experiment
- The experiment ran but results weren't properly recorded
- The agent encountered issues but didn't report them

## Next Steps

1. **Check experiment log** - Review `experiments/experiment-log-{exp_num:03d}.md` for details
2. **Check code directory** - See if any implementation was created in `code/{exp_num:03d}/`
3. **Review agent output** - Check the agent's message history for clues
4. **Retry** - Consider re-running this experiment with clearer instructions

## Decision

**DEBUG** - This experiment needs investigation before being marked as complete.
"""
            try:
                with open(results_file, 'w') as f:
                    f.write(incomplete_results)
                print(f"[{agent_id}] Created placeholder results file for incomplete experiment")
            except Exception as e:
                print(f"[{agent_id}] Failed to write placeholder results: {e}")

            # Mark as failed since no results were produced
            release_proposal(proposal_id, agent_id, status="failed")
            proposal_file = PROPOSALS_DIR / f"{proposal_id}.md"
            update_proposal_status(proposal_file, "incomplete", {
                'results_file': f"{exp_num:03d}_results.md",
                'experiment_log': f"experiment-log-{exp_num:03d}.md"
            })
        else:
            # Success - results file exists
            release_proposal(proposal_id, agent_id, status="completed")

            # Update proposal status to completed and add results file
            proposal_file = PROPOSALS_DIR / f"{proposal_id}.md"
            update_proposal_status(proposal_file, "completed", {
                'results_file': f"{exp_num:03d}_results.md"
            })

    except Exception as e:
        # Failure - document what went wrong
        import traceback
        error_details = traceback.format_exc()

        print(f"[{agent_id}] Error during experiment: {e}")

        # Write failure log to experiment-log file
        exp_log_file = EXPERIMENTS_DIR / f"experiment-log-{exp_num:03d}.md"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

        failure_log = f"""# Experiment Log - {exp_num:03d}: {proposal_id}

## [{timestamp}] Experiment Failed

**Proposal**: {proposal_id}
**Agent**: {agent_id}
**Status**: FAILED

### Error Details

```
{error_details}
```

### Error Message
{str(e)}

### Context
The experiment agent encountered an error during implementation or execution.
This prevented the MVE from being completed.

### Next Steps
- Review the error message above
- Check if this is a bug in the agent's implementation
- Check if the proposal's MVE specification is incomplete or unclear
- Consider whether this proposal needs to be revised
"""

        try:
            with open(exp_log_file, 'w') as f:
                f.write(failure_log)
            print(f"[{agent_id}] Wrote failure log to {exp_log_file}")
        except Exception as log_error:
            print(f"[{agent_id}] Failed to write experiment log: {log_error}")

        # Write failure results file
        results_file = EXPERIMENTS_DIR / f"{exp_num:03d}_results.md"
        failure_results = f"""# Experiment {exp_num:03d} Results: FAILED

**Proposal**: proposals/{proposal_id}.md
**Code**: code/{exp_num:03d}/ (not created or incomplete)
**Experiment Log**: experiments/experiment-log-{exp_num:03d}.md
**Date**: {datetime.now().strftime("%Y-%m-%d")}
**Agent**: {agent_id}

## Status

**FAILED** - The experiment agent encountered an error during implementation or execution.

## Error Summary

```
{str(e)}
```

## Error Type

`{type(e).__name__}`

## What Happened

The experiment agent attempted to implement and run the MVE from proposal {proposal_id},
but encountered an error that prevented completion.

## Full Error Details

See the experiment log file for complete stack trace and error context:
- `experiments/experiment-log-{exp_num:03d}.md`

## Next Steps

1. **Review the error** - Check if this is a systematic issue (e.g., missing dependencies, API errors) or proposal-specific
2. **Check the proposal** - Does the MVE specification have all necessary details?
3. **Debug or revise** - Either fix the agent's implementation approach or revise the proposal
4. **Retry** - Once issues are addressed, the proposal can be retried

## Decision

**DEBUG** - This experiment needs debugging before it can be completed.

The proposal should remain available for retry after addressing the root cause.
"""

        try:
            with open(results_file, 'w') as f:
                f.write(failure_results)
            print(f"[{agent_id}] Wrote failure results to {results_file}")
        except Exception as results_error:
            print(f"[{agent_id}] Failed to write results file: {results_error}")

        # Release proposal and update status
        release_proposal(proposal_id, agent_id, status="failed")

        # Update proposal status to failed
        proposal_file = PROPOSALS_DIR / f"{proposal_id}.md"
        update_proposal_status(proposal_file, "failed", {
            'results_file': f"{exp_num:03d}_results.md",
            'experiment_log': f"experiment-log-{exp_num:03d}.md"
        })

        raise

    finally:
        # Stop heartbeat
        if heartbeat_task:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass


# =============================================================================
# CLI Entry Point
# =============================================================================

async def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Experiment Agent - Implement and run MVEs from proposals"
    )
    parser.add_argument(
        "--proposal", "-p",
        help="Specific proposal ID to implement (e.g., '001-column-sparse')"
    )
    parser.add_argument(
        "--list-proposals",
        action="store_true",
        help="List available proposals and exit"
    )
    parser.add_argument(
        "--list-experiments",
        action="store_true",
        help="List implemented experiments and exit"
    )

    args = parser.parse_args()

    if args.list_proposals:
        proposals = list_proposals()
        implemented = list_implemented_experiments()
        log_priorities = parse_log_prioritizations()

        print("\n" + "=" * 60)
        print("Available Proposals")
        print("=" * 60)
        for p in proposals:
            status_icon = "✅" if p.get('has_mve') else "❌"
            exp_num = int(re.match(r'(\d+)', p['id']).group(1)) if re.match(r'(\d+)', p['id']) else None
            impl_icon = "🔧" if exp_num and exp_num in implemented else "  "
            cost = estimate_cost(p['content'])

            # Add log agent info if available
            log_info = ""
            if exp_num and exp_num in log_priorities:
                log_data = log_priorities[exp_num]
                impact = log_data.get('impact_score', '?')
                order = log_data.get('recommended_order', '')
                highlight = "🎯" if log_data.get('mentioned_in_high_impact') else ""
                log_info = f" | Impact: {impact}/10"
                if order and order < 10:
                    log_info += f" | Order: #{order}"
                if highlight:
                    log_info = f" {highlight}" + log_info

            print(f"{impl_icon} {p['id']}: {status_icon} MVE | "
                  f"Priority: {p.get('priority', '?')} | "
                  f"Est: ${cost:.2f}{log_info}")
        return

    if args.list_experiments:
        experiments = list_implemented_experiments()
        print("\n" + "=" * 60)
        print("Implemented Experiments")
        print("=" * 60)
        for exp_num in sorted(experiments):
            exp_dir = CODE_DIR / f"{exp_num:03d}"
            results = get_experiment_results(f"{exp_num:03d}")
            results_icon = "📊" if results else "  "
            print(f"{results_icon} {exp_num:03d}: {exp_dir}")
        return

    print("\n" + "=" * 60)
    print(" Experiment Agent - MVE Implementation Cycle")
    print("=" * 60)

    if args.proposal:
        print(f"\nImplementing specific proposal: {args.proposal}\n")
    else:
        print(f"\nAuto-selecting best proposal to implement...\n")

    result = {}
    async for msg in run_experiment_cycle(specific_proposal=args.proposal):
        if hasattr(msg, 'content'):
            for block in getattr(msg, 'content', []):
                if hasattr(block, 'text'):
                    print(block.text, end="", flush=True)
        elif hasattr(msg, 'result'):
            print(f"\n\nDone! {msg.result}")

    print("\n")


if __name__ == "__main__":
    asyncio.run(main())
