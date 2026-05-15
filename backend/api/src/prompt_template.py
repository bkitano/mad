"""
Agent prompt template — wraps proposal content with criteria, constraints, and workflow.

The raw proposal markdown is wrapped in a structured prompt that:
  1. Shows the agent its success criteria as a table
  2. Constrains which files it can edit
  3. Specifies the model interface contract
  4. Defines the workflow (implement -> train -> evaluate)
"""

from __future__ import annotations

from typing import Optional


def build_agent_prompt(
    proposal_content: str,
    criteria: Optional[dict],
    experiment_id: str,
    proposal_id: str,
    service_url: str = "",
) -> str:
    """Build the full agent prompt from proposal content and criteria.

    Args:
        proposal_content: Raw markdown proposal text.
        criteria: Parsed ExperimentCriteria dict (from proposal.criteria JSONB),
                  or None if no criteria are attached.
        experiment_id: The experiment ID (e.g. "042").
        proposal_id: The proposal ID slug.
        service_url: MAD API base URL for verdict reporting.

    Returns:
        Complete prompt string to send to the agent.
    """
    if criteria is None:
        # No harness — fall back to raw proposal (backwards compatible)
        return proposal_content

    criteria_table = _render_criteria_table(criteria)
    allowed = ", ".join(f"`{p}`" for p in criteria.get("allowed_edit_paths", ["models/", "configs/"]))
    forbidden = ", ".join(f"`{p}`" for p in criteria.get("forbidden_edit_paths", ["train/", "tasks/", "harness/", "evaluate.py"]))
    time_budget = criteria.get("time_budget_minutes", 30)
    baseline_config = criteria.get("baseline_config")
    baseline_section = f"Baseline config for comparison: `{baseline_config}`" if baseline_config else "No baseline config specified."

    report_cmd = ""
    if service_url:
        report_cmd = f"""
6. **Report results**: After evaluation, report the verdict:
   ```bash
   curl -X POST {service_url}/experiments/{experiment_id}/verdict \\
     -H "Content-Type: application/json" \\
     -d @verdict.json
   ```"""

    return f"""You are an ML researcher agent. You have been assigned an experiment.

## Your Task

{proposal_content}

## Success Criteria

Your experiment will be automatically evaluated against these criteria.
Criteria marked with * are **required** — all must pass for the experiment to succeed.

{criteria_table}

## Rules

1. **You may ONLY create/modify files in**: {allowed}
   This means you can create new model architectures in `models/` and
   training configs in `configs/`.

2. **You MUST NOT edit files in**: {forbidden}
   The training loop, task definitions, and evaluation script are fixed
   infrastructure. Do not modify them.

3. **Model interface contract**: Your model class must follow this interface:
   ```python
   class YourModel(nn.Module):
       def __init__(
           self,
           num_tokens: int,    # total vocab size (group elements + BOS/EOS/PAD)
           num_classes: int,    # number of output classes (group elements)
           eos_idx: int,        # index of EOS token
           max_seq_len: int,
           d_model: int,
           nhead: int,
           num_layers: int,
           dropout: float,
           **kwargs,            # accept additional config params
       ):
           ...

       def forward(self, tokens: Tensor, mask: Tensor) -> Tensor:
           # tokens: (batch, seq_len)  — token indices
           # mask:   (batch, seq_len)  — 1 for real tokens, 0 for padding
           # returns: (batch, seq_len, num_classes) — logits at each position
           ...
   ```

4. **Register your model** in `models/__init__.py` and reference it by name
   in your config's `model.type` field.

## Workflow

1. **Read** the existing codebase: understand `models/`, `configs/`, `tasks/`, `train/`
2. **Implement** your model architecture in `models/your_model.py`
3. **Create** a YAML config in `configs/` pointing to your model (see `configs/s5_example.yaml` for the format)
4. **Train**: `uv run accelerate launch -m train.run_config --config configs/your_config.yaml`
   - Your training script must call `wandb.init(...)` and write the resulting
     `wandb.run.id` into `results.json` under the key `wandb_run_id` (and ideally
     also `wandb_url`). This lets the evaluator attach the verdict to the same
     W&B run for cross-experiment comparison.
5. **Evaluate**: `uv run python -m harness.evaluate --criteria criteria.yaml --results results.json --experiment-id {experiment_id} --proposal-id {proposal_id}`
{report_cmd}

## Time Budget

You have **{time_budget} minutes**. Prioritize getting a working training run
over perfecting the architecture. A simple model that trains and passes criteria
is better than a complex one that doesn't finish.

## Baseline

{baseline_section}

---

Begin by reading the existing model files and task definitions to understand the interface.
Then implement your model, create a config, and run training + evaluation.
"""


def _render_criteria_table(criteria: dict) -> str:
    """Render the criteria as a markdown table."""
    rows = []
    rows.append("| Criterion | Metric Key | Target | Comparator | Required |")
    rows.append("|-----------|-----------|--------|------------|----------|")

    for task in criteria.get("tasks", []):
        task_name = task.get("task_name", "")
        for c in task.get("criteria", []):
            name = c.get("name", "")
            metric_key = c.get("metric_key", "")
            target = c.get("target", "")
            comparator = c.get("comparator", ">=")
            required = c.get("required", True)

            if c.get("comparator") == "between" and c.get("target_upper") is not None:
                target_str = f"{target} - {c['target_upper']}"
            else:
                target_str = f"{comparator} {target}"

            req_str = "Yes *" if required else "No"
            rows.append(
                f"| {name} | `{metric_key}` | {target_str} | {comparator} | {req_str} |"
            )

    return "\n".join(rows)
