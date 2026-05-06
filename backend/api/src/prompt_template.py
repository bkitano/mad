"""
Build structured agent prompts from proposal content and evaluation criteria.
"""

from __future__ import annotations

from typing import Optional


def _render_criteria_table(criteria: dict) -> str:
    rows = []
    for c in criteria.get("criteria", []):
        req = "Yes" if c.get("required", True) else "No"
        rows.append(
            f"| {c['name']} | `{c['metric_key']}` | {c['comparator']} {c['target']} | {req} | {c.get('description', '')} |"
        )
    if not rows:
        return ""
    header = "| Criterion | Metric Key | Target | Required | Description |\n"
    header += "|-----------|-----------|--------|----------|-------------|\n"
    return header + "\n".join(rows)


def build_agent_prompt(
    proposal_content: str,
    criteria: Optional[dict] = None,
    experiment_id: Optional[str] = None,
    service_url: Optional[str] = None,
) -> str:
    sections = []

    sections.append("# Experiment Proposal\n")
    sections.append(proposal_content)

    if criteria:
        sections.append("\n---\n")
        sections.append("# Success Criteria\n")
        table = _render_criteria_table(criteria)
        if table:
            sections.append(table)

        time_budget = criteria.get("time_budget_minutes", 120)
        sections.append(f"\nTime budget: {time_budget} minutes.\n")

        allowed = criteria.get("allowed_edit_globs", ["models/**", "configs/**"])
        forbidden = criteria.get("forbidden_edit_globs", ["train/**", "tasks/**", "evaluate.py"])
        sections.append("## Edit Constraints\n")
        sections.append(f"You may ONLY modify files matching: {', '.join(f'`{g}`' for g in allowed)}")
        sections.append(f"\nDo NOT edit files matching: {', '.join(f'`{g}`' for g in forbidden)}")

        if criteria.get("baseline_config"):
            sections.append(f"\nBaseline config for reference: `{criteria['baseline_config']}`")

    sections.append("\n---\n")
    sections.append("# Model Interface Contract\n")
    sections.append(
        "Your model class must implement:\n"
        "- `__init__(self, num_tokens, num_classes, eos_idx, max_seq_len, d_model, nhead, num_layers, dropout)`\n"
        "- `forward(self, tokens, mask) -> logits`\n"
    )

    sections.append("# Workflow\n")
    sections.append(
        "1. Read the codebase to understand the task and training pipeline\n"
        "2. Implement your model in `models/`\n"
        "3. Create a config YAML in `configs/`\n"
        "4. Run training: `uv run python main.py --config configs/your_config.yaml`\n"
        "5. Run evaluation: `uv run python -m harness.evaluate`\n"
        "6. Verify `verdict.json` shows `overall_pass: true`\n"
    )

    if experiment_id:
        sections.append(f"\nExperiment ID: `{experiment_id}`")
    if service_url:
        sections.append(f"\nMAD Service URL: `{service_url}`")

    return "\n".join(sections)
