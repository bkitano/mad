"""
Research Agent

An agent that reviews documented tricks and papers to propose new experiments.
It analyzes the catalog of algorithmic techniques and suggests concrete experiments
that could validate or combine these techniques for improved model architectures.

The Research Agent:
1. Reads all documented tricks from tricks/
2. Reviews downloaded papers in papers/
3. Cross-references techniques to find synergies
4. Proposes concrete experiments with hypotheses and expected outcomes

Usage:
    # One-shot research cycle
    python -m agents.research_agent

    # Programmatic usage
    from agents.research_agent import run_research_cycle
    async for msg in run_research_cycle():
        print(msg)
"""

import asyncio
import os
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator, Optional

from claude_agent_sdk import query, ClaudeAgentOptions


PROJECT_ROOT = Path(__file__).parent.parent
TRICKS_DIR = PROJECT_ROOT / "tricks"
PAPERS_DIR = PROJECT_ROOT / "papers"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
PROPOSALS_DIR = PROJECT_ROOT / "proposals"


# =============================================================================
# System Prompt
# =============================================================================

RESEARCH_AGENT_SYSTEM_PROMPT = """You are the Research Agent, an expert in designing experiments
that validate and combine algorithmic techniques for neural network architectures.

Your goal is to review the documented tricks and papers, identify promising combinations
or untested applications, and propose concrete experiments.

## Human Feedback

**IMPORTANT**: Always check `human_feedback.md` at the project root for ongoing instructions, preferences, and constraints from the human researcher. This feedback takes priority over general guidelines and should guide your proposal generation.

## Your Knowledge Base

- **Tricks folder**: {tricks_dir} - Contains documented algorithmic techniques
- **Papers folder**: {papers_dir} - Contains source PDFs for reference
- **Proposals folder**: {proposals_dir} - Where you write experiment proposals. Especially focus on the Human Review section, where I manually review the ideas and highlight pros/cons.

## What to Look For

1. **Untested combinations**: Can two tricks be combined for compounding benefits?
2. **Architecture gaps**: Are there tricks that haven't been applied to certain architectures?
3. **Complexity frontiers**: Where can we push efficiency further?
4. **Ablation opportunities**: What assumptions in existing tricks should be tested?
5. **Scaling questions**: How do tricks behave at different scales?

## How to Propose Experiments

For each promising experiment idea, create a markdown file in proposals/ with this format:

```markdown
# [Experiment Name]

**Status**: proposed
**Priority**: [high|medium|low]
**Created**: [YYYY-MM-DD]
**Based on**: [List of tricks/papers this builds on]

## Hypothesis

[Clear statement of what you expect to find]

## Background

[Why this experiment is worth running - what gap it fills]

## Related Work

**IMPORTANT**: Use WebSearch to check if similar work exists before proposing.

[Summarize related papers/approaches and how this proposal differs]

Example:
- **[Paper 1]**: Did [X] but didn't test [Y]
- **[Paper 2]**: Proposed [mechanism A] but for [different architecture]
- **Our approach**: Combines [A] + [B] which hasn't been explored

If no closely related work found after searching, state: "No directly related work found combining these specific techniques."

## Mathematical Formulation

Use LaTeX to precisely describe the mathematical setup.

**Standard Approach:**

$$
y_t = \sum_{{i=1}}^{{t}} A^{{t-i}} B x_i \quad \text{{(naive: }} O(T^2 d^2) \text{{)}}
$$

**Proposed Modification:**

$$
y_t = C h_t, \quad h_t = A h_{{t-1}} + B x_t \quad \text{{(with trick: }} O(T d^2) \text{{)}}
$$

**Key Variables:**
- $x_t \in \mathbb{{R}}^d$ — input at time $t$
- $h_t \in \mathbb{{R}}^n$ — hidden state
- $A \in \mathbb{{R}}^{{n \times n}}$ — state transition matrix

## Method

### Architecture

| Component | Configuration |
|-----------|---------------|
| Model | [e.g., Mamba, Transformer] |
| Layers | $L = [num]$ |
| Hidden dim | $d = [num]$ |
| State dim | $n = [num]$ |

### Baseline
[What to compare against, with complexity: $O(\cdot)$]

### Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Throughput | $> X$ tokens/sec | [how measured] |
| Memory | $< Y$ GB | Peak GPU memory |
| Quality | $\geq Z$ | [perplexity, accuracy, etc.] |

### Estimated Compute
[Rough estimate: small/medium/large, with GPU-hours if possible]

## Expected Outcome

**If hypothesis is correct:**
- [Quantitative prediction, e.g., "$2\times$ speedup at sequence length $T = 4096$"]

**If hypothesis is wrong:**
- [What we learn from negative result]

## Minimum Viable Experiment

**CRITICAL**: Before running the full experiment, define the smallest possible test that would show "signs of life" — evidence that the core idea has merit and is worth scaling up.

### Setup
- **Model**: Tiny model (e.g., 1-2 layers, $d = 32$–$64$, ~100K params)
- **Task**: Simplest task that exercises the key property (e.g., synthetic copying, single-step state tracking)
- **Data**: Minimal synthetic dataset (e.g., 1K–10K samples)
- **Compute**: Single GPU, $< 10$ minutes

### Success Criteria
- [Specific, measurable outcome that would indicate the idea works]
- [e.g., "$> 90\%$ accuracy on 10-step state tracking where baseline achieves $< 60\%$"]

### Failure Criteria
- [What result would kill the idea immediately]
- [e.g., "If tiny model can't beat random baseline, the mechanism is broken"]

### Why This Test Is Sufficient
- [Explain why success here implies the full experiment is worth running]
- [e.g., "State tracking is the core capability; if it works at small scale, scaling adds capacity not capability"]

## Theoretical Analysis

Complexity comparison:

| Operation | Baseline | Proposed |
|-----------|----------|----------|
| Forward pass | $O(T^2 d)$ | $O(T d^2)$ |
| Backward pass | $O(T^2 d)$ | $O(T d^2)$ |
| Memory | $O(T^2)$ | $O(T d)$ |

Crossover point: $T^* = d$ (proposed is better when $T > d$)

## Risks & Limitations

[What could go wrong or invalidate results]

## Follow-up Experiments

[What to try next based on possible outcomes]
```

## LaTeX Guidelines

- Use `$...$` for inline math: "complexity is $O(n^2)$"
- Use `$$...$$` for display equations on their own line
- Common symbols:
  - Matrices: $A, B, W$ (uppercase)
  - Vectors: $x, h, y$ (lowercase)
  - Dimensions: $d, n, T, L$
  - Big-O: $O(n^2)$, $O(n \log n)$, $\Theta(n)$
  - Expectations: $\mathbb{{E}}[X]$
  - Norms: $\|x\|_2$, $\|W\|_F$
  - Sequences: $\{{x_t\}}_{{t=1}}^T$

## Guidelines

- Be specific and actionable - vague ideas aren't useful
- Ground proposals in the documented tricks - reference them explicitly
- Use LaTeX for all mathematical expressions - precision matters
- Include complexity analysis with Big-O notation
- Consider computational cost - prefer small experiments that give signal
- Think about ablations - what's the minimal test of the core idea?
- Avoid duplicating existing proposals - check proposals/ first
- **CRITICAL: Check if the work already exists** before proposing:
  - Use WebSearch to check if the proposed experiment or similar combinations have been published
  - Search for: "[trick combination] [architecture type] paper"
  - Search for: "[key mechanism] state space model" or similar
  - If closely related work exists, either:
    - Skip the proposal if it's essentially the same
    - OR modify the proposal to test something the existing work didn't
    - OR propose an ablation/extension of the existing work
  - Include a "Related Work" section noting what's been done and how this differs
- **ALWAYS include a Minimum Viable Experiment** - every proposal MUST have a tiny, fast test that shows "signs of life" before committing to the full experiment. This is non-negotiable. The MVE should:
  - Run in under 10 minutes on a single GPU
  - Use a toy model (1-2 layers, ~100K params)
  - Test the core mechanism on a synthetic task
  - Have clear success/failure criteria
"""


# =============================================================================
# Helper Functions
# =============================================================================

def list_tricks() -> list[dict]:
    """List all documented tricks with their metadata."""
    tricks = []
    if not TRICKS_DIR.exists():
        return tricks

    for filepath in sorted(TRICKS_DIR.glob("*.md")):
        content = filepath.read_text()
        trick = {
            "name": filepath.stem,
            "path": str(filepath),
            "content": content,
        }

        # Parse metadata
        for line in content.split("\n"):
            if line.startswith("**Category**:"):
                trick["category"] = line.split(":", 1)[1].strip()
            elif line.startswith("**Gain type**:"):
                trick["gain_type"] = line.split(":", 1)[1].strip()
            elif line.startswith("**Paper**:"):
                trick["paper"] = line.split(":", 1)[1].strip()

        tricks.append(trick)

    return tricks


def list_papers() -> list[str]:
    """List all downloaded papers."""
    if not PAPERS_DIR.exists():
        return []
    return sorted([f.name for f in PAPERS_DIR.glob("*.pdf")])


def list_proposals() -> list[dict]:
    """List existing experiment proposals."""
    proposals = []
    if not PROPOSALS_DIR.exists():
        return proposals

    for filepath in sorted(PROPOSALS_DIR.glob("*.md")):
        content = filepath.read_text()
        proposal = {
            "name": filepath.stem,
            "path": str(filepath),
        }

        # Parse status and priority
        for line in content.split("\n"):
            if line.startswith("**Status**:"):
                proposal["status"] = line.split(":", 1)[1].strip()
            elif line.startswith("**Priority**:"):
                proposal["priority"] = line.split(":", 1)[1].strip()

        proposals.append(proposal)

    return proposals


def get_tricks_summary() -> str:
    """Generate a summary of all tricks for the agent."""
    tricks = list_tricks()
    if not tricks:
        return "No tricks documented yet."

    lines = ["## Documented Tricks\n"]
    for t in tricks:
        category = t.get("category", "?")
        gain = t.get("gain_type", "?")
        lines.append(f"- **{t['name']}** [{category}] - {gain}")

    return "\n".join(lines)


def get_proposals_summary() -> str:
    """Generate a summary of existing proposals."""
    proposals = list_proposals()
    if not proposals:
        return "No proposals yet."

    lines = ["## Existing Proposals\n"]
    for p in proposals:
        status = p.get("status", "?")
        priority = p.get("priority", "?")
        lines.append(f"- **{p['name']}** [{status}] - Priority: {priority}")

    return "\n".join(lines)


def ensure_proposals_dir() -> Path:
    """Ensure the proposals directory exists."""
    PROPOSALS_DIR.mkdir(parents=True, exist_ok=True)
    return PROPOSALS_DIR


# =============================================================================
# Research Agent
# =============================================================================

async def run_research_cycle(
    focus_area: Optional[str] = None,
    max_proposals: int = 3,
) -> AsyncIterator:
    """
    Run a research cycle to propose new experiments.

    Args:
        focus_area: Optional area to focus on (e.g., "parallelization", "SSMs")
        max_proposals: Maximum number of new proposals to generate

    Yields:
        Agent messages as they stream in
    """
    ensure_proposals_dir()

    tricks_summary = get_tricks_summary()
    proposals_summary = get_proposals_summary()
    papers = list_papers()
    papers_str = "\n".join(f"- {p}" for p in papers) if papers else "None downloaded"

    focus_instruction = ""
    if focus_area:
        focus_instruction = f"\n\n**Focus Area**: Prioritize experiments related to: {focus_area}"

    prompt = f"""Review the documented tricks and propose up to {max_proposals} new experiments.

{tricks_summary}

## Downloaded Papers
{papers_str}

{proposals_summary}

## Instructions

1. Read all the trick documentation files in tricks/ to understand each technique deeply
2. Check existing proposals in proposals/ to avoid duplicates
3. **CRITICAL**: For each promising idea, use WebSearch to check if it already exists:
   - Search: "[trick 1] + [trick 2] + [architecture] paper"
   - Search: "[key mechanism] state space model"
   - Search: "[core technique] transformer/attention"
   - Review search results to see if the work has been done
   - If similar work exists, either skip or differentiate your proposal
4. Identify promising experiment opportunities:
   - Combinations of tricks that haven't been tested together
   - Applications to architectures not yet explored
   - Ablations that could reveal important insights
5. Write detailed proposals for the most promising ideas
6. Include a "Related Work" section in each proposal noting prior work
7. Prioritize experiments that:
   - Have clear hypotheses
   - Fill a gap not addressed by existing literature
   - Can be validated with reasonable compute
   - Build on multiple documented tricks
{focus_instruction}

Start by reading the trick files, then propose experiments.
"""

    async for message in query(
        prompt=prompt,
        options=ClaudeAgentOptions(
            model="opus",
            system_prompt=RESEARCH_AGENT_SYSTEM_PROMPT.format(
                tricks_dir=TRICKS_DIR,
                papers_dir=PAPERS_DIR,
                proposals_dir=PROPOSALS_DIR,
            ),
            allowed_tools=["Read", "Write", "Glob", "Grep", "WebSearch"],
            permission_mode="acceptEdits",
            cwd=str(PROJECT_ROOT),
        )
    ):
        yield message


# =============================================================================
# CLI Entry Point
# =============================================================================

async def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Research Agent - Propose experiments based on documented tricks"
    )
    parser.add_argument(
        "--focus", "-f",
        help="Focus area for experiments (e.g., 'parallelization', 'SSMs')"
    )
    parser.add_argument(
        "--max-proposals", "-n",
        type=int,
        default=3,
        help="Maximum number of proposals to generate (default: 3)"
    )
    parser.add_argument(
        "--list-tricks",
        action="store_true",
        help="List documented tricks and exit"
    )
    parser.add_argument(
        "--list-proposals",
        action="store_true",
        help="List existing proposals and exit"
    )

    args = parser.parse_args()

    if args.list_tricks:
        print("\n" + get_tricks_summary())
        return

    if args.list_proposals:
        print("\n" + get_proposals_summary())
        return

    print("\n" + "=" * 60)
    print(" Research Agent - Experiment Proposal Cycle")
    print("=" * 60)

    if args.focus:
        print(f"\nFocus area: {args.focus}")

    print(f"\nGenerating up to {args.max_proposals} proposals...\n")

    async for msg in run_research_cycle(
        focus_area=args.focus,
        max_proposals=args.max_proposals
    ):
        if hasattr(msg, 'content'):
            for block in getattr(msg, 'content', []):
                if hasattr(block, 'text'):
                    print(block.text, end="", flush=True)
        elif hasattr(msg, 'result'):
            print(f"\n\nDone! {msg.result}")

    print("\n")


if __name__ == "__main__":
    asyncio.run(main())
