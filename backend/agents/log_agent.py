"""
Log Agent

An agent that reviews recently added tricks and proposals (last 12 hours)
and generates human-readable update logs in notes/log.md.

The Log Agent:
1. Scans tricks/ and proposals/ for recent additions (last 12 hours)
2. Generates concise summaries of new discoveries
3. Appends to notes/log.md with timestamps

Usage:
    # One-shot log generation
    python -m agents.log_agent

    # Programmatic usage
    from agents.log_agent import run_log_cycle
    await run_log_cycle()
"""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional

from claude_agent_sdk import query, ClaudeAgentOptions


PROJECT_ROOT = Path(__file__).parent.parent
TRICKS_DIR = PROJECT_ROOT / "tricks"
PROPOSALS_DIR = PROJECT_ROOT / "proposals"
NOTES_DIR = PROJECT_ROOT / "notes"
RESEARCH_LOGS_DIR = NOTES_DIR / "research_logs"
CODE_DIR = PROJECT_ROOT / "code"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
LOG_FILE = NOTES_DIR / "log.md"  # Legacy, kept for backwards compatibility


# =============================================================================
# System Prompt
# =============================================================================

LOG_AGENT_SYSTEM_PROMPT = """You are the Log Agent, a technical writer and research strategist
that reviews recent research activity and generates concise, readable update logs.

Your goal is to review tricks and proposals added in the last 12 hours and create
a digestible summary for a human researcher to stay updated, with a critical focus
on identifying the most impactful proposals.

**CRITICAL**: Output ONLY the final formatted log entry. Do NOT include any intermediate
thinking, planning, or meta-commentary like "I need to analyze..." or "Let me read...".
Your entire output will be saved directly to the research log, so it must be polished
and ready for the researcher to read.

## Your Task

You will be provided with:
1. **New tricks** - Recently documented algorithmic techniques
2. **New proposals** - Recently created experiment proposals
3. **Experiment updates** - Recently implemented or completed experiments

Create a brief, informative log entry that:
- Summarizes what was discovered/proposed
- Highlights key insights or connections
- **Prioritizes proposals by impact** - which are most likely to yield breakthrough results?
- Uses clear, concise language
- Focuses on the "why this matters" rather than just listing items

## Impact Assessment Criteria

When prioritizing proposals, consider:
1. **Novelty**: Does it combine tricks in an untested way?
2. **Theoretical foundation**: Is the hypothesis well-grounded?
3. **Practical feasibility**: Can it be implemented/tested easily **on small hardware for <$10**?
4. **Resource constraints**: Can it be trained on modest GPUs (e.g., single consumer GPU, cloud spot instances)?
5. **Potential upside**: Could it unlock new capabilities or efficiency gains?
6. **Risk/reward**: What's the cost if it fails vs. benefit if it succeeds?

**CRITICAL**: Prioritize proposals that can be validated cheaply (<$10 compute budget) on accessible hardware.
Proposals requiring expensive infrastructure should be ranked lower, even if theoretically interesting.

## Output Format

Generate a markdown section like this:

```markdown
## [Date] - [Time]

### 🎯 High-Impact Proposals
[Rank the top 1-2 proposals with the highest potential impact]

- **[Proposal Name]** (Priority: [high/medium/low])
  - **Hypothesis**: [1 sentence]
  - **Why it matters**: [2-3 sentences on potential impact and novelty]
  - **Estimated cost**: [<$5 / <$10 / >$10]
  - **Impact score**: [X/10] - [brief justification focusing on cost-effectiveness]

### 🧪 Experiment Updates
[If experiments were implemented or completed, report on them]

- **Experiment XXX: [Name]** (Status: [implemented/running/completed])
  - **Proposal**: [Which proposal this implements]
  - **Progress**: [What was done - implementation, results, metrics]
  - **Key findings**: [1-2 sentences on results if completed]
  - **Cost**: [$X.XX actual vs $Y.YY estimated]

### 📚 New Discoveries
- **[Trick Name]**: [1-2 sentence summary emphasizing the key insight and potential impact]

### Other Proposals
[Other proposals not in the high-impact section]
- **[Proposal Name]**: [1 sentence summary]

### Strategic Insights
[2-3 sentences connecting the dots between recent additions, identifying themes,
opportunities, or suggesting which direction to focus research efforts]

---
```

Keep it concise but insightful - the entire entry should be readable in under 3 minutes.
Focus on actionable strategic guidance for the researcher.
"""


# =============================================================================
# Helper Functions
# =============================================================================

def get_recent_items(directory: Path, hours: int = 12) -> List[Dict[str, str]]:
    """
    Get items (tricks or proposals) added/modified in the last N hours.

    Args:
        directory: Path to tricks/ or proposals/
        hours: Time window to check (default: 12)

    Returns:
        List of dicts with file info and parsed metadata
    """
    cutoff_time = datetime.now() - timedelta(hours=hours)
    recent_items = []

    if not directory.exists():
        return recent_items

    for md_file in directory.glob("*.md"):
        try:
            # Check file modification time
            mtime = datetime.fromtimestamp(md_file.stat().st_mtime)

            if mtime < cutoff_time:
                continue

            # Parse the file
            content = md_file.read_text()
            lines = content.split('\n')

            # Extract title (first line)
            title = lines[0].strip('# ') if lines else md_file.stem

            # Extract relevant metadata
            metadata = {
                'file': md_file.name,
                'title': title,
                'modified': mtime.isoformat(),
                'content': content
            }

            # Parse specific fields
            for line in lines[:30]:  # Check first 30 lines for metadata
                if line.startswith("**Documented**:") or line.startswith("**Created**:"):
                    metadata['date'] = line.split(":", 1)[1].strip()
                elif line.startswith("**Category**:"):
                    metadata['category'] = line.split(":", 1)[1].strip()
                elif line.startswith("**Gain type**:"):
                    metadata['gain_type'] = line.split(":", 1)[1].strip()
                elif line.startswith("**Status**:"):
                    metadata['status'] = line.split(":", 1)[1].strip()
                elif line.startswith("**Priority**:"):
                    metadata['priority'] = line.split(":", 1)[1].strip()
                elif line.startswith("**Based on**:"):
                    metadata['based_on'] = line.split(":", 1)[1].strip()

            recent_items.append(metadata)

        except Exception as e:
            print(f"Error processing {md_file}: {e}")
            continue

    return recent_items


def get_recent_experiments(hours: int = 12) -> List[Dict[str, str]]:
    """
    Get experiments created/updated in the last N hours.

    Args:
        hours: Time window to check (default: 12)

    Returns:
        List of dicts with experiment info
    """
    cutoff_time = datetime.now() - timedelta(hours=hours)
    recent_experiments = []

    # Check code/ directory for new experiment implementations
    if CODE_DIR.exists():
        for exp_dir in CODE_DIR.iterdir():
            if not exp_dir.is_dir():
                continue

            # Check if experiment was created/modified recently
            try:
                mtime = datetime.fromtimestamp(exp_dir.stat().st_mtime)
                if mtime < cutoff_time:
                    continue

                exp_num = exp_dir.name

                # Try to read README for context
                readme_path = exp_dir / "README.md"
                readme_content = ""
                proposal_name = "Unknown"

                if readme_path.exists():
                    readme_content = readme_path.read_text()
                    # Try to extract proposal name from README
                    for line in readme_content.split('\n')[:20]:
                        if 'proposal' in line.lower() or 'experiment' in line.lower():
                            proposal_name = line.strip('# ').strip()
                            break

                # Check for results file
                results_file = EXPERIMENTS_DIR / f"{exp_num}_results.md"
                status = "implemented"
                results_content = ""

                if results_file.exists():
                    status = "completed"
                    results_content = results_file.read_text()[:1000]  # First 1000 chars

                recent_experiments.append({
                    'experiment_num': exp_num,
                    'name': proposal_name,
                    'status': status,
                    'modified': mtime.isoformat(),
                    'readme': readme_content[:800],
                    'results': results_content,
                    'path': str(exp_dir)
                })

            except Exception as e:
                print(f"Error processing experiment {exp_dir}: {e}")
                continue

    return recent_experiments


# =============================================================================
# Main Agent Function
# =============================================================================

async def run_log_cycle(hours_back: int = 12) -> Optional[str]:
    """
    Run one cycle of the log agent:
    1. Find recent tricks and proposals
    2. Generate log entry
    3. Append to log.md

    Args:
        hours_back: How many hours back to look for new items (default: 12)

    Returns:
        The generated log entry, or None if no new items
    """
    print(f"[LOG AGENT] Scanning for items from last {hours_back} hours...")

    # Get recent items
    recent_tricks = get_recent_items(TRICKS_DIR, hours=hours_back)
    recent_proposals = get_recent_items(PROPOSALS_DIR, hours=hours_back)
    recent_experiments = get_recent_experiments(hours=hours_back)

    if not recent_tricks and not recent_proposals and not recent_experiments:
        print("[LOG AGENT] No new items found in the specified time window.")
        return None

    print(f"[LOG AGENT] Found {len(recent_tricks)} new tricks, {len(recent_proposals)} new proposals, {len(recent_experiments)} experiments")

    # Build context for the agent
    context = f"""# Recent Activity Summary

Time window: Last {hours_back} hours
Timestamp: {datetime.now().isoformat()}

## New Tricks ({len(recent_tricks)})

"""

    for trick in recent_tricks:
        context += f"### {trick['title']}\n"
        context += f"- File: {trick['file']}\n"
        if 'category' in trick:
            context += f"- Category: {trick['category']}\n"
        if 'gain_type' in trick:
            context += f"- Gain type: {trick['gain_type']}\n"
        context += f"\nContent preview:\n```\n{trick['content'][:800]}\n```\n\n"

    context += f"## New Proposals ({len(recent_proposals)})\n\n"

    for proposal in recent_proposals:
        context += f"### {proposal['title']}\n"
        context += f"- File: {proposal['file']}\n"
        if 'status' in proposal:
            context += f"- Status: {proposal['status']}\n"
        if 'priority' in proposal:
            context += f"- Priority: {proposal['priority']}\n"
        if 'based_on' in proposal:
            context += f"- Based on: {proposal['based_on']}\n"
        context += f"\nContent preview:\n```\n{proposal['content'][:800]}\n```\n\n"

    context += f"## Recent Experiments ({len(recent_experiments)})\n\n"

    for exp in recent_experiments:
        context += f"### Experiment {exp['experiment_num']}: {exp['name']}\n"
        context += f"- Status: {exp['status']}\n"
        context += f"- Path: {exp['path']}\n"
        context += f"- Modified: {exp['modified']}\n"

        if exp['readme']:
            context += f"\nREADME preview:\n```\n{exp['readme']}\n```\n"

        if exp['results']:
            context += f"\nResults:\n```\n{exp['results']}\n```\n"

        context += "\n"

    # Generate log entry using Claude
    user_prompt = f"""Review the following recent activity and generate a log entry with impact prioritization.

{context}

Remember:
1. **Critically assess which proposals have the highest impact potential** using the criteria:
   - Novelty, theoretical foundation, COST-EFFECTIVE feasibility (<$10 budget), potential upside, risk/reward
   - **Heavily favor proposals that can be validated on small hardware for under $10**
2. **Review experiment progress**: If experiments were implemented/completed, highlight what was learned
3. Focus on insights and connections, not just listing
4. What makes these discoveries interesting?
5. What themes or opportunities emerge?
6. Which proposals should the researcher focus on first given budget constraints?
7. Estimate training costs for each proposal (consider: model size, dataset, training time, GPU type)
8. Keep it concise, readable, and actionable.

**IMPORTANT**: Output ONLY the formatted log entry starting with "## [Date] - [Time]".
Do NOT include any preamble, thinking, or meta-commentary. Your output will be saved
directly to the research log file.
"""

    print("[LOG AGENT] Generating log entry...")

    all_text = ""
    async for message in query(
        prompt=user_prompt,
        options=ClaudeAgentOptions(
            model="opus",
            system_prompt=LOG_AGENT_SYSTEM_PROMPT,
            allowed_tools=[],  # Log agent only needs to read and write text
            permission_mode="acceptEdits",
            cwd=str(PROJECT_ROOT),
        )
    ):
        # Extract text from message content blocks
        if hasattr(message, 'content'):
            for block in getattr(message, 'content', []):
                if hasattr(block, 'text'):
                    all_text += block.text

    if not all_text.strip():
        print("[LOG AGENT] Failed to generate log entry")
        return None

    # Extract only the formatted log entry (starting with ##)
    # This filters out any intermediate thinking or preamble
    lines = all_text.split('\n')
    log_entry_lines = []
    capturing = False

    for line in lines:
        # Start capturing when we see a line starting with "##"
        if line.strip().startswith('##'):
            capturing = True

        if capturing:
            log_entry_lines.append(line)

    # If we captured formatted content, use it; otherwise use all text
    log_entry = '\n'.join(log_entry_lines) if log_entry_lines else all_text

    if not log_entry.strip():
        print("[LOG AGENT] Failed to extract formatted log entry")
        return None

    # Ensure research logs directory exists
    RESEARCH_LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # Create timestamped file
    timestamp = datetime.now()
    timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
    filename = timestamp.strftime("%Y-%m-%dT%H-%M-%S") + ".md"
    log_file_path = RESEARCH_LOGS_DIR / filename

    # Write atomic log file with frontmatter
    log_content = f"""---
title: Research Log Update
date: {timestamp.isoformat()}
timestamp: {timestamp_str}
tricks: {len(recent_tricks)}
proposals: {len(recent_proposals)}
experiments: {len(recent_experiments)}
---

{log_entry.strip()}
"""

    log_file_path.write_text(log_content)

    # Also append to legacy log.md for backwards compatibility
    if LOG_FILE.exists():
        existing_log = LOG_FILE.read_text()
    else:
        existing_log = "# MAD Architecture Search - Activity Log\n\nAutomated updates from the research loop.\n\n---\n\n"

    updated_log = existing_log + "\n" + log_entry.strip() + "\n\n---\n\n"
    LOG_FILE.write_text(updated_log)

    print(f"[LOG AGENT] ✓ Log entry written to {log_file_path}")
    print(f"[LOG AGENT] Summary: {len(recent_tricks)} new tricks, {len(recent_proposals)} new proposals, {len(recent_experiments)} experiments")

    return log_entry


# =============================================================================
# CLI Entry Point
# =============================================================================

async def main():
    """CLI entry point for running the log agent once."""
    print("=" * 60)
    print("MAD Architecture Search - Log Agent")
    print("=" * 60)

    log_entry = await run_log_cycle(hours_back=12)

    if log_entry:
        print("\n" + "=" * 60)
        print("Generated Log Entry:")
        print("=" * 60)
        print(log_entry)
    else:
        print("\nNo updates to log.")


if __name__ == "__main__":
    asyncio.run(main())
