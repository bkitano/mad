"""
MAD Architecture Search Runner

An autonomous loop that runs four concurrent agents:
1. Trick Search Agent (every 10 minutes): Searches for and documents new algorithmic tricks
2. Research Agent (every 15 minutes): Reviews tricks and generates experiment proposals
3. Experiment Agents (10 parallel, every 15 minutes): Implements and runs MVEs from high-priority proposals
4. Log Agent (every 1 hour): Reviews recent activity and updates notes/log.md

Usage:
    python runner.py [--trick-interval MINUTES] [--research-interval MINUTES] [--experiment-interval MINUTES] [--log-interval MINUTES] [--num-experiment-agents NUM]
"""

import argparse
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict

from agents.trick_search import run_trick_search, TRICKS_DIR, PAPERS_DIR, PROJECT_ROOT
from agents.research_agent import run_research_cycle
from agents.experiment_agent import (
    run_experiment_cycle,
    list_proposals,
    list_implemented_experiments,
    select_best_proposal,
)
from agents.log_agent import run_log_cycle
from agents.experiment_scaler import run_scaler_cycle
from agents.work_tracker import get_claimed_proposals, print_active_work
from agents.agent_state import (
    save_agent_state,
    load_agent_state,
    clear_agent_state,
    should_retry,
    get_retry_delay,
    get_all_agent_states,
    is_agent_active,
)
from agents.agent_log import (
    write_agent_log,
    read_agent_log,
    read_all_agent_logs,
)
from agents.human_feedback import (
    read_human_feedback,
    has_recent_feedback,
    get_feedback_summary,
)
from agents.agent_status_tracker import update_agent_status

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('runner.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def log_agent_message(agent_name: str, message, message_num: int):
    """
    Log all messages from Claude SDK to runner.log for debugging.

    Args:
        agent_name: Name of the agent (e.g., "trick_search", "experiment-agent-exp-01")
        message: Message object from Claude SDK
        message_num: Message number in the sequence
    """
    try:
        # Log message type
        msg_type = type(message).__name__

        # Extract text content if available
        if hasattr(message, 'content'):
            for block in getattr(message, 'content', []):
                if hasattr(block, 'text'):
                    text = block.text
                    # Log first 500 chars to avoid flooding
                    preview = text[:500] + "..." if len(text) > 500 else text
                    logger.info(f"[{agent_name}][msg-{message_num}] {msg_type}: {preview}")
                elif hasattr(block, 'type'):
                    # Tool use or other non-text blocks
                    logger.info(f"[{agent_name}][msg-{message_num}] {msg_type}: {block.type}")

        # Log tool use
        if hasattr(message, 'tool_use'):
            tool_use = getattr(message, 'tool_use', None)
            if tool_use:
                tool_name = getattr(tool_use, 'name', 'unknown')
                logger.info(f"[{agent_name}][msg-{message_num}] Tool: {tool_name}")

        # Log stop reason if available
        if hasattr(message, 'stop_reason'):
            stop_reason = getattr(message, 'stop_reason', None)
            if stop_reason:
                logger.info(f"[{agent_name}][msg-{message_num}] Stop reason: {stop_reason}")

    except Exception as e:
        logger.error(f"[{agent_name}] Error logging message: {e}")

PROPOSALS_DIR = PROJECT_ROOT / "proposals"


def read_proposals() -> List[Dict[str, str]]:
    """
    Read all proposal files and extract key information.

    Returns:
        List of dicts with proposal metadata and content
    """
    proposals = []

    if not PROPOSALS_DIR.exists():
        logger.warning(f"Proposals directory not found: {PROPOSALS_DIR}")
        return proposals

    for proposal_file in sorted(PROPOSALS_DIR.glob("*.md")):
        try:
            content = proposal_file.read_text()

            # Extract metadata from frontmatter
            lines = content.split('\n')
            title = lines[0].strip('# ') if lines else proposal_file.stem

            # Look for "Based on" field to understand dependencies
            based_on = []
            hypothesis = ""
            for i, line in enumerate(lines):
                if line.startswith("**Based on**:"):
                    based_on = [t.strip() for t in line.split(":", 1)[1].split(",")]
                elif line.startswith("## Hypothesis"):
                    # Get the next non-empty line
                    for j in range(i+1, len(lines)):
                        if lines[j].strip():
                            hypothesis = lines[j].strip()
                            break

            proposals.append({
                "file": proposal_file.name,
                "title": title,
                "based_on": based_on,
                "hypothesis": hypothesis[:200],  # First 200 chars
                "content": content
            })

        except Exception as e:
            logger.error(f"Error reading proposal {proposal_file}: {e}")

    logger.info(f"Loaded {len(proposals)} proposals")
    return proposals


def read_tricks() -> List[Dict[str, str]]:
    """
    Read all documented tricks and extract key information.

    Returns:
        List of dicts with trick metadata
    """
    tricks = []

    if not TRICKS_DIR.exists():
        logger.warning(f"Tricks directory not found: {TRICKS_DIR}")
        return tricks

    for trick_file in sorted(TRICKS_DIR.glob("*.md")):
        try:
            content = trick_file.read_text()
            lines = content.split('\n')

            title = lines[0].strip('# ') if lines else trick_file.stem
            category = ""
            gain_type = ""

            for line in lines[:10]:  # Check first 10 lines for metadata
                if line.startswith("**Category**:"):
                    category = line.split(":", 1)[1].strip()
                elif line.startswith("**Gain type**:"):
                    gain_type = line.split(":", 1)[1].strip()

            tricks.append({
                "file": trick_file.name,
                "title": title,
                "category": category,
                "gain_type": gain_type
            })

        except Exception as e:
            logger.error(f"Error reading trick {trick_file}: {e}")

    logger.info(f"Loaded {len(tricks)} tricks")
    return tricks


def generate_search_queries(proposals: List[Dict], tricks: List[Dict]) -> List[str]:
    """
    Analyze proposals and tricks to generate targeted search queries for new tricks.

    Args:
        proposals: List of proposal metadata
        tricks: List of existing tricks

    Returns:
        List of search queries for the trick agent
    """
    queries = []

    # Extract concepts mentioned in proposals that might need tricks
    mentioned_concepts = set()
    for proposal in proposals:
        # Get concepts from "based on" field
        mentioned_concepts.update(proposal.get("based_on", []))

    # Get already documented trick names
    documented = {trick["title"].lower() for trick in tricks}
    documented_categories = {trick["category"] for trick in tricks}

    logger.info(f"Mentioned concepts: {mentioned_concepts}")
    logger.info(f"Documented tricks: {documented}")

    # Base queries to ensure comprehensive coverage
    base_queries = [
        "efficient matrix decompositions for state space models and linear attention",
        "parallelization tricks for sequence processing and recurrent architectures",
        "numerical stability techniques for SSMs and transformers",
        "kernel optimization methods from FlashAttention and related work",
    ]

    # Add queries for concepts mentioned but not documented
    for concept in mentioned_concepts:
        concept_lower = concept.lower()
        if not any(concept_lower in doc for doc in documented):
            queries.append(f"computational tricks and algorithms for {concept}")

    # Category-specific queries if we're light in certain areas
    if "approximation" not in documented_categories:
        queries.append("approximation methods for efficient neural network computation")

    if "algebraic" not in documented_categories:
        queries.append("algebraic structures enabling efficient computation in neural networks")

    # Add base queries
    queries.extend(base_queries)

    # Deduplicate and limit
    queries = list(dict.fromkeys(queries))[:3]  # Max 3 queries per run

    return queries


async def run_trick_search_iteration():
    """
    Run one iteration of trick search:
    1. Check other agents' states
    2. Load proposals and tricks
    3. Generate search queries
    4. Run trick search agent
    5. Log findings
    """
    logger.info("=" * 60)
    logger.info(f"[TRICK SEARCH] Starting iteration at {datetime.now().isoformat()}")
    logger.info("=" * 60)

    # Check other agents' states for coordination
    agent_states = get_all_agent_states(max_age_hours=24)

    # Log what other agents are doing
    active_agents = [name for name, state in agent_states.items()
                     if state and is_agent_active(name, max_age_minutes=30)]
    if active_agents:
        logger.info(f"Active agents: {', '.join(active_agents)}")

    # Check recent research agent activity
    research_log = read_agent_log("research", last_n_entries=2)
    if research_log:
        logger.info("Found recent research agent activity - considering their focus areas")

    # Read human feedback
    if has_recent_feedback(hours=168):  # Last week
        logger.info("Reading human feedback for guidance")
        human_feedback = read_human_feedback(last_n_entries=5)
    else:
        human_feedback = None

    # Check for existing state (resume if agent failed previously)
    existing_state = load_agent_state("trick_search", max_age_hours=2)

    # Load current state
    proposals = read_proposals()
    tricks = read_tricks()

    logger.info(f"Current state: {len(proposals)} proposals, {len(tricks)} tricks")

    # Generate search queries
    if existing_state and "queries" in existing_state.get("context", {}):
        # Resume with remaining queries
        queries = existing_state["context"]["queries"]
        completed_queries = existing_state["progress"].get("completed_queries", [])
        queries = [q for q in queries if q not in completed_queries]
        logger.info(f"Resuming with {len(queries)} remaining queries (attempt {existing_state['attempt_count']})")
    else:
        queries = generate_search_queries(proposals, tricks)
        completed_queries = []

    if not queries:
        logger.info("No new queries generated this iteration")
        clear_agent_state("trick_search")
        return

    # Save initial state
    save_agent_state("trick_search", {
        "task_description": f"Searching for tricks with {len(queries)} queries",
        "progress": {"completed_queries": completed_queries, "total_queries": len(queries) + len(completed_queries)},
        "context": {"queries": queries + completed_queries},
        "started_at": existing_state["started_at"] if existing_state else datetime.now().isoformat(),
        "attempt_count": existing_state["attempt_count"] if existing_state else 0,
    })

    logger.info(f"Generated {len(queries)} search queries:")
    for i, query in enumerate(queries, 1):
        logger.info(f"  {i}. {query}")

    # Run trick search for each query
    for query in queries:
        logger.info(f"\nExecuting search: {query}")
        logger.info("-" * 60)

        try:
            message_count = 0
            async for msg in run_trick_search(query, max_tricks=2):
                message_count += 1
                # Log all messages from Claude SDK
                log_agent_message("trick_search", msg, message_count)

            logger.info(f"  ✓ Completed search: {query}")

            # Update progress
            completed_queries.append(query)
            save_agent_state("trick_search", {
                "task_description": f"Searching for tricks with {len(queries)} queries",
                "progress": {"completed_queries": completed_queries, "total_queries": len(queries)},
                "context": {"queries": queries + completed_queries},
                "started_at": existing_state["started_at"] if existing_state else datetime.now().isoformat(),
                "attempt_count": existing_state["attempt_count"] if existing_state else 0,
            })

        except Exception as e:
            logger.error(f"  ✗ Error during search '{query}': {e}", exc_info=True)
            # Don't clear state on error - allow retry
            raise

        # Brief pause between queries
        await asyncio.sleep(5)

    # Log summary of work done
    tricks_after = read_tricks()
    new_tricks_count = len(tricks_after) - len(tricks)

    if new_tricks_count > 0:
        write_agent_log(
            "trick_search",
            f"Completed search iteration:\n"
            f"- Queries: {len(queries)}\n"
            f"- New tricks documented: {new_tricks_count}\n"
            f"- Total tricks now: {len(tricks_after)}\n\n"
            f"Search queries executed:\n" + "\n".join(f"- {q}" for q in queries),
            entry_type="summary"
        )

    # Clear state on successful completion
    clear_agent_state("trick_search")
    logger.info(f"\n[TRICK SEARCH] Iteration complete at {datetime.now().isoformat()}")
    logger.info("=" * 60 + "\n")


async def run_research_iteration():
    """
    Run one iteration of research agent:
    1. Check other agents' activities
    2. Review all documented tricks
    3. Read existing proposals
    4. Generate new experiment proposals
    5. Log decisions
    """
    logger.info("=" * 60)
    logger.info(f"[RESEARCH AGENT] Starting iteration at {datetime.now().isoformat()}")
    logger.info("=" * 60)

    # Check other agents' states
    agent_states = get_all_agent_states(max_age_hours=24)

    # Read recent trick search activity to understand new tricks
    trick_log = read_agent_log("trick_search", last_n_entries=3)
    if trick_log:
        logger.info("Reviewing recent trick discoveries from trick agent")

    # Check if experiment agent is active
    if is_agent_active("experiment", max_age_minutes=60):
        logger.info("Experiment agent is active - proposals may be under implementation")

    # Read human feedback
    if has_recent_feedback(hours=168):  # Last week
        logger.info("Reading human feedback for guidance")
        human_feedback = read_human_feedback(last_n_entries=5)
    else:
        human_feedback = None

    # Check for existing state
    existing_state = load_agent_state("research", max_age_hours=3)

    # Load current state
    tricks = read_tricks()
    proposals = read_proposals()

    logger.info(f"Current state: {len(tricks)} tricks, {len(proposals)} proposals")

    if len(tricks) < 5:
        logger.info("Not enough tricks documented yet (need at least 5). Skipping research iteration.")
        clear_agent_state("research")
        return

    # Save initial state
    save_agent_state("research", {
        "task_description": f"Generating proposals from {len(tricks)} tricks",
        "progress": {"tricks_count": len(tricks), "proposals_before": len(proposals)},
        "context": {"max_proposals": 2},
        "started_at": existing_state["started_at"] if existing_state else datetime.now().isoformat(),
        "attempt_count": existing_state["attempt_count"] if existing_state else 0,
    })

    logger.info("Generating new experiment proposals based on documented tricks...")
    logger.info("-" * 60)

    try:
        message_count = 0
        async for msg in run_research_cycle(focus_area=None, max_proposals=2):
            message_count += 1
            # Log all messages from Claude SDK
            log_agent_message("research", msg, message_count)

        logger.info(f"  ✓ Completed research cycle")

        # Log summary
        proposals_after = read_proposals()
        new_proposals_count = len(proposals_after) - len(proposals)

        if new_proposals_count > 0:
            write_agent_log(
                "research",
                f"Completed research cycle:\n"
                f"- Analyzed {len(tricks)} tricks\n"
                f"- Generated {new_proposals_count} new proposals\n"
                f"- Total proposals now: {len(proposals_after)}\n\n"
                f"Focus areas: Combining recent tricks into testable hypotheses",
                entry_type="summary"
            )

        # Clear state on success
        clear_agent_state("research")

    except Exception as e:
        logger.error(f"  ✗ Error during research cycle: {e}", exc_info=True)
        raise  # Allow retry logic to handle

    logger.info(f"\n[RESEARCH AGENT] Iteration complete at {datetime.now().isoformat()}")
    logger.info("=" * 60 + "\n")


async def run_single_experiment_agent(agent_id: str, agent_num: int):
    """
    Run a single experiment agent.

    Args:
        agent_id: Agent identifier for work tracking
        agent_num: Agent number (1-5) for logging
    """
    logger.info(f"[{agent_id}] Starting...")

    try:
        message_count = 0
        result = {}
        async for msg in run_experiment_cycle(specific_proposal=None, agent_id=agent_id):
            message_count += 1
            # Log all messages from Claude SDK
            log_agent_message(agent_id, msg, message_count)

            # Capture result
            if isinstance(msg, dict):
                result = msg

        if result.get("error"):
            logger.info(f"[{agent_id}] ⚠ {result['error']}")
        elif "experiment_num" in result:
            exp_num = result['experiment_num']
            logger.info(f"[{agent_id}] ✓ Completed experiment {exp_num:03d}")

            write_agent_log(
                "experiment",
                f"[{agent_id}] Implemented experiment {exp_num:03d}:\n"
                f"- Proposal: {result.get('proposal_id', 'unknown')}\n"
                f"- Cost estimate: ${result.get('estimated_cost', 0):.2f}\n",
                entry_type="implementation"
            )
        else:
            logger.info(f"[{agent_id}] Completed (no experiment selected)")

        return result

    except Exception as e:
        logger.error(f"[{agent_id}] Error: {e}", exc_info=True)
        return {"error": str(e), "agent_id": agent_id}


async def run_experiment_iteration(num_parallel_agents: int = 10):
    """
    Run multiple experiment agents in parallel.

    Args:
        num_parallel_agents: Number of agents to run concurrently (default: 10)

    Each agent:
    1. Checks work tracker for available proposals
    2. Claims a high-priority proposal
    3. Implements the MVE
    4. Runs it if cost < $10
    5. Reports results
    """
    logger.info("=" * 60)
    logger.info(f"[EXPERIMENT AGENTS] Starting iteration with {num_parallel_agents} parallel agents at {datetime.now().isoformat()}")
    logger.info("=" * 60)

    # Check other agents' states
    agent_states = get_all_agent_states(max_age_hours=24)

    # Read recent research activity to see new proposals
    research_log = read_agent_log("research", last_n_entries=2)
    if research_log:
        logger.info("Reviewing recent proposals from research agent")

    # Read human feedback
    if has_recent_feedback(hours=168):  # Last week
        logger.info("Reading human feedback for guidance")

    # Load current state
    proposals = list_proposals()
    implemented = list_implemented_experiments()
    claimed = get_claimed_proposals()

    logger.info(f"Current state: {len(proposals)} proposals, {len(implemented)} implemented, {len(claimed)} in progress")

    if len(proposals) == 0:
        logger.info("No proposals available yet. Skipping experiment iteration.")
        return

    # Show active work
    if claimed:
        logger.info(f"Currently working on: {', '.join(claimed)}")

    # Count available proposals (not implemented, not claimed)
    import re
    available_count = 0
    for p in proposals:
        id_match = re.match(r'(\d+)-', p['id'])
        if id_match:
            exp_num = int(id_match.group(1))
            if exp_num not in implemented and p['id'] not in claimed:
                if p.get('has_mve', False) and p.get('status', '').lower() == 'proposed':
                    available_count += 1

    if available_count == 0:
        logger.info("No available proposals to work on (all are implemented or in progress)")
        return

    # Limit agents to available proposals
    num_agents = min(num_parallel_agents, available_count)
    logger.info(f"Launching {num_agents} agents ({available_count} proposals available)...")
    logger.info("-" * 60)

    try:
        # Launch agents in parallel
        tasks = []
        for i in range(1, num_agents + 1):
            agent_id = f"agent-exp-{i:02d}"
            task = asyncio.create_task(run_single_experiment_agent(agent_id, i))
            tasks.append(task)

        # Wait for all agents to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Summarize results
        success_count = 0
        error_count = 0

        for i, result in enumerate(results, 1):
            if isinstance(result, Exception):
                logger.error(f"  Agent {i}: ❌ Exception - {str(result)[:60]}")
                error_count += 1
            elif isinstance(result, dict):
                if result.get("error"):
                    logger.info(f"  Agent {i}: ⚠ {result.get('error')}")
                    error_count += 1
                elif "experiment_num" in result:
                    success_count += 1
                else:
                    logger.info(f"  Agent {i}: No work done (no available proposals)")

        logger.info(f"\n✓ Experiment iteration complete: {success_count} successful, {error_count} errors")

    except Exception as e:
        logger.error(f"Error in experiment iteration: {e}", exc_info=True)
        raise

    logger.info(f"\n[EXPERIMENT AGENTS] Iteration complete at {datetime.now().isoformat()}")
    logger.info("=" * 60 + "\n")


async def trick_search_loop(interval_minutes: int = 10):
    """
    Run trick search loop indefinitely with retry logic.

    Args:
        interval_minutes: Time to wait between iterations (default: 10)
    """
    interval_seconds = interval_minutes * 60

    logger.info(f"[TRICK SEARCH] Loop started (interval: {interval_minutes} minutes)")

    iteration = 0
    while True:
        iteration += 1
        logger.info(f"\n[TRICK SEARCH] Iteration {iteration}")

        update_agent_status("trick_search", "running", {
            "iteration": iteration,
            "interval_minutes": interval_minutes
        })

        try:
            await run_trick_search_iteration()
        except Exception as e:
            logger.error(f"[TRICK SEARCH] Error in iteration {iteration}: {e}", exc_info=True)

            # Check if we should retry
            if should_retry("trick_search", max_attempts=3):
                state = load_agent_state("trick_search", max_age_hours=2)
                retry_delay = get_retry_delay(state["attempt_count"]) if state else 300
                logger.info(f"[TRICK SEARCH] Will retry in {retry_delay}s (attempt {state['attempt_count'] if state else 1})")
                await asyncio.sleep(retry_delay)
                continue  # Retry immediately instead of waiting full interval
            else:
                logger.error(f"[TRICK SEARCH] Max retries reached, skipping to next iteration")

        logger.info(f"[TRICK SEARCH] Waiting {interval_minutes} minutes until next iteration...")

        next_run = datetime.now().timestamp() + interval_seconds
        update_agent_status("trick_search", "waiting", {
            "iteration": iteration,
            "interval_minutes": interval_minutes,
            "next_run": next_run
        })

        await asyncio.sleep(interval_seconds)


async def research_agent_loop(interval_minutes: int = 15):
    """
    Run research agent loop indefinitely with retry logic.

    Args:
        interval_minutes: Time to wait between iterations (default: 15)
    """
    interval_seconds = interval_minutes * 60

    logger.info(f"[RESEARCH AGENT] Loop started (interval: {interval_minutes} minutes)")

    iteration = 0
    while True:
        iteration += 1
        logger.info(f"\n[RESEARCH AGENT] Iteration {iteration}")

        update_agent_status("research", "running", {
            "iteration": iteration,
            "interval_minutes": interval_minutes
        })

        try:
            await run_research_iteration()
        except Exception as e:
            logger.error(f"[RESEARCH AGENT] Error in iteration {iteration}: {e}", exc_info=True)

            # Check if we should retry
            if should_retry("research", max_attempts=3):
                state = load_agent_state("research", max_age_hours=3)
                retry_delay = get_retry_delay(state["attempt_count"]) if state else 300
                logger.info(f"[RESEARCH AGENT] Will retry in {retry_delay}s (attempt {state['attempt_count'] if state else 1})")
                await asyncio.sleep(retry_delay)
                continue
            else:
                logger.error(f"[RESEARCH AGENT] Max retries reached, skipping to next iteration")

        logger.info(f"[RESEARCH AGENT] Waiting {interval_minutes} minutes until next iteration...")

        next_run = datetime.now().timestamp() + interval_seconds
        update_agent_status("research", "waiting", {
            "iteration": iteration,
            "interval_minutes": interval_minutes,
            "next_run": next_run
        })

        await asyncio.sleep(interval_seconds)


async def experiment_agent_loop(interval_minutes: int = 60, num_parallel_agents: int = 10):
    """
    Run experiment agent loop indefinitely with retry logic.

    Args:
        interval_minutes: Time to wait between iterations (default: 60 = 1 hour)
        num_parallel_agents: Number of agents to run in parallel (default: 10)
    """
    interval_seconds = interval_minutes * 60

    logger.info(f"[EXPERIMENT AGENTS] Loop started (interval: {interval_minutes} minutes, {num_parallel_agents} parallel agents)")

    iteration = 0
    while True:
        iteration += 1
        logger.info(f"\n[EXPERIMENT AGENTS] Iteration {iteration}")

        update_agent_status("experiment", "running", {
            "iteration": iteration,
            "interval_minutes": interval_minutes,
            "num_parallel_agents": num_parallel_agents
        })

        try:
            await run_experiment_iteration(num_parallel_agents=num_parallel_agents)
        except Exception as e:
            logger.error(f"[EXPERIMENT AGENTS] Error in iteration {iteration}: {e}", exc_info=True)

            # Don't retry on iteration level - individual agents handle their own retries
            # Just skip to next iteration

        logger.info(f"[EXPERIMENT AGENTS] Waiting {interval_minutes} minutes until next iteration...")

        next_run = datetime.now().timestamp() + interval_seconds
        update_agent_status("experiment", "waiting", {
            "iteration": iteration,
            "interval_minutes": interval_minutes,
            "num_parallel_agents": num_parallel_agents,
            "next_run": next_run
        })

        await asyncio.sleep(interval_seconds)


async def log_agent_loop(interval_minutes: int = 60):
    """
    Run log agent loop indefinitely with retry logic.

    Args:
        interval_minutes: Time to wait between iterations (default: 60 = 1 hour)
    """
    interval_seconds = interval_minutes * 60

    logger.info(f"[LOG AGENT] Loop started (interval: {interval_minutes} minutes)")

    # Wait a bit before first run to let some activity accumulate
    logger.info(f"[LOG AGENT] Waiting 5 minutes before first run...")
    await asyncio.sleep(300)

    iteration = 0
    while True:
        iteration += 1
        logger.info(f"\n[LOG AGENT] Iteration {iteration}")

        update_agent_status("log", "running", {
            "iteration": iteration,
            "interval_minutes": interval_minutes
        })

        try:
            # Save state before starting
            save_agent_state("log", {
                "task_description": "Generating log summary for last 12 hours",
                "progress": {},
                "context": {"hours_back": 12},
                "started_at": datetime.now().isoformat(),
                "attempt_count": 0,
            })

            log_entry = await run_log_cycle(hours_back=12)
            if log_entry:
                # Log the entry content
                preview = log_entry[:500] + "..." if len(log_entry) > 500 else log_entry
                logger.info(f"[LOG AGENT] Log entry created: {preview}")
                logger.info(f"[LOG AGENT] ✓ Log entry saved to notes/log.md")
            else:
                logger.info(f"[LOG AGENT] No new activity to log")

            # Clear state on success
            clear_agent_state("log")

        except Exception as e:
            logger.error(f"[LOG AGENT] Error in iteration {iteration}: {e}", exc_info=True)

            # Check if we should retry
            if should_retry("log", max_attempts=3):
                state = load_agent_state("log", max_age_hours=4)
                retry_delay = get_retry_delay(state["attempt_count"]) if state else 300
                logger.info(f"[LOG AGENT] Will retry in {retry_delay}s (attempt {state['attempt_count'] if state else 1})")
                await asyncio.sleep(retry_delay)
                continue
            else:
                logger.error(f"[LOG AGENT] Max retries reached, skipping to next iteration")

        logger.info(f"[LOG AGENT] Waiting {interval_minutes} minutes until next iteration...")

        next_run = datetime.now().timestamp() + interval_seconds
        update_agent_status("log", "waiting", {
            "iteration": iteration,
            "interval_minutes": interval_minutes,
            "next_run": next_run
        })

        await asyncio.sleep(interval_seconds)


async def run_scaler_iteration(num_parallel_agents: int = 3):
    """
    Run one iteration of parallel scaler agents.

    Args:
        num_parallel_agents: Number of agents to run in parallel (default: 3)
    """
    from agents.work_tracker import print_active_work, get_claimed_experiments

    # Show what's currently being scaled
    claimed = get_claimed_experiments()
    if claimed:
        logger.info(f"[SCALER AGENTS] Currently scaling: {', '.join(claimed)}")

    logger.info(f"[SCALER AGENTS] Launching {num_parallel_agents} parallel agents...")

    # Launch multiple scaler agents in parallel
    tasks = []
    for i in range(num_parallel_agents):
        agent_id = f"scaler-{i+1}"
        tasks.append(run_single_scaler_agent(agent_id))

    # Wait for all agents to complete
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Log results
    successes = sum(1 for r in results if not isinstance(r, Exception))
    failures = sum(1 for r in results if isinstance(r, Exception))

    logger.info(f"[SCALER AGENTS] ✓ Iteration complete: {successes} succeeded, {failures} failed")


async def run_single_scaler_agent(agent_id: str):
    """
    Run a single scaler agent with error handling.

    Args:
        agent_id: Agent identifier
    """
    try:
        message_count = 0
        async for msg in run_scaler_cycle(agent_id=agent_id):
            message_count += 1
            # Log all messages from Claude SDK
            log_agent_message(agent_id, msg, message_count)

        logger.info(f"[{agent_id}] ✓ Complete")

    except Exception as e:
        logger.error(f"[{agent_id}] ✗ Error: {e}", exc_info=True)
        raise

async def scaler_agent_loop(interval_minutes: int = 180, num_parallel_agents: int = 3):
    """
    Run experiment scaler agent loop indefinitely with retry logic.

    Args:
        interval_minutes: Time to wait between iterations (default: 180 = 3 hours)
        num_parallel_agents: Number of agents to run in parallel (default: 3)
    """
    interval_seconds = interval_minutes * 60

    logger.info(f"[SCALER AGENTS] Loop started (interval: {interval_minutes} minutes, {num_parallel_agents} parallel agents)")

    # Wait before first run to let experiments complete
    logger.info(f"[SCALER AGENTS] Waiting 30 minutes before first run...")
    await asyncio.sleep(1800)

    iteration = 0
    while True:
        iteration += 1
        logger.info(f"\n[SCALER AGENTS] Iteration {iteration}")

        try:
            await run_scaler_iteration(num_parallel_agents=num_parallel_agents)

        except Exception as e:
            logger.error(f"[SCALER AGENTS] Error in iteration {iteration}: {e}", exc_info=True)

            # Don't retry on iteration level - individual agents handle their own errors
            # Just skip to next iteration

        logger.info(f"[SCALER AGENTS] Waiting {interval_minutes} minutes until next iteration...")
        await asyncio.sleep(interval_seconds)


async def run_forever(trick_interval: int = 10, research_interval: int = 15, experiment_interval: int = 15, log_interval: int = 60, scaler_interval: int = 180, num_experiment_agents: int = 10, num_scaler_agents: int = 3):
    """
    Run all loops concurrently with different intervals.

    Args:
        trick_interval: Minutes between trick search iterations (default: 10)
        research_interval: Minutes between research agent iterations (default: 15)
        experiment_interval: Minutes between experiment agent iterations (default: 15)
        log_interval: Minutes between log agent iterations (default: 60)
        scaler_interval: Minutes between scaler agent iterations (default: 180 = 3 hours)
        num_experiment_agents: Number of parallel experiment agents (default: 10)
        num_scaler_agents: Number of parallel scaler agents (default: 3)
    """
    logger.info("=" * 80)
    logger.info("MAD Architecture Search Runner Started")
    logger.info("=" * 80)
    logger.info(f"Trick Search Agent: every {trick_interval} minutes")
    logger.info(f"Research Agent: every {research_interval} minutes")
    logger.info(f"Experiment Agents: {num_experiment_agents} parallel agents, every {experiment_interval} minutes")
    logger.info(f"Log Agent: every {log_interval} minutes")
    logger.info(f"Scaler Agents: {num_scaler_agents} parallel agents, every {scaler_interval} minutes")
    logger.info(f"Proposals: {PROPOSALS_DIR}")
    logger.info(f"Tricks: {TRICKS_DIR}")
    logger.info(f"Papers: {PAPERS_DIR}")
    logger.info("=" * 80 + "\n")

    # Run all five loops concurrently
    await asyncio.gather(
        trick_search_loop(trick_interval),
        research_agent_loop(research_interval),
        experiment_agent_loop(experiment_interval, num_experiment_agents),
        log_agent_loop(log_interval),
        scaler_agent_loop(scaler_interval, num_scaler_agents),
    )


async def main():
    parser = argparse.ArgumentParser(
        description="MAD Architecture Search Runner - Autonomous trick discovery and proposal generation"
    )
    parser.add_argument(
        "--trick-interval",
        type=int,
        default=10,
        help="Minutes between trick search iterations (default: 10)"
    )
    parser.add_argument(
        "--research-interval",
        type=int,
        default=30,
        help="Minutes between research agent iterations (default: 30)"
    )
    parser.add_argument(
        "--experiment-interval",
        type=int,
        default=15,
        help="Minutes between experiment agent iterations (default: 15 minutes)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=60,
        help="Minutes between log agent iterations (default: 60 = 1 hour)"
    )
    parser.add_argument(
        "--scaler-interval",
        type=int,
        default=180,
        help="Minutes between scaler agent iterations (default: 180 = 3 hours)"
    )
    parser.add_argument(
        "--num-experiment-agents",
        type=int,
        default=10,
        help="Number of parallel experiment agents to run (default: 10)"
    )
    parser.add_argument(
        "--num-scaler-agents",
        type=int,
        default=3,
        help="Number of parallel scaler agents to run (default: 3)"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one iteration of each agent then exit (for testing)"
    )
    parser.add_argument(
        "--once-tricks",
        action="store_true",
        help="Run one trick search iteration then exit"
    )
    parser.add_argument(
        "--once-research",
        action="store_true",
        help="Run one research iteration then exit"
    )
    parser.add_argument(
        "--once-experiment",
        action="store_true",
        help="Run one experiment iteration then exit"
    )
    parser.add_argument(
        "--once-log",
        action="store_true",
        help="Run one log agent iteration then exit"
    )
    parser.add_argument(
        "--once-scaler",
        action="store_true",
        help="Run one scaler agent iteration then exit"
    )
    args = parser.parse_args()

    if args.once:
        logger.info("Running single iteration of each agent (--once mode)")
        logger.info("Running trick search iteration...")
        await run_trick_search_iteration()
        logger.info("\nRunning research iteration...")
        await run_research_iteration()
        logger.info(f"\nRunning experiment iteration with {args.num_experiment_agents} parallel agents...")
        await run_experiment_iteration(num_parallel_agents=args.num_experiment_agents)
        logger.info("\nRunning log agent iteration...")
        await run_log_cycle(hours_back=12)
    elif args.once_tricks:
        logger.info("Running single trick search iteration (--once-tricks mode)")
        await run_trick_search_iteration()
    elif args.once_research:
        logger.info("Running single research iteration (--once-research mode)")
        await run_research_iteration()
    elif args.once_experiment:
        logger.info(f"Running single experiment iteration with {args.num_experiment_agents} parallel agents (--once-experiment mode)")
        await run_experiment_iteration(num_parallel_agents=args.num_experiment_agents)
    elif args.once_log:
        logger.info("Running single log agent iteration (--once-log mode)")
        await run_log_cycle(hours_back=12)
    elif args.once_scaler:
        logger.info(f"Running single scaler agent iteration with {args.num_scaler_agents} parallel agents (--once-scaler mode)")
        await run_scaler_iteration(num_parallel_agents=args.num_scaler_agents)
    else:
        await run_forever(
            trick_interval=args.trick_interval,
            research_interval=args.research_interval,
            experiment_interval=args.experiment_interval,
            log_interval=args.log_interval,
            scaler_interval=args.scaler_interval,
            num_experiment_agents=args.num_experiment_agents,
            num_scaler_agents=args.num_scaler_agents,
        )




if __name__ == "__main__":
    asyncio.run(main())

