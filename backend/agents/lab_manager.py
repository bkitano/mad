"""
Lab Manager Agent

Monitors and manages cloud resources to prevent runaway costs.

The Lab Manager:
1. Monitors Modal jobs/apps running
2. Enforces max 5 concurrent jobs
3. Ensures only T4 or smaller GPUs are used
4. Kills jobs running longer than 1-2 hours
5. Logs all enforcement actions

Usage:
    # One-shot check
    python -m agents.lab_manager

    # Programmatic usage
    from agents.lab_manager import run_lab_manager_cycle
    await run_lab_manager_cycle()
"""

import asyncio
import json
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional


PROJECT_ROOT = Path(__file__).parent.parent
NOTES_DIR = PROJECT_ROOT / "notes"
LAB_LOG = NOTES_DIR / "lab_manager_log.md"

# Configuration
MAX_CONCURRENT_JOBS = 5
MAX_JOB_DURATION_HOURS = 2
ALLOWED_GPU_TYPES = ["t4", "cpu", None]  # T4 or smaller (CPU jobs are fine)


def get_modal_apps() -> List[Dict]:
    """
    Get list of running Modal apps.

    Returns:
        List of app dicts with metadata
    """
    try:
        result = subprocess.run(
            ["modal", "app", "list", "--json"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            print(f"[LAB MANAGER] Warning: modal app list failed: {result.stderr}")
            return []

        if not result.stdout.strip():
            return []

        apps = json.loads(result.stdout)
        return apps if isinstance(apps, list) else []

    except subprocess.TimeoutExpired:
        print("[LAB MANAGER] Warning: modal app list timed out")
        return []
    except json.JSONDecodeError:
        print("[LAB MANAGER] Warning: Could not parse modal app list output")
        return []
    except Exception as e:
        print(f"[LAB MANAGER] Error getting apps: {e}")
        return []


def get_modal_runs() -> List[Dict]:
    """
    Get list of running Modal jobs/runs.

    Returns:
        List of run dicts with metadata
    """
    try:
        result = subprocess.run(
            ["modal", "run", "list", "--json"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            # modal run list might not exist in all versions, try alternative
            print(f"[LAB MANAGER] modal run list not available, checking apps only")
            return []

        if not result.stdout.strip():
            return []

        runs = json.loads(result.stdout)
        return runs if isinstance(runs, list) else []

    except subprocess.TimeoutExpired:
        print("[LAB MANAGER] Warning: modal run list timed out")
        return []
    except json.JSONDecodeError:
        print("[LAB MANAGER] Warning: Could not parse modal run list output")
        return []
    except FileNotFoundError:
        # modal CLI might not be installed
        print("[LAB MANAGER] Warning: modal CLI not found")
        return []
    except Exception as e:
        print(f"[LAB MANAGER] Error getting runs: {e}")
        return []


def stop_modal_app(app_id: str, reason: str) -> bool:
    """
    Stop a Modal app.

    Args:
        app_id: The app ID to stop
        reason: Reason for stopping

    Returns:
        True if successful
    """
    try:
        print(f"[LAB MANAGER] Stopping app {app_id}: {reason}")

        result = subprocess.run(
            ["modal", "app", "stop", app_id],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=30
        )

        success = result.returncode == 0

        if success:
            log_enforcement_action(f"Stopped app {app_id}", reason)
        else:
            print(f"[LAB MANAGER] Failed to stop app: {result.stderr}")

        return success

    except Exception as e:
        print(f"[LAB MANAGER] Error stopping app {app_id}: {e}")
        return False


def check_gpu_type(gpu_spec: Optional[str]) -> bool:
    """
    Check if GPU type is allowed.

    Args:
        gpu_spec: GPU specification string (e.g., "T4", "A100", etc.)

    Returns:
        True if allowed, False otherwise
    """
    if not gpu_spec:
        return True  # No GPU = CPU only = allowed

    gpu_lower = gpu_spec.lower()

    # Check if it matches allowed types
    for allowed in ALLOWED_GPU_TYPES:
        if allowed and allowed in gpu_lower:
            return True

    return False


def parse_duration(started_at: str) -> float:
    """
    Parse duration from start time.

    Args:
        started_at: ISO timestamp or relative time string

    Returns:
        Duration in hours
    """
    try:
        # Try parsing ISO format
        start_time = datetime.fromisoformat(started_at.replace('Z', '+00:00'))
        duration = datetime.now(start_time.tzinfo) - start_time
        return duration.total_seconds() / 3600

    except Exception:
        # If parsing fails, return 0 (assume recent)
        return 0


def log_enforcement_action(action: str, reason: str):
    """
    Log an enforcement action to the lab manager log.

    Args:
        action: What action was taken
        reason: Why it was taken
    """
    NOTES_DIR.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Initialize log if needed
    if not LAB_LOG.exists():
        header = "# Lab Manager - Enforcement Log\n\n"
        header += "Actions taken to prevent runaway cloud costs.\n\n"
        header += "---\n\n"
        LAB_LOG.write_text(header)

    existing = LAB_LOG.read_text()

    entry = f"## {timestamp}\n\n"
    entry += f"**Action**: {action}\n\n"
    entry += f"**Reason**: {reason}\n\n"
    entry += "---\n\n"

    LAB_LOG.write_text(existing + entry)
    print(f"[LAB MANAGER] Logged: {action}")


async def run_lab_manager_cycle() -> Dict[str, any]:
    """
    Run one cycle of lab manager checks.

    Returns:
        Dict with summary of actions taken
    """
    print("[LAB MANAGER] Starting resource check...")

    actions_taken = []
    warnings = []

    # Get running apps and jobs
    apps = get_modal_apps()
    runs = get_modal_runs()

    total_jobs = len(apps) + len(runs)

    print(f"[LAB MANAGER] Found {len(apps)} apps, {len(runs)} runs")

    # Check 1: Max concurrent jobs
    if total_jobs > MAX_CONCURRENT_JOBS:
        warning = f"WARNING: {total_jobs} concurrent jobs (max: {MAX_CONCURRENT_JOBS})"
        print(f"[LAB MANAGER] {warning}")
        warnings.append(warning)

        # Stop oldest apps first
        sorted_apps = sorted(apps, key=lambda a: a.get('created_at', ''), reverse=False)
        to_stop = total_jobs - MAX_CONCURRENT_JOBS

        for app in sorted_apps[:to_stop]:
            app_id = app.get('app_id') or app.get('id')
            if app_id:
                success = stop_modal_app(app_id, f"Exceeded max concurrent jobs ({total_jobs}/{MAX_CONCURRENT_JOBS})")
                if success:
                    actions_taken.append(f"Stopped app {app_id}")

    # Check 2: GPU types
    for app in apps:
        gpu_type = app.get('gpu_type') or app.get('gpu')

        if gpu_type and not check_gpu_type(gpu_type):
            warning = f"Disallowed GPU type: {gpu_type} in app {app.get('app_id', 'unknown')}"
            print(f"[LAB MANAGER] {warning}")
            warnings.append(warning)

            app_id = app.get('app_id') or app.get('id')
            if app_id:
                success = stop_modal_app(app_id, f"Disallowed GPU type: {gpu_type} (only T4 or smaller allowed)")
                if success:
                    actions_taken.append(f"Stopped app {app_id} (bad GPU)")

    # Check 3: Job duration
    for app in apps:
        started_at = app.get('created_at') or app.get('started_at')

        if started_at:
            duration_hours = parse_duration(started_at)

            if duration_hours > MAX_JOB_DURATION_HOURS:
                warning = f"Long-running job: {duration_hours:.1f}h in app {app.get('app_id', 'unknown')}"
                print(f"[LAB MANAGER] {warning}")
                warnings.append(warning)

                app_id = app.get('app_id') or app.get('id')
                if app_id:
                    success = stop_modal_app(app_id, f"Exceeded max duration ({duration_hours:.1f}h > {MAX_JOB_DURATION_HOURS}h)")
                    if success:
                        actions_taken.append(f"Stopped app {app_id} (timeout)")

    # Summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_jobs": total_jobs,
        "apps_checked": len(apps),
        "runs_checked": len(runs),
        "actions_taken": actions_taken,
        "warnings": warnings,
        "status": "clean" if not actions_taken and not warnings else "actions_required"
    }

    if actions_taken:
        print(f"[LAB MANAGER] ✓ Took {len(actions_taken)} enforcement actions")
    else:
        print(f"[LAB MANAGER] ✓ All resources within limits")

    return summary


# =============================================================================
# CLI Entry Point
# =============================================================================

async def main():
    """CLI entry point for running the lab manager once."""
    print("=" * 60)
    print("MAD Architecture Search - Lab Manager")
    print("=" * 60)

    summary = await run_lab_manager_cycle()

    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    print(f"Total jobs: {summary['total_jobs']}")
    print(f"Actions taken: {len(summary['actions_taken'])}")
    print(f"Warnings: {len(summary['warnings'])}")
    print(f"Status: {summary['status']}")

    if summary['actions_taken']:
        print("\nActions:")
        for action in summary['actions_taken']:
            print(f"  - {action}")

    if summary['warnings']:
        print("\nWarnings:")
        for warning in summary['warnings']:
            print(f"  - {warning}")


if __name__ == "__main__":
    asyncio.run(main())
