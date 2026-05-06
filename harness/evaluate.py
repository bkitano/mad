"""
Post-training evaluation script for MAD experiments.

Loads criteria.yaml and results.json from the workspace, compares each metric
to its target, and writes verdict.json. Optionally reports back to the MAD API.

Usage:
    python -m harness.evaluate
    python -m harness.evaluate --criteria criteria.yaml --results results.json
"""

from __future__ import annotations

import argparse
import json
import operator
import os
import sys
from pathlib import Path

import yaml

from harness.schema import CriterionResult, EvaluationVerdict, ExperimentCriteria

COMPARATORS = {
    ">=": operator.ge,
    "<=": operator.le,
    ">": operator.gt,
    "<": operator.lt,
    "==": operator.eq,
}


def load_criteria(path: Path) -> ExperimentCriteria:
    with open(path) as f:
        data = yaml.safe_load(f)
    return ExperimentCriteria(**data)


def load_results(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def evaluate(criteria: ExperimentCriteria, metrics: dict) -> EvaluationVerdict:
    results: list[CriterionResult] = []
    all_required_pass = True

    for c in criteria.criteria:
        achieved = metrics.get(c.metric_key)
        if achieved is None:
            passed = False
        else:
            cmp = COMPARATORS.get(c.comparator)
            if cmp is None:
                raise ValueError(f"Unknown comparator {c.comparator!r} for criterion {c.name!r}")
            passed = cmp(float(achieved), c.target)

        if c.required and not passed:
            all_required_pass = False

        results.append(CriterionResult(
            name=c.name,
            metric_key=c.metric_key,
            target=c.target,
            achieved=float(achieved) if achieved is not None else None,
            passed=passed,
        ))

    training_completed = bool(metrics)

    return EvaluationVerdict(
        overall_pass=all_required_pass and training_completed,
        results=results,
        training_completed=training_completed,
        wall_time_seconds=metrics.get("wall_time_seconds"),
        raw_metrics=metrics,
    )


def report_verdict(verdict: EvaluationVerdict, service_url: str, experiment_id: str) -> None:
    import httpx

    url = f"{service_url.rstrip('/')}/experiments/{experiment_id}"
    status = "completed" if verdict.overall_pass else "failed"
    httpx.patch(
        url,
        json={"results": verdict.model_dump(), "status": status},
        timeout=10.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate experiment against criteria")
    parser.add_argument("--criteria", default="criteria.yaml", help="Path to criteria YAML")
    parser.add_argument("--results", default="results.json", help="Path to results JSON")
    parser.add_argument("--output", default="verdict.json", help="Path to write verdict")
    args = parser.parse_args()

    criteria_path = Path(args.criteria)
    results_path = Path(args.results)
    output_path = Path(args.output)

    if not criteria_path.exists():
        print(f"ERROR: criteria file not found: {criteria_path}", file=sys.stderr)
        sys.exit(2)

    criteria = load_criteria(criteria_path)

    if not results_path.exists():
        print(f"WARNING: results file not found: {results_path} — treating as empty", file=sys.stderr)
        metrics = {}
    else:
        metrics = load_results(results_path)

    verdict = evaluate(criteria, metrics)

    output_path.write_text(json.dumps(verdict.model_dump(), indent=2))
    print(f"Verdict written to {output_path}")

    for r in verdict.results:
        mark = "PASS" if r.passed else "FAIL"
        achieved_str = f"{r.achieved:.4f}" if r.achieved is not None else "MISSING"
        print(f"  [{mark}] {r.name}: {achieved_str} (target: {r.target})")

    print(f"\nOverall: {'PASS' if verdict.overall_pass else 'FAIL'}")

    service_url = os.environ.get("MAD_SERVICE_URL")
    experiment_id = os.environ.get("MAD_EXPERIMENT_ID")
    if service_url and experiment_id:
        try:
            report_verdict(verdict, service_url, experiment_id)
            print(f"Verdict reported to {service_url}")
        except Exception as e:
            print(f"WARNING: failed to report verdict: {e}", file=sys.stderr)

    sys.exit(0 if verdict.overall_pass else 1)


if __name__ == "__main__":
    main()
