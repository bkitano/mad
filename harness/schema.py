from __future__ import annotations

from pydantic import BaseModel


class Criterion(BaseModel):
    name: str
    metric_key: str
    target: float
    comparator: str = ">="
    required: bool = True
    description: str = ""


class ExperimentCriteria(BaseModel):
    task: str
    criteria: list[Criterion]
    time_budget_minutes: int = 120
    allowed_edit_globs: list[str] = ["models/**", "configs/**"]
    forbidden_edit_globs: list[str] = ["train/**", "tasks/**", "evaluate.py"]
    baseline_config: str | None = None


class CriterionResult(BaseModel):
    name: str
    metric_key: str
    target: float
    achieved: float | None
    passed: bool


class EvaluationVerdict(BaseModel):
    overall_pass: bool
    results: list[CriterionResult]
    training_completed: bool
    wall_time_seconds: float | None = None
    raw_metrics: dict = {}
