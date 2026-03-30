from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel


class Paper(BaseModel):
    paper_id: str
    title: str
    abstract: str | None = None
    year: int
    publication_date: date | None = None
    venue: str | None = None
    citation_count: int = 0
    authors: list[str] = []
    arxiv_id: str | None = None
    doi: str | None = None
    s2_fields: list[str] = []
    citation_velocity: float | None = None


class CitationEdge(BaseModel):
    citing_paper_id: str
    cited_paper_id: str
    is_influential: bool = False
    intents: list[str] = []


class Conjecture(BaseModel):
    conjecture_id: str
    claim_text: str
    level: Literal["sota", "method"]
    source_paper_id: str
    created_date: date
    task: str | None = None
    dataset: str | None = None
    metric: str | None = None
    value: str | None = None


class EvidenceEvent(BaseModel):
    event_id: str
    conjecture_id: str
    paper_id: str
    date: date
    direction: Literal["supports", "contradicts", "extends"]
    strength: float = 0.0
    abstract_text: str = ""


class ConjectureTimeline(BaseModel):
    conjecture_id: str
    conjecture: Conjecture
    events: list[EvidenceEvent] = []
    dependency_ids: list[str] = []
