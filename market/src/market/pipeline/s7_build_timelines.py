"""Stage 7: Build evidence timelines for each conjecture."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import date

from tqdm import tqdm

from ..config import PROCESSED_DIR, ensure_dirs
from ..db import MarketDB
from ..models import Conjecture, ConjectureTimeline, EvidenceEvent

log = logging.getLogger(__name__)


def make_event_id(conjecture_id: str, paper_id: str) -> str:
    h = hashlib.sha256(f"{conjecture_id}:{paper_id}".encode()).hexdigest()
    return h[:12]


def run():
    ensure_dirs()
    db = MarketDB()
    timelines_dir = PROCESSED_DIR / "timelines"

    conjectures = db.get_all_conjectures()
    log.info("Building timelines for %d conjectures", len(conjectures))

    # Build citation lookup: cited_paper_id -> [(citing_paper_id, intents)]
    edges = db.conn.execute(
        "SELECT citing_paper_id, cited_paper_id, intents_json, is_influential FROM citation_edges"
    ).fetchall()

    cited_by: dict[str, list[dict]] = {}
    for e in edges:
        cited_by.setdefault(e["cited_paper_id"], []).append({
            "citing_paper_id": e["citing_paper_id"],
            "intents": json.loads(e["intents_json"]) if e["intents_json"] else [],
            "is_influential": bool(e["is_influential"]),
        })

    # Load dependency graph
    graph_path = PROCESSED_DIR / "graph.json"
    dep_graph: dict[str, list[str]] = {}
    if graph_path.exists():
        dep_graph = json.loads(graph_path.read_text())

    timeline_count = 0
    event_count = 0

    for c in tqdm(conjectures, desc="Building timelines"):
        cid = c["conjecture_id"]
        source_pid = c["source_paper_id"]

        # Find papers that cite the source paper
        citers = cited_by.get(source_pid, [])
        events: list[EvidenceEvent] = []

        for citer in citers:
            citing_pid = citer["citing_paper_id"]
            paper = db.get_paper(citing_pid)
            if not paper or not paper.get("abstract"):
                continue

            # Get publication date
            try:
                ev_date = date.fromisoformat(paper["publication_date"]) if paper.get("publication_date") else None
            except ValueError:
                ev_date = None
            if not ev_date:
                ev_date = date(paper.get("year", 2020), 1, 1)

            # Direction heuristic
            intents = citer["intents"]
            if "result" in intents:
                direction = "supports"
            elif "methodology" in intents:
                direction = "extends"
            else:
                direction = "extends"

            # Strength = citation velocity of the citing paper
            strength = paper.get("citation_velocity") or 0.0

            eid = make_event_id(cid, citing_pid)
            ev = EvidenceEvent(
                event_id=eid,
                conjecture_id=cid,
                paper_id=citing_pid,
                date=ev_date,
                direction=direction,
                strength=strength,
                abstract_text=paper["abstract"],
            )
            events.append(ev)
            db.upsert_evidence(ev)
            event_count += 1

        # Sort chronologically
        events.sort(key=lambda e: e.date)

        conjecture_model = Conjecture(
            conjecture_id=cid,
            claim_text=c["claim_text"],
            level=c["level"],
            source_paper_id=source_pid,
            created_date=date.fromisoformat(c["created_date"]),
            task=c.get("task"),
            dataset=c.get("dataset"),
            metric=c.get("metric"),
            value=c.get("value"),
        )

        timeline = ConjectureTimeline(
            conjecture_id=cid,
            conjecture=conjecture_model,
            events=events,
            dependency_ids=dep_graph.get(cid, []),
        )

        (timelines_dir / f"{cid}.json").write_text(timeline.model_dump_json(indent=2))
        timeline_count += 1

    db.commit()
    db.close()

    print(f"Done. Built {timeline_count} timelines with {event_count} total evidence events.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run()
