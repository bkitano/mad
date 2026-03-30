"""Stage 5: Build conjecture dependency graph from citation edges."""

from __future__ import annotations

import json
import logging

from ..config import PROCESSED_DIR, ensure_dirs
from ..db import MarketDB

log = logging.getLogger(__name__)


def run():
    ensure_dirs()
    db = MarketDB()

    # Load all conjectures indexed by source_paper_id
    conjectures = db.get_all_conjectures()
    paper_to_conjectures: dict[str, list[str]] = {}
    for c in conjectures:
        pid = c["source_paper_id"]
        paper_to_conjectures.setdefault(pid, []).append(c["conjecture_id"])

    log.info("Building dependency graph from %d conjectures across %d papers",
             len(conjectures), len(paper_to_conjectures))

    # For each citation edge where both papers have conjectures, create dependencies
    edges = db.conn.execute(
        "SELECT citing_paper_id, cited_paper_id, is_influential, intents_json FROM citation_edges"
    ).fetchall()

    dep_count = 0
    graph: dict[str, list[str]] = {}

    for edge in edges:
        citing_pid = edge["citing_paper_id"]
        cited_pid = edge["cited_paper_id"]

        citing_conjs = paper_to_conjectures.get(citing_pid, [])
        cited_conjs = paper_to_conjectures.get(cited_pid, [])

        if not citing_conjs or not cited_conjs:
            continue

        # Weight: influential citations and methodology intents get higher weight
        intents = json.loads(edge["intents_json"]) if edge["intents_json"] else []
        weight = 0.5
        if edge["is_influential"]:
            weight += 0.3
        if "methodology" in intents:
            weight += 0.2
        elif "result" in intents:
            weight += 0.1

        intent = intents[0] if intents else "background"

        # Each conjecture in the citing paper depends on each conjecture in the cited paper
        for src_cid in citing_conjs:
            for tgt_cid in cited_conjs:
                if src_cid == tgt_cid:
                    continue
                db.upsert_dependency(src_cid, tgt_cid, weight=weight, intent=intent)
                graph.setdefault(src_cid, []).append(tgt_cid)
                dep_count += 1

    db.commit()

    # Write graph JSON
    graph_path = PROCESSED_DIR / "graph.json"
    graph_path.write_text(json.dumps(graph, indent=2))

    db.close()
    print(f"Done. Created {dep_count} dependency edges. Graph saved to {graph_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run()
