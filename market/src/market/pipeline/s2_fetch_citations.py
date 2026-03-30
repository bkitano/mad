"""Stage 2: Fetch citation edges for top-cited papers."""

from __future__ import annotations

import json
import logging

from tqdm import tqdm

from ..clients.semantic_scholar import S2Client
from ..config import CHECKPOINTS_DIR, RAW_DIR, ensure_dirs
from ..db import MarketDB
from ..models import CitationEdge

log = logging.getLogger(__name__)

CHECKPOINT_FILE = CHECKPOINTS_DIR / "s2_fetched_papers.json"


def load_checkpoint() -> set[str]:
    if CHECKPOINT_FILE.exists():
        return set(json.loads(CHECKPOINT_FILE.read_text()))
    return set()


def save_checkpoint(done: set[str]):
    CHECKPOINT_FILE.write_text(json.dumps(sorted(done)))


def run(top_n: int = 2000):
    ensure_dirs()
    db = MarketDB()
    client = S2Client()
    citations_dir = RAW_DIR / "citations"

    paper_ids = db.get_top_cited_papers(top_n)
    done = load_checkpoint()
    remaining = [pid for pid in paper_ids if pid not in done]

    log.info("Top %d papers, %d already fetched, %d remaining", top_n, len(done), len(remaining))

    try:
        for pid in tqdm(remaining, desc="Fetching citations"):
            # Fetch citations (papers that cite this one)
            try:
                citations = client.fetch_citations(pid)
            except Exception as e:
                log.warning("Failed to fetch citations for %s: %s", pid, e)
                citations = []

            # Fetch references (papers this one cites)
            try:
                references = client.fetch_references(pid)
            except Exception as e:
                log.warning("Failed to fetch references for %s: %s", pid, e)
                references = []

            # Save raw
            raw = {"citations": citations, "references": references}
            (citations_dir / f"{pid}.json").write_text(json.dumps(raw, default=str))

            # Store citation edges
            for c in citations:
                citing = c.get("citingPaper", {})
                citing_id = citing.get("paperId")
                if not citing_id:
                    continue
                db.upsert_citation(CitationEdge(
                    citing_paper_id=citing_id,
                    cited_paper_id=pid,
                    is_influential=c.get("isInfluential", False),
                    intents=c.get("intents", []),
                ))

            for r in references:
                cited = r.get("citedPaper", {})
                cited_id = cited.get("paperId")
                if not cited_id:
                    continue
                db.upsert_citation(CitationEdge(
                    citing_paper_id=pid,
                    cited_paper_id=cited_id,
                    is_influential=r.get("isInfluential", False),
                    intents=r.get("intents", []),
                ))

            db.commit()
            done.add(pid)
            save_checkpoint(done)
    finally:
        client.close()
        db.close()

    print(f"Done. Fetched citations for {len(done)} papers. {db.citation_count()} edges total.")


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Fetch citation edges from Semantic Scholar")
    parser.add_argument("--top-n", type=int, default=2000)
    args = parser.parse_args()

    run(top_n=args.top_n)
