"""Stage 1: Fetch ML papers from Semantic Scholar bulk search API."""

from __future__ import annotations

import json
import logging
from datetime import date

from tqdm import tqdm

from ..clients.semantic_scholar import S2Client
from ..config import RAW_DIR, ensure_dirs
from ..db import MarketDB
from ..models import Paper

log = logging.getLogger(__name__)


def parse_paper(raw: dict) -> Paper | None:
    """Convert raw S2 API response to a Paper model."""
    pid = raw.get("paperId")
    if not pid or not raw.get("title"):
        return None

    ext = raw.get("externalIds") or {}
    pub_date = None
    if raw.get("publicationDate"):
        try:
            pub_date = date.fromisoformat(raw["publicationDate"])
        except ValueError:
            pass

    authors = [a.get("name", "") for a in (raw.get("authors") or []) if a.get("name")]
    fields = [f.get("category", "") for f in (raw.get("s2FieldsOfStudy") or []) if f.get("category")]

    return Paper(
        paper_id=pid,
        title=raw["title"],
        abstract=raw.get("abstract"),
        year=raw.get("year", 0),
        publication_date=pub_date,
        venue=raw.get("venue"),
        citation_count=raw.get("citationCount", 0),
        authors=authors,
        arxiv_id=ext.get("ArXiv"),
        doi=ext.get("DOI"),
        s2_fields=fields,
    )


def run(
    query: str = "machine learning",
    year: str = "2020-2024",
    min_citations: int = 10,
):
    ensure_dirs()
    db = MarketDB()
    client = S2Client()
    papers_dir = RAW_DIR / "papers"

    existing_ids = db.get_paper_ids()
    log.info("Existing papers in DB: %d", len(existing_ids))

    token = None
    total_fetched = 0
    total_new = 0

    pbar = tqdm(desc="Fetching papers", unit=" papers")

    try:
        while True:
            result = client.fetch_papers_bulk(
                query=query, year=year, min_citations=min_citations, token=token,
            )
            batch = result.get("data", [])
            if not batch:
                break

            for raw in batch:
                paper = parse_paper(raw)
                if paper is None:
                    continue

                total_fetched += 1

                if paper.paper_id in existing_ids:
                    continue

                # Write raw JSON
                (papers_dir / f"{paper.paper_id}.json").write_text(
                    json.dumps(raw, default=str)
                )

                db.upsert_paper(paper)
                existing_ids.add(paper.paper_id)
                total_new += 1

            db.commit()
            pbar.update(len(batch))

            token = result.get("token")
            if not token:
                break
    finally:
        pbar.close()
        client.close()
        db.close()

    log.info("Fetched %d papers total, %d new", total_fetched, total_new)
    print(f"Done. Fetched {total_fetched} papers, {total_new} new.")


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Fetch ML papers from Semantic Scholar")
    parser.add_argument("--query", default="machine learning")
    parser.add_argument("--year", default="2020-2024")
    parser.add_argument("--min-citations", type=int, default=10)
    args = parser.parse_args()

    run(query=args.query, year=args.year, min_citations=args.min_citations)
