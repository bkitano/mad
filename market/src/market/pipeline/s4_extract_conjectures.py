"""Stage 4: Extract Level 2 (methodological) conjectures from abstracts using Claude."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import date

from tqdm import tqdm

from ..clients.anthropic_llm import LLMClient
from ..config import CHECKPOINTS_DIR, PROCESSED_DIR, ensure_dirs
from ..db import MarketDB
from ..models import Conjecture

log = logging.getLogger(__name__)

LLM_CACHE_DIR = CHECKPOINTS_DIR / "s4_llm_cache"


def make_conjecture_id(source_paper_id: str, claim_text: str) -> str:
    h = hashlib.sha256(f"{source_paper_id}:{claim_text}".encode()).hexdigest()
    return h[:12]


def run(limit: int | None = None, model: str | None = None):
    ensure_dirs()
    LLM_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    db = MarketDB()
    llm = LLMClient(model=model) if model else LLMClient()
    conj_dir = PROCESSED_DIR / "conjectures"

    papers = db.get_papers_with_abstracts()
    if limit:
        papers = papers[:limit]

    log.info("Processing %d papers with abstracts", len(papers))
    new_count = 0

    for p in tqdm(papers, desc="Extracting conjectures"):
        pid = p["paper_id"]
        abstract = p["abstract"]
        cache_path = LLM_CACHE_DIR / f"{pid}.json"

        # Check cache
        if cache_path.exists():
            claims = json.loads(cache_path.read_text())
        else:
            try:
                claims = llm.extract_conjectures(abstract)
            except Exception as e:
                log.warning("LLM extraction failed for %s: %s", pid, e)
                claims = []
            cache_path.write_text(json.dumps(claims, default=str))

        # Get paper's publication date
        paper = db.get_paper(pid)
        pub_date_str = paper.get("publication_date") if paper else None
        try:
            pub_date = date.fromisoformat(pub_date_str) if pub_date_str else date(int(paper["year"]), 1, 1)
        except (ValueError, TypeError):
            pub_date = date(2020, 1, 1)

        for claim in claims:
            text = claim.get("text", "").strip()
            confidence = claim.get("confidence", "medium")

            if not text or confidence == "low":
                continue

            cid = make_conjecture_id(pid, text)
            conj = Conjecture(
                conjecture_id=cid,
                claim_text=text,
                level="method",
                source_paper_id=pid,
                created_date=pub_date,
            )

            db.upsert_conjecture(conj)
            (conj_dir / f"{cid}.json").write_text(conj.model_dump_json(indent=2))
            new_count += 1

    db.commit()
    db.close()

    print(f"Done. Extracted {new_count} methodological conjectures from {len(papers)} papers.")


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Extract conjectures from paper abstracts via LLM")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N papers")
    parser.add_argument("--model", default=None, help="Override LLM model")
    args = parser.parse_args()

    run(limit=args.limit, model=args.model)
