"""Stage 3: Load Papers With Code SOTA tables as Level 1 conjectures."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import date

from tqdm import tqdm

from ..config import PROCESSED_DIR, RAW_DIR, ensure_dirs
from ..db import MarketDB
from ..models import Conjecture

log = logging.getLogger(__name__)

PWC_DATA_URL = "https://raw.githubusercontent.com/paperswithcode/paperswithcode-data/master/evaluation-tables.json"


def download_pwc_data() -> list[dict]:
    """Download PWC evaluation tables from GitHub archive."""
    import httpx

    cache_path = RAW_DIR / "pwc_evaluation_tables.json"
    if cache_path.exists():
        log.info("Using cached PWC data from %s", cache_path)
        return json.loads(cache_path.read_text())

    log.info("Downloading PWC evaluation tables...")
    resp = httpx.get(PWC_DATA_URL, timeout=60.0, follow_redirects=True)
    resp.raise_for_status()
    data = resp.json()
    cache_path.write_text(json.dumps(data))
    return data


def make_conjecture_id(source_paper_id: str, claim_text: str) -> str:
    h = hashlib.sha256(f"{source_paper_id}:{claim_text}".encode()).hexdigest()
    return h[:12]


def run():
    ensure_dirs()
    db = MarketDB()
    conj_dir = PROCESSED_DIR / "conjectures"

    # Build lookup: arxiv_id -> paper_id, doi -> paper_id
    paper_ids = db.get_paper_ids()
    arxiv_lookup: dict[str, str] = {}
    doi_lookup: dict[str, str] = {}

    for pid in paper_ids:
        p = db.get_paper(pid)
        if not p:
            continue
        arxiv = p.get("arxiv_id")
        doi = p.get("doi")
        if arxiv:
            arxiv_lookup[arxiv.lower()] = pid
        if doi:
            doi_lookup[doi.lower()] = pid

    log.info("Loaded %d arxiv IDs and %d DOIs for matching", len(arxiv_lookup), len(doi_lookup))

    data = download_pwc_data()
    total = 0
    matched = 0

    for task_entry in tqdm(data, desc="Processing PWC tasks"):
        task_name = task_entry.get("task", "")
        for dataset_entry in task_entry.get("datasets", []):
            dataset_name = dataset_entry.get("dataset", "")
            for sota in dataset_entry.get("sota", {}).get("rows", []):
                model_name = sota.get("model_name", "")
                metric_name = sota.get("metric_name", "")
                metric_value = sota.get("metric_value", "")
                paper_url = sota.get("paper_url", "")
                paper_date_str = sota.get("paper_date", "")

                if not model_name or not metric_value:
                    continue

                # Filter to 2020-2024
                try:
                    pd = date.fromisoformat(paper_date_str) if paper_date_str else None
                except ValueError:
                    pd = None

                if not pd or pd.year < 2020 or pd.year > 2024:
                    continue

                total += 1

                # Try to match to our corpus
                source_paper_id = None
                if paper_url:
                    # Extract arxiv ID from URL
                    url_lower = paper_url.lower()
                    if "arxiv.org" in url_lower:
                        parts = url_lower.rstrip("/").split("/")
                        arxiv_id = parts[-1].replace("v1", "").replace("v2", "").replace("v3", "")
                        source_paper_id = arxiv_lookup.get(arxiv_id)

                if not source_paper_id:
                    continue

                matched += 1

                claim = f"{model_name} achieves {metric_value} {metric_name} on {dataset_name} ({task_name})"
                cid = make_conjecture_id(source_paper_id, claim)

                conj = Conjecture(
                    conjecture_id=cid,
                    claim_text=claim,
                    level="sota",
                    source_paper_id=source_paper_id,
                    created_date=pd,
                    task=task_name,
                    dataset=dataset_name,
                    metric=metric_name,
                    value=str(metric_value),
                )

                db.upsert_conjecture(conj)
                (conj_dir / f"{cid}.json").write_text(conj.model_dump_json(indent=2))

    db.commit()
    db.close()

    print(f"Done. {total} PWC SOTA rows in 2020-2024, {matched} matched to corpus, {db.conjecture_count('sota')} stored.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run()
