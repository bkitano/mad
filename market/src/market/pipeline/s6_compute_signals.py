"""Stage 6: Compute citation velocity strength signals."""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta

from tqdm import tqdm

from ..config import RAW_DIR, ensure_dirs
from ..db import MarketDB

log = logging.getLogger(__name__)


def run():
    ensure_dirs()
    db = MarketDB()

    # Get all papers with publication dates
    rows = db.conn.execute(
        "SELECT paper_id, publication_date, citation_count FROM papers WHERE publication_date IS NOT NULL"
    ).fetchall()

    log.info("Computing citation velocity for %d papers", len(rows))

    # Build a lookup of all citation edges: cited_paper_id -> list of citing_paper_ids
    citation_rows = db.conn.execute(
        "SELECT citing_paper_id, cited_paper_id FROM citation_edges"
    ).fetchall()

    cited_by: dict[str, list[str]] = {}
    for cr in citation_rows:
        cited_by.setdefault(cr["cited_paper_id"], []).append(cr["citing_paper_id"])

    # Get publication dates for all papers
    all_dates: dict[str, date | None] = {}
    date_rows = db.conn.execute("SELECT paper_id, publication_date FROM papers").fetchall()
    for dr in date_rows:
        try:
            all_dates[dr["paper_id"]] = date.fromisoformat(dr["publication_date"]) if dr["publication_date"] else None
        except ValueError:
            all_dates[dr["paper_id"]] = None

    velocities: dict[str, float] = {}

    for row in tqdm(rows, desc="Computing velocities"):
        pid = row["paper_id"]
        try:
            pub_date = date.fromisoformat(row["publication_date"])
        except ValueError:
            continue

        window_end = pub_date + timedelta(days=365)
        today = date.today()

        # Count citations within 12 months
        citers = cited_by.get(pid, [])
        citations_in_window = 0
        for citer_id in citers:
            citer_date = all_dates.get(citer_id)
            if citer_date and pub_date <= citer_date <= window_end:
                citations_in_window += 1

        # Time-adjust for recent papers: normalize by months elapsed
        months_elapsed = max(1, min(12, (min(today, window_end) - pub_date).days / 30))
        velocity = citations_in_window / months_elapsed * 12  # annualized

        velocities[pid] = velocity

    # Percentile-normalize to 0-1
    if velocities:
        sorted_vals = sorted(velocities.values())
        n = len(sorted_vals)
        for pid, vel in velocities.items():
            # Find percentile rank
            rank = sum(1 for v in sorted_vals if v <= vel)
            normalized = rank / n
            db.conn.execute(
                "UPDATE papers SET citation_velocity = ? WHERE paper_id = ?",
                (normalized, pid),
            )

    db.commit()
    db.close()

    print(f"Done. Computed citation velocity for {len(velocities)} papers.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run()
