import json
import sqlite3
from pathlib import Path

from .config import DB_PATH
from .models import CitationEdge, Conjecture, EvidenceEvent, Paper

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    paper_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    abstract TEXT,
    year INTEGER NOT NULL,
    publication_date TEXT,
    venue TEXT,
    citation_count INTEGER DEFAULT 0,
    authors_json TEXT,
    arxiv_id TEXT,
    doi TEXT,
    s2_fields_json TEXT,
    citation_velocity REAL
);

CREATE TABLE IF NOT EXISTS citation_edges (
    citing_paper_id TEXT NOT NULL,
    cited_paper_id TEXT NOT NULL,
    is_influential BOOLEAN DEFAULT FALSE,
    intents_json TEXT,
    PRIMARY KEY (citing_paper_id, cited_paper_id)
);

CREATE TABLE IF NOT EXISTS conjectures (
    conjecture_id TEXT PRIMARY KEY,
    claim_text TEXT NOT NULL,
    level TEXT NOT NULL,
    source_paper_id TEXT NOT NULL,
    created_date TEXT NOT NULL,
    task TEXT,
    dataset TEXT,
    metric TEXT,
    value TEXT
);

CREATE TABLE IF NOT EXISTS conjecture_dependencies (
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    weight REAL DEFAULT 1.0,
    intent TEXT,
    PRIMARY KEY (source_id, target_id)
);

CREATE TABLE IF NOT EXISTS evidence_events (
    event_id TEXT PRIMARY KEY,
    conjecture_id TEXT NOT NULL,
    paper_id TEXT NOT NULL,
    date TEXT NOT NULL,
    direction TEXT NOT NULL,
    strength REAL DEFAULT 0.0,
    abstract_text TEXT
);

CREATE INDEX IF NOT EXISTS idx_papers_year ON papers(year);
CREATE INDEX IF NOT EXISTS idx_papers_citations ON papers(citation_count);
CREATE INDEX IF NOT EXISTS idx_conjectures_level ON conjectures(level);
CREATE INDEX IF NOT EXISTS idx_conjectures_source ON conjectures(source_paper_id);
CREATE INDEX IF NOT EXISTS idx_evidence_conjecture ON evidence_events(conjecture_id);
CREATE INDEX IF NOT EXISTS idx_evidence_date ON evidence_events(date);
"""


class MarketDB:
    def __init__(self, db_path: Path = DB_PATH):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def close(self):
        self.conn.close()

    # --- Papers ---

    def upsert_paper(self, p: Paper):
        self.conn.execute(
            """INSERT INTO papers (paper_id, title, abstract, year, publication_date,
               venue, citation_count, authors_json, arxiv_id, doi, s2_fields_json, citation_velocity)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(paper_id) DO UPDATE SET
               citation_count=excluded.citation_count,
               citation_velocity=excluded.citation_velocity""",
            (
                p.paper_id, p.title, p.abstract, p.year,
                p.publication_date.isoformat() if p.publication_date else None,
                p.venue, p.citation_count,
                json.dumps(p.authors), p.arxiv_id, p.doi,
                json.dumps(p.s2_fields), p.citation_velocity,
            ),
        )

    def get_paper_ids(self) -> set[str]:
        rows = self.conn.execute("SELECT paper_id FROM papers").fetchall()
        return {r["paper_id"] for r in rows}

    def get_top_cited_papers(self, n: int) -> list[str]:
        rows = self.conn.execute(
            "SELECT paper_id FROM papers ORDER BY citation_count DESC LIMIT ?", (n,)
        ).fetchall()
        return [r["paper_id"] for r in rows]

    def get_papers_with_abstracts(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT paper_id, abstract FROM papers WHERE abstract IS NOT NULL AND abstract != ''"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_paper(self, paper_id: str) -> dict | None:
        row = self.conn.execute("SELECT * FROM papers WHERE paper_id = ?", (paper_id,)).fetchone()
        return dict(row) if row else None

    def paper_count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]

    # --- Citation Edges ---

    def upsert_citation(self, e: CitationEdge):
        self.conn.execute(
            """INSERT INTO citation_edges (citing_paper_id, cited_paper_id, is_influential, intents_json)
               VALUES (?, ?, ?, ?)
               ON CONFLICT DO NOTHING""",
            (e.citing_paper_id, e.cited_paper_id, e.is_influential, json.dumps(e.intents)),
        )

    def get_fetched_citation_paper_ids(self) -> set[str]:
        """Paper IDs we've already fetched citations for (as cited)."""
        rows = self.conn.execute(
            "SELECT DISTINCT cited_paper_id FROM citation_edges"
        ).fetchall()
        return {r["cited_paper_id"] for r in rows}

    def citation_count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM citation_edges").fetchone()[0]

    # --- Conjectures ---

    def upsert_conjecture(self, c: Conjecture):
        self.conn.execute(
            """INSERT INTO conjectures (conjecture_id, claim_text, level, source_paper_id,
               created_date, task, dataset, metric, value)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(conjecture_id) DO NOTHING""",
            (
                c.conjecture_id, c.claim_text, c.level, c.source_paper_id,
                c.created_date.isoformat(), c.task, c.dataset, c.metric, c.value,
            ),
        )

    def conjecture_count(self, level: str | None = None) -> int:
        if level:
            return self.conn.execute(
                "SELECT COUNT(*) FROM conjectures WHERE level = ?", (level,)
            ).fetchone()[0]
        return self.conn.execute("SELECT COUNT(*) FROM conjectures").fetchone()[0]

    def get_all_conjectures(self) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM conjectures").fetchall()
        return [dict(r) for r in rows]

    # --- Dependencies ---

    def upsert_dependency(self, source_id: str, target_id: str, weight: float = 1.0, intent: str = ""):
        self.conn.execute(
            """INSERT INTO conjecture_dependencies (source_id, target_id, weight, intent)
               VALUES (?, ?, ?, ?) ON CONFLICT DO NOTHING""",
            (source_id, target_id, weight, intent),
        )

    def dependency_count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM conjecture_dependencies").fetchone()[0]

    # --- Evidence Events ---

    def upsert_evidence(self, ev: EvidenceEvent):
        self.conn.execute(
            """INSERT INTO evidence_events (event_id, conjecture_id, paper_id, date, direction, strength, abstract_text)
               VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT DO NOTHING""",
            (ev.event_id, ev.conjecture_id, ev.paper_id, ev.date.isoformat(),
             ev.direction, ev.strength, ev.abstract_text),
        )

    def evidence_count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM evidence_events").fetchone()[0]

    # --- Utility ---

    def commit(self):
        self.conn.commit()

    def stats(self) -> dict:
        return {
            "papers": self.paper_count(),
            "citations": self.citation_count(),
            "conjectures_sota": self.conjecture_count("sota"),
            "conjectures_method": self.conjecture_count("method"),
            "conjectures_total": self.conjecture_count(),
            "dependencies": self.dependency_count(),
            "evidence_events": self.evidence_count(),
        }
