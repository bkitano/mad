from pathlib import Path
from dotenv import load_dotenv
import os

# Load root .env
_root = Path(__file__).resolve().parents[3]
load_dotenv(_root / ".env")

# Paths
MARKET_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = MARKET_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CHECKPOINTS_DIR = DATA_DIR / "checkpoints"
DB_PATH = DATA_DIR / "market.db"

# Semantic Scholar
S2_API_KEY = os.getenv("S2_API_KEY", "")
S2_BASE_URL = "https://api.semanticscholar.org/graph/v1"
S2_RATE_LIMIT = 1.0  # seconds between requests
S2_BULK_FIELDS = ",".join([
    "paperId", "externalIds", "title", "abstract", "year",
    "publicationDate", "venue", "citationCount", "authors",
    "s2FieldsOfStudy",
])

# Anthropic
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
LLM_MODEL = os.getenv("MARKET_LLM_MODEL", "claude-haiku-4-5-20251001")


def ensure_dirs():
    """Create all data directories if they don't exist."""
    for d in [
        RAW_DIR / "papers",
        RAW_DIR / "citations",
        PROCESSED_DIR / "conjectures",
        PROCESSED_DIR / "timelines",
        CHECKPOINTS_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)
