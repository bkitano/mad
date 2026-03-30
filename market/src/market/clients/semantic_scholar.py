"""Semantic Scholar API client with rate limiting and retries."""

from __future__ import annotations

import logging
import time

import httpx

from ..config import S2_API_KEY, S2_BASE_URL, S2_BULK_FIELDS, S2_RATE_LIMIT

log = logging.getLogger(__name__)

RETRY_STATUSES = {429, 500, 502, 503, 504}
MAX_RETRIES = 5


class S2Client:
    def __init__(self):
        headers = {}
        if S2_API_KEY:
            headers["x-api-key"] = S2_API_KEY
        self.client = httpx.Client(
            base_url=S2_BASE_URL,
            headers=headers,
            timeout=30.0,
        )
        self._last_request = 0.0

    def _throttle(self):
        elapsed = time.time() - self._last_request
        if elapsed < S2_RATE_LIMIT:
            time.sleep(S2_RATE_LIMIT - elapsed)
        self._last_request = time.time()

    def _get(self, path: str, params: dict | None = None) -> dict:
        for attempt in range(MAX_RETRIES):
            self._throttle()
            try:
                resp = self.client.get(path, params=params)
                if resp.status_code == 200:
                    return resp.json()
                if resp.status_code in RETRY_STATUSES:
                    wait = 2 ** attempt
                    log.warning("S2 %s returned %d, retrying in %ds", path, resp.status_code, wait)
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
            except httpx.ReadTimeout:
                wait = 2 ** attempt
                log.warning("S2 timeout on %s, retrying in %ds", path, wait)
                time.sleep(wait)
                continue
        raise RuntimeError(f"S2 API failed after {MAX_RETRIES} retries: {path}")

    def fetch_papers_bulk(
        self,
        query: str = "machine learning",
        year: str = "2020-2024",
        min_citations: int = 10,
        fields: str = S2_BULK_FIELDS,
        token: str | None = None,
    ) -> dict:
        """Fetch a page of papers from the bulk search endpoint.

        Returns dict with 'data' (list of papers) and optionally 'token' for next page.
        """
        params = {
            "query": query,
            "year": year,
            "minCitationCount": min_citations,
            "fields": fields,
        }
        if token:
            params["token"] = token
        return self._get("/paper/search/bulk", params=params)

    def fetch_citations(self, paper_id: str, limit: int = 1000) -> list[dict]:
        """Fetch papers that cite the given paper."""
        results = []
        offset = 0
        while True:
            data = self._get(
                f"/paper/{paper_id}/citations",
                params={
                    "fields": "citingPaper.paperId,intents,isInfluential",
                    "offset": offset,
                    "limit": min(limit - len(results), 1000),
                },
            )
            batch = data.get("data", [])
            if not batch:
                break
            results.extend(batch)
            offset += len(batch)
            if offset >= data.get("total", 0) or len(results) >= limit:
                break
        return results

    def fetch_references(self, paper_id: str, limit: int = 1000) -> list[dict]:
        """Fetch papers referenced by the given paper."""
        results = []
        offset = 0
        while True:
            data = self._get(
                f"/paper/{paper_id}/references",
                params={
                    "fields": "citedPaper.paperId,intents,isInfluential",
                    "offset": offset,
                    "limit": min(limit - len(results), 1000),
                },
            )
            batch = data.get("data", [])
            if not batch:
                break
            results.extend(batch)
            offset += len(batch)
            if offset >= data.get("total", 0) or len(results) >= limit:
                break
        return results

    def close(self):
        self.client.close()
