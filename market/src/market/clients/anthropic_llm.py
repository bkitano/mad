"""Anthropic Claude client for conjecture extraction."""

from __future__ import annotations

import json
import logging

import anthropic

from ..config import ANTHROPIC_API_KEY, LLM_MODEL

log = logging.getLogger(__name__)

EXTRACT_PROMPT = """\
Extract the core falsifiable claims from this ML paper abstract.

Rules:
- Each claim must be a statement that could be true or false.
- Do NOT include descriptions of what the paper does ("We propose X").
- DO include claims about what works, what's better, what's true ("X outperforms Y", "Z reduces Q").
- Return 1-5 claims maximum.
- Each claim should be self-contained (understandable without reading the paper).

Abstract:
{abstract}

Return valid JSON only: {{"claims": [{{"text": "...", "confidence": "high|medium|low"}}]}}"""

DIRECTION_PROMPT = """\
Given this scientific claim and a paper abstract, classify the relationship.

Claim: {claim}

Abstract: {abstract}

Does this paper's content support, contradict, or extend the claim?
- "supports": provides evidence that the claim is true
- "contradicts": provides evidence that the claim is false
- "extends": builds on or refines the claim without clearly supporting or contradicting it

Return valid JSON only: {{"direction": "supports|contradicts|extends"}}"""


class LLMClient:
    def __init__(self, model: str = LLM_MODEL):
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self.model = model

    def _call(self, prompt: str) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text

    def extract_conjectures(self, abstract: str) -> list[dict]:
        """Extract falsifiable claims from an abstract.

        Returns list of {"text": str, "confidence": "high"|"medium"|"low"}.
        """
        raw = self._call(EXTRACT_PROMPT.format(abstract=abstract))
        try:
            # Handle markdown code blocks
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            parsed = json.loads(raw.strip())
            return parsed.get("claims", [])
        except (json.JSONDecodeError, IndexError):
            log.warning("Failed to parse LLM response: %s", raw[:200])
            return []

    def classify_direction(self, claim: str, abstract: str) -> str:
        """Classify whether an abstract supports, contradicts, or extends a claim."""
        raw = self._call(DIRECTION_PROMPT.format(claim=claim, abstract=abstract))
        try:
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            parsed = json.loads(raw.strip())
            direction = parsed.get("direction", "extends")
            if direction in ("supports", "contradicts", "extends"):
                return direction
        except (json.JSONDecodeError, IndexError):
            pass
        return "extends"
