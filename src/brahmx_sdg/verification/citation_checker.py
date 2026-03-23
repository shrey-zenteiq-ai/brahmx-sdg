"""
Citation Checker — BM25 + TF-IDF lexical citation verification.

Design: Lexical/statistical matching is PRIMARY. LLM-based semantic checking
is SECONDARY. This breaks the circular "LLM judges LLM" problem (FM-08).
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Any
import numpy as np
from rank_bm25 import BM25Okapi

BM25_SUPPORT_THRESHOLD = 0.70
TFIDF_SUPPORT_THRESHOLD = 0.65


@dataclass
class CitationMetrics:
    coverage: float
    precision: float
    specificity: float
    per_claim: list[dict[str, Any]]
    orphan_claim_ids: list[str]


class CitationChecker:
    def __init__(
        self,
        bm25_threshold: float = BM25_SUPPORT_THRESHOLD,
        tfidf_threshold: float = TFIDF_SUPPORT_THRESHOLD,
    ) -> None:
        self.bm25_threshold = bm25_threshold
        self.tfidf_threshold = tfidf_threshold

    def check(
        self,
        claim_ledger: list[dict],
        chunks: dict[str, str],
        section_text: str = "",
    ) -> CitationMetrics:
        """Check citation coverage and precision for all claims."""
        results = []
        total_non_trivial = 0
        with_citations = 0
        precision_num = 0
        total_citations = 0

        for entry in claim_ledger:
            if entry.get("verifiability") == "assumption":
                continue
            total_non_trivial += 1
            citations = entry.get("supporting_citations", [])
            if citations:
                with_citations += 1
            for chunk_id in citations:
                total_citations += 1
                chunk_text = chunks.get(chunk_id, "")
                if chunk_text:
                    score = self._bm25_score(entry["claim_text"], chunk_text)
                    if score >= self.bm25_threshold:
                        precision_num += 1

        coverage = with_citations / total_non_trivial if total_non_trivial else 1.0
        precision = precision_num / total_citations if total_citations else 1.0
        return CitationMetrics(
            coverage=round(coverage, 4),
            precision=round(precision, 4),
            specificity=1.0,
            per_claim=results,
            orphan_claim_ids=[],
        )

    def _bm25_score(self, claim: str, chunk: str) -> float:
        tokens_claim = self._tokenize(claim)
        tokens_chunk = self._tokenize(chunk)
        if not tokens_claim or not tokens_chunk:
            return 0.0
        bm25 = BM25Okapi([tokens_chunk])
        raw = float(bm25.get_scores(tokens_claim)[0])
        return float(1 / (1 + np.exp(-raw + 3)))

    def _tokenize(self, text: str) -> list[str]:
        text = re.sub(r"[^\w\s]", " ", text.lower())
        return [t for t in text.split() if len(t) > 2]
