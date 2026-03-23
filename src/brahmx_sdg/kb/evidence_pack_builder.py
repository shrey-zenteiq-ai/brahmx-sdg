"""
Evidence Pack Builder — builds task-local evidence bundles from CFT + chunks.
Mirrors the reference repo implementation with Pydantic schema integration.
"""
from __future__ import annotations
import json, re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
import numpy as np
from rank_bm25 import BM25Okapi
import structlog

logger = structlog.get_logger()
DEFAULT_TOP_K = 12
CHUNK_RELEVANCE_THRESHOLD = 0.05
LOW_CONFIDENCE_THRESHOLD = 0.60


class EvidencePackBuilder:
    def __init__(self, top_k: int = DEFAULT_TOP_K):
        self.top_k = top_k

    def build(self, section_spec: dict, kb_path: Path) -> dict[str, Any]:
        kb_path = Path(kb_path)
        section_id = section_spec.get("section_id", "UNKNOWN")
        query_tokens = self._build_query(section_spec)
        raw_chunks = self._load_chunks(kb_path)
        top_chunks = self._rank_chunks(query_tokens, raw_chunks)
        required_ids = [rc["claim_id"] for rc in section_spec.get("required_claims", [])]
        # TODO: load CFT and check claims
        missing = []

        return {
            "section_id": section_id,
            "top_chunks": top_chunks,
            "canonical_claims": [],
            "glossary_terms": [],
            "known_constraints": [],
            "retrieval_confidence": min(1.0, len(top_chunks) / max(1, self.top_k)),
            "missing_required_claims": missing,
            "cft_snapshot_hash": "",
            "kb_path": str(kb_path),
            "built_at": datetime.now(tz=timezone.utc).isoformat(),
        }

    def _build_query(self, spec: dict) -> list[str]:
        parts = re.split(r"[-_]", spec.get("section_id", ""))
        for obj in spec.get("objectives", []):
            parts.append(str(obj))
        return [t for t in " ".join(parts).lower().split() if len(t) > 2]

    def _load_chunks(self, kb_path: Path) -> list[dict]:
        chunks_dir = kb_path / "chunks"
        if not chunks_dir.exists():
            return []
        chunks = []
        for f in sorted(chunks_dir.rglob("*.json")):
            try:
                raw = json.loads(f.read_text())
                if isinstance(raw, list):
                    chunks.extend(raw)
                elif isinstance(raw, dict):
                    chunks.append(raw)
            except Exception:
                continue
        return chunks

    def _rank_chunks(self, query: list[str], chunks: list[dict]) -> list[dict]:
        if not chunks or not query:
            return []
        corpus = [re.sub(r"[^\w\s]", " ", c.get("text", "").lower()).split() for c in chunks]
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(query)
        max_s = float(scores.max()) if scores.size and scores.max() > 0 else 1.0
        ranked = sorted(zip(scores / max_s, chunks), key=lambda t: t[0], reverse=True)
        return [c for s, c in ranked[:self.top_k] if s >= CHUNK_RELEVANCE_THRESHOLD]
