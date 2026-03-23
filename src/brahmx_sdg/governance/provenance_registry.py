"""
Provenance Registry — stores manifests, hashes, lineages, promotion status.

Trust is meaningless if you cannot trace or revoke it.
No sample enters training without manifests, lineage, and release approval.
"""
from __future__ import annotations
from datetime import datetime, timezone
from typing import Any, Optional
import structlog
from brahmx_sdg.schemas import PromotionStatus

logger = structlog.get_logger()

class ProvenanceEntry:
    def __init__(self, record_id: str, record_type: str, bundle_hash: str,
                 parent_ids: list[str] = None, metadata: dict = None):
        self.record_id = record_id
        self.record_type = record_type
        self.bundle_hash = bundle_hash
        self.parent_ids = parent_ids or []
        self.promotion_status = PromotionStatus.PENDING
        self.metadata = metadata or {}
        self.registered_at = datetime.now(timezone.utc)
        self.promoted_at: Optional[datetime] = None
        self.rolled_back_at: Optional[datetime] = None

class ProvenanceRegistry:
    """In-memory registry (replace with DB-backed service in production)."""
    def __init__(self):
        self._store: dict[str, ProvenanceEntry] = {}

    def register(self, record_id: str, record_type: str, bundle_hash: str,
                 parent_ids: list[str] = None, metadata: dict = None) -> ProvenanceEntry:
        entry = ProvenanceEntry(record_id, record_type, bundle_hash, parent_ids, metadata)
        self._store[record_id] = entry
        logger.info("provenance_registered", record_id=record_id, type=record_type)
        return entry

    def promote(self, record_id: str, approved_by: str = "") -> bool:
        entry = self._store.get(record_id)
        if not entry:
            return False
        entry.promotion_status = PromotionStatus.PROMOTED
        entry.promoted_at = datetime.now(timezone.utc)
        entry.metadata["approved_by"] = approved_by
        logger.info("provenance_promoted", record_id=record_id)
        return True

    def rollback(self, record_id: str, reason: str = "") -> bool:
        entry = self._store.get(record_id)
        if not entry:
            return False
        entry.promotion_status = PromotionStatus.ROLLED_BACK
        entry.rolled_back_at = datetime.now(timezone.utc)
        entry.metadata["rollback_reason"] = reason
        logger.info("provenance_rolled_back", record_id=record_id)
        return True

    def get(self, record_id: str) -> Optional[ProvenanceEntry]:
        return self._store.get(record_id)

    def get_lineage(self, record_id: str) -> list[ProvenanceEntry]:
        """Trace full lineage back to source."""
        lineage = []
        visited = set()
        def _walk(rid):
            if rid in visited:
                return
            visited.add(rid)
            entry = self._store.get(rid)
            if entry:
                lineage.append(entry)
                for parent in entry.parent_ids:
                    _walk(parent)
        _walk(record_id)
        return lineage
