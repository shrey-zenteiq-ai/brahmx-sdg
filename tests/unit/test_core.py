"""
Tests for BrahmX SDG core components.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ── Schema Tests ──────────────────────────────────────────────────────────────


class TestSchemas:
    """Test Pydantic schema validation and serialization."""

    def test_source_document_creation(self):
        from brahmx_sdg.schemas import SourceDocument
        doc = SourceDocument(title="Test Paper", source_type="pdf", uri="/test.pdf")
        assert doc.doc_id.startswith("DOC-")
        assert doc.title == "Test Paper"
        assert doc.redistribution_allowed is False

    def test_source_document_serialization(self):
        from brahmx_sdg.schemas import SourceDocument
        doc = SourceDocument(title="Test", source_type="pdf", uri="/t.pdf")
        data = doc.model_dump()
        assert "doc_id" in data
        assert data["title"] == "Test"
        roundtrip = SourceDocument.model_validate(data)
        assert roundtrip.title == doc.title

    def test_task_spec_defaults(self):
        from brahmx_sdg.schemas import TaskSpec
        spec = TaskSpec(section_id="SEC-001", task_type="gold_qa")
        assert spec.task_id.startswith("TASK-")
        assert spec.difficulty == "medium"
        assert spec.language == "en"

    def test_gold_record_bundle_hash(self):
        from brahmx_sdg.schemas import (
            GoldRecordBundle, TaskSpec, ClaimLedger,
            DeanScore, AuditorReport, ValidationReport,
        )
        spec = TaskSpec(section_id="SEC-001", task_type="gold_qa")
        bundle = GoldRecordBundle(
            task_spec=spec,
            evidence_pack_hash="abc123",
            prompt_spec_hash="def456",
            claim_ledger=ClaimLedger(candidate_id="C1"),
            dean_score=DeanScore(candidate_id="C1"),
            auditor_report=AuditorReport(candidate_id="C1", auditor_model="test"),
            validation_report=ValidationReport(candidate_id="C1"),
        )
        h1 = bundle.compute_hash()
        assert len(h1) == 64  # SHA-256 hex
        h2 = bundle.compute_hash()
        assert h1 == h2  # Deterministic

    def test_training_run_spec(self):
        from brahmx_sdg.schemas import TrainingRunSpec, TrainingStage
        spec = TrainingRunSpec(
            stage=TrainingStage.SFT_GOLD,
            student_model="brahmx-sci-8b",
            corpus_version="v1.0",
            release_manifest_id="REL-abc",
            num_steps=10000,
        )
        assert spec.tpu_topology == "v5e-256"

    def test_lane_name_enum(self):
        from brahmx_sdg.schemas import LaneName
        assert LaneName.SIMULATION_JSON.value == "simulation_json"
        assert LaneName.LATEX_ARTIFACT.value == "latex_artifact"

    def test_evidence_pack_model(self):
        from brahmx_sdg.schemas import EvidencePack
        pack = EvidencePack(section_id="SEC-001")
        assert pack.retrieval_confidence == 0.0
        assert pack.missing_required_claims == []


# ── Routing Tests ─────────────────────────────────────────────────────────────


class TestRouting:
    """Test teacher routing logic."""

    def test_model_registry(self):
        from brahmx_sdg.routing import ModelRegistry, ModelEndpoint, ModelRole, WorkloadClass, InferenceRuntime
        registry = ModelRegistry()
        endpoint = ModelEndpoint(
            model_id="test-model",
            model_name="Test Model",
            runtime=InferenceRuntime.VLLM_TPU,
            base_url="http://localhost:8000",
            roles=[ModelRole.TEACHER_A],
            workload_classes=[WorkloadClass.BULK],
            validated_on_tpu=True,
            quality_score=0.9,
        )
        registry.register(endpoint)
        assert registry.get("test-model") is not None
        assert len(registry.get_by_role(ModelRole.TEACHER_A)) == 1

    def test_default_routing_prefers_tpu_for_bulk(self):
        from brahmx_sdg.routing import (
            DefaultRoutingStrategy, ModelEndpoint, ModelRole,
            WorkloadClass, InferenceRuntime,
        )
        strategy = DefaultRoutingStrategy()
        tpu = ModelEndpoint(
            model_id="tpu-model", model_name="TPU", runtime=InferenceRuntime.VLLM_TPU,
            base_url="http://tpu:8000", roles=[ModelRole.TEACHER_A],
            workload_classes=[WorkloadClass.BULK], quality_score=0.9,
        )
        gpu = ModelEndpoint(
            model_id="gpu-model", model_name="GPU", runtime=InferenceRuntime.VLLM_GPU,
            base_url="http://gpu:8000", roles=[ModelRole.TEACHER_A],
            workload_classes=[WorkloadClass.BULK], quality_score=0.9,
        )
        decision = strategy.select([tpu, gpu], {"workload_class": "bulk"})
        assert decision.endpoint.runtime == InferenceRuntime.VLLM_TPU

    def test_default_routing_prefers_gpu_for_frontier(self):
        from brahmx_sdg.routing import (
            DefaultRoutingStrategy, ModelEndpoint, ModelRole,
            WorkloadClass, InferenceRuntime,
        )
        strategy = DefaultRoutingStrategy()
        tpu = ModelEndpoint(
            model_id="tpu-model", model_name="TPU", runtime=InferenceRuntime.VLLM_TPU,
            base_url="http://tpu:8000", roles=[ModelRole.TEACHER_B],
            workload_classes=[WorkloadClass.FRONTIER], quality_score=0.9,
        )
        gpu = ModelEndpoint(
            model_id="gpu-model", model_name="GPU", runtime=InferenceRuntime.VLLM_GPU,
            base_url="http://gpu:8000", roles=[ModelRole.TEACHER_B],
            workload_classes=[WorkloadClass.FRONTIER], quality_score=0.95,
        )
        decision = strategy.select([tpu, gpu], {"workload_class": "frontier"})
        assert decision.endpoint.runtime == InferenceRuntime.VLLM_GPU

    def test_admission_controller(self):
        from brahmx_sdg.routing import AdmissionController, WorkloadClass
        ac = AdmissionController(bulk_concurrency=2)
        assert ac.acquire(WorkloadClass.BULK) is True
        assert ac.acquire(WorkloadClass.BULK) is True
        assert ac.acquire(WorkloadClass.BULK) is False  # At capacity
        ac.release(WorkloadClass.BULK)
        assert ac.acquire(WorkloadClass.BULK) is True


# ── Verification Tests ────────────────────────────────────────────────────────


class TestVerification:
    """Test verification components."""

    def test_symbolic_numeric_validator_symbolic(self):
        from brahmx_sdg.verification.symbolic_numeric_validator import SymbolicNumericValidator
        v = SymbolicNumericValidator()
        result = v._check_symbolic({
            "check_id": "TC-01",
            "expression": "x**2 + 2*x + 1",
            "expected": "(x + 1)**2",
        })
        assert result.passed is True

    def test_symbolic_numeric_validator_range(self):
        from brahmx_sdg.verification.symbolic_numeric_validator import SymbolicNumericValidator
        v = SymbolicNumericValidator()
        result = v._check_range({
            "check_id": "TC-02",
            "value": 5.0,
            "min": 0.0,
            "max": 10.0,
        })
        assert result.passed is True

    def test_symbolic_numeric_validator_range_fail(self):
        from brahmx_sdg.verification.symbolic_numeric_validator import SymbolicNumericValidator
        v = SymbolicNumericValidator()
        result = v._check_range({
            "check_id": "TC-03",
            "value": 15.0,
            "min": 0.0,
            "max": 10.0,
        })
        assert result.passed is False

    def test_canonicalizer_basic(self):
        from brahmx_sdg.packaging.canonicalizer import Canonicalizer
        c = Canonicalizer()
        result = c.canonicalize("Hello world")
        assert not result.rejected
        assert result.text == "Hello world"

    def test_canonicalizer_strips_control_chars(self):
        from brahmx_sdg.packaging.canonicalizer import Canonicalizer
        c = Canonicalizer()
        result = c.canonicalize("Hello\x00world\x07test")
        assert "\x00" not in result.text
        assert "\x07" not in result.text


# ── Common Tests ──────────────────────────────────────────────────────────────


class TestCommon:
    """Test common utilities."""

    def test_deterministic_hash(self):
        from brahmx_sdg.common import deterministic_hash
        h1 = deterministic_hash("hello world")
        h2 = deterministic_hash("  hello   world  ")
        assert h1 == h2  # Normalized

    def test_local_artifact_store(self, tmp_path):
        from brahmx_sdg.common import LocalArtifactStore
        store = LocalArtifactStore(tmp_path)
        store.put("test/file.txt", b"hello")
        assert store.exists("test/file.txt")
        data, _ = store.get("test/file.txt")
        assert data == b"hello"

    def test_local_artifact_store_json(self, tmp_path):
        from brahmx_sdg.common import LocalArtifactStore
        store = LocalArtifactStore(tmp_path)
        store.put_json("test/data.json", {"key": "value"})
        result = store.get_json("test/data.json")
        assert result["key"] == "value"


# ── Governance Tests ──────────────────────────────────────────────────────────


class TestProvenance:
    """Test provenance registry."""

    def test_register_and_get(self):
        from brahmx_sdg.governance.provenance_registry import ProvenanceRegistry
        registry = ProvenanceRegistry()
        entry = registry.register("GOLD-001", "gold_bundle", "hash123")
        assert entry.record_id == "GOLD-001"
        assert registry.get("GOLD-001") is not None

    def test_promote(self):
        from brahmx_sdg.governance.provenance_registry import ProvenanceRegistry
        from brahmx_sdg.schemas import PromotionStatus
        registry = ProvenanceRegistry()
        registry.register("GOLD-001", "gold_bundle", "hash123")
        assert registry.promote("GOLD-001", "reviewer@test.com")
        entry = registry.get("GOLD-001")
        assert entry.promotion_status == PromotionStatus.PROMOTED

    def test_rollback(self):
        from brahmx_sdg.governance.provenance_registry import ProvenanceRegistry
        from brahmx_sdg.schemas import PromotionStatus
        registry = ProvenanceRegistry()
        registry.register("GOLD-001", "gold_bundle", "hash123")
        registry.promote("GOLD-001")
        registry.rollback("GOLD-001", reason="quality regression")
        entry = registry.get("GOLD-001")
        assert entry.promotion_status == PromotionStatus.ROLLED_BACK

    def test_lineage_tracing(self):
        from brahmx_sdg.governance.provenance_registry import ProvenanceRegistry
        registry = ProvenanceRegistry()
        registry.register("SRC-001", "source", "h1")
        registry.register("GOLD-001", "gold_bundle", "h2", parent_ids=["SRC-001"])
        registry.register("SILVER-001", "silver_bundle", "h3", parent_ids=["GOLD-001"])
        lineage = registry.get_lineage("SILVER-001")
        assert len(lineage) == 3
