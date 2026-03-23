"""
BrahmX SDG — Core Data Models / Contracts / Schemas

All canonical schema definitions for the pipeline. These are the source of truth
for inter-service contracts, artifact formats, and storage structures.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


# ── Enumerations ──────────────────────────────────────────────────────────────


class ReliabilityTier(str, Enum):
    GOLD = "gold"
    REVIEWED = "reviewed"
    DRAFT = "draft"


class ClaimType(str, Enum):
    DEFINITION = "definition"
    EQUATION = "equation"
    POLICY = "policy"
    THRESHOLD = "threshold"
    PROCEDURE = "procedure"
    FACT = "fact"


class PromotionStatus(str, Enum):
    PENDING = "pending"
    PROMOTED = "promoted"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"


class BundleTier(str, Enum):
    GOLD = "gold"
    SILVER = "silver"
    BRONZE = "bronze"


class TrainingStage(str, Enum):
    CPT = "cpt"
    SFT_GOLD = "sft_gold"
    SFT_SILVER = "sft_silver"
    PREFERENCE = "preference"
    RL_ALIGNMENT = "rl_alignment"
    DISTILL_RECOVERY = "distill_recovery"
    LONG_CONTEXT_RECOVERY = "long_context_recovery"


class InferenceRuntime(str, Enum):
    VLLM_TPU = "vllm_tpu"
    VLLM_GPU = "vllm_gpu"
    JETSTREAM_MAXTEXT = "jetstream_maxtext"
    HF_TRANSFORMERS = "hf_transformers"


class LaneName(str, Enum):
    SIMULATION_JSON = "simulation_json"
    BOUNDED_DIALOGUE = "bounded_dialogue"
    LATEX_ARTIFACT = "latex_artifact"
    CODE_TOOL_USE = "code_tool_use"
    MULTILINGUAL = "multilingual"
    REASONING_BUDGET = "reasoning_budget"
    CURRICULUM_DIFFICULTY = "curriculum_difficulty"


# ── Source Plane Models ───────────────────────────────────────────────────────


class SourceDocument(BaseModel):
    """A single ingested source document with metadata."""
    doc_id: str = Field(default_factory=lambda: f"DOC-{uuid4().hex[:12]}")
    title: str
    source_type: str  # textbook, paper, report, code, simulation_json, etc.
    uri: str  # Original location
    license: str = "unknown"
    redistribution_allowed: bool = False
    language: str = "en"
    domain: str = ""
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    raw_hash: str = ""  # SHA-256 of raw content
    metadata: dict[str, Any] = Field(default_factory=dict)


class SourceAtom(BaseModel):
    """An atomic chunk of a source document, ready for evidence pack building."""
    atom_id: str = Field(default_factory=lambda: f"ATOM-{uuid4().hex[:12]}")
    doc_id: str
    text: str
    chunk_index: int = 0
    domain: str = ""
    license: str = "unknown"
    redistribution_allowed: bool = False
    quality_tags: list[str] = Field(default_factory=list)
    decontam_status: str = "unchecked"  # unchecked, clean, flagged
    token_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── Knowledge Base Models ─────────────────────────────────────────────────────


class CanonicalFact(BaseModel):
    """An atomic scientific claim in the Canonical Fact Table."""
    claim_id: str  # CFT-DOMAIN-NNNN
    statement: str
    claim_type: ClaimType
    domain: str
    valid_from: Optional[str] = None
    valid_to: Optional[str] = None
    supporting_sources: list[str] = Field(default_factory=list)
    reliability_tier: ReliabilityTier = ReliabilityTier.DRAFT
    license: str = "unknown"
    redistribution_allowed: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def fingerprint(self) -> str:
        normalized = " ".join(self.statement.lower().split())
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]


class EvidencePack(BaseModel):
    """Task-local evidence bundle consumed by the Prompt Constructor."""
    section_id: str
    top_chunks: list[dict[str, Any]] = Field(default_factory=list)
    canonical_claims: list[dict[str, Any]] = Field(default_factory=list)
    glossary_terms: list[dict[str, Any]] = Field(default_factory=list)
    known_constraints: list[dict[str, Any]] = Field(default_factory=list)
    retrieval_confidence: float = 0.0
    missing_required_claims: list[str] = Field(default_factory=list)
    cft_snapshot_hash: str = ""
    kb_path: str = ""
    built_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ── Generation Models ─────────────────────────────────────────────────────────


class TaskSpec(BaseModel):
    """Specification for a single generation task."""
    task_id: str = Field(default_factory=lambda: f"TASK-{uuid4().hex[:12]}")
    section_id: str
    task_type: str  # gold_qa, gold_explanation, silver_paraphrase, lane_simulation, etc.
    domain: str = ""
    language: str = "en"
    difficulty: str = "medium"  # easy, medium, hard, expert
    objectives: list[str] = Field(default_factory=list)
    required_claims: list[dict[str, str]] = Field(default_factory=list)
    lane: Optional[LaneName] = None
    constraints: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class PromptSpec(BaseModel):
    """A fully-assembled, deterministic prompt ready for teacher generation."""
    prompt_id: str = Field(default_factory=lambda: f"PROMPT-{uuid4().hex[:12]}")
    task_id: str
    system_prompt: str
    user_prompt: str
    must_cover: list[str] = Field(default_factory=list)
    must_not_contradict: list[str] = Field(default_factory=list)
    format_requirements: dict[str, Any] = Field(default_factory=dict)
    evidence_pack_hash: str = ""
    prompt_hash: str = ""  # SHA-256 for determinism verification


class TeacherCandidate(BaseModel):
    """A single teacher-generated candidate response."""
    candidate_id: str = Field(default_factory=lambda: f"CAND-{uuid4().hex[:12]}")
    task_id: str
    prompt_id: str
    teacher_model: str
    teacher_runtime: InferenceRuntime
    content: str
    structured_output: Optional[dict[str, Any]] = None
    claim_ledger: list[dict[str, Any]] = Field(default_factory=list)
    tool_checks_required: list[dict[str, Any]] = Field(default_factory=list)
    generation_metadata: dict[str, Any] = Field(default_factory=dict)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ── Verification Models ───────────────────────────────────────────────────────


class Claim(BaseModel):
    """A single claim extracted from a teacher candidate."""
    claim_id: str
    claim_text: str
    claim_type: str = "fact"
    verifiability: str = "KB"  # KB, assumption, self-evident
    supporting_citations: list[str] = Field(default_factory=list)
    tool_check_ref: Optional[str] = None
    is_critical: bool = False


class ClaimLedger(BaseModel):
    """Complete claim ledger for a candidate response."""
    candidate_id: str
    claims: list[Claim] = Field(default_factory=list)
    total_claims: int = 0
    critical_claims: int = 0
    unsupported_claims: int = 0
    orphan_claims: list[str] = Field(default_factory=list)


class DeanScore(BaseModel):
    """Dean scoring report for a single candidate."""
    candidate_id: str
    composite_score: float = 0.0
    gate_results: dict[str, Any] = Field(default_factory=dict)
    verdict: str = "FAIL"  # PASS, PASS_WITH_EDITS, FAIL
    failure_reasons: list[str] = Field(default_factory=list)
    repair_targets: list[str] = Field(default_factory=list)
    tool_outputs: list[dict[str, Any]] = Field(default_factory=list)


class AuditorReport(BaseModel):
    """Independent auditor review report."""
    candidate_id: str
    auditor_model: str
    status: str = "FAIL"  # PASS, FAIL, ESCALATE
    findings: list[str] = Field(default_factory=list)
    severity: str = "low"  # low, medium, high, critical
    override_dean: bool = False
    escalate_to_human: bool = False
    reviewed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ValidationReport(BaseModel):
    """Aggregated validation report for a candidate."""
    candidate_id: str
    dean_score: Optional[DeanScore] = None
    auditor_report: Optional[AuditorReport] = None
    citation_metrics: dict[str, float] = Field(default_factory=dict)
    symbolic_checks: list[dict[str, Any]] = Field(default_factory=list)
    code_exec_results: list[dict[str, Any]] = Field(default_factory=list)
    latex_compile_results: list[dict[str, Any]] = Field(default_factory=list)
    overall_verdict: str = "FAIL"
    human_review_required: bool = False


# ── Bundle Models ─────────────────────────────────────────────────────────────


class GoldRecordBundle(BaseModel):
    """The atomic unit of trust. Contains everything needed to audit a gold sample."""
    record_id: str = Field(default_factory=lambda: f"GOLD-{uuid4().hex[:12]}")
    task_spec: TaskSpec
    evidence_pack_hash: str
    prompt_spec_hash: str
    candidates: list[TeacherCandidate] = Field(default_factory=list)
    selected_candidate_id: str = ""
    claim_ledger: ClaimLedger
    dean_score: DeanScore
    auditor_report: AuditorReport
    validation_report: ValidationReport
    human_approved: bool = False
    human_reviewer: Optional[str] = None
    tier: BundleTier = BundleTier.GOLD
    promotion_status: PromotionStatus = PromotionStatus.PENDING
    provenance: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    bundle_hash: str = ""  # SHA-256 of serialized bundle

    def compute_hash(self) -> str:
        content = self.model_dump_json(exclude={"bundle_hash"})
        return hashlib.sha256(content.encode()).hexdigest()


class SilverBundle(BaseModel):
    """Validator-backed breadth data with source lineage."""
    record_id: str = Field(default_factory=lambda: f"SILVER-{uuid4().hex[:12]}")
    source_gold_refs: list[str] = Field(default_factory=list)
    source_atoms: list[str] = Field(default_factory=list)
    lane: Optional[LaneName] = None
    content: str = ""
    structured_output: Optional[dict[str, Any]] = None
    validator_results: list[dict[str, Any]] = Field(default_factory=list)
    validator_pass: bool = False
    tokenizer_safe: bool = False
    tier: BundleTier = BundleTier.SILVER
    promotion_status: PromotionStatus = PromotionStatus.PENDING
    provenance: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ── Corpus & Release Models ───────────────────────────────────────────────────


class DatasetSlice(BaseModel):
    """A single training example in a specific task format."""
    slice_id: str
    task_type: str  # explanation_generation, qa_with_citation, tool_use_grounding, etc.
    input_data: dict[str, Any]
    target_data: dict[str, Any]
    source_bundle_id: str
    tier: BundleTier
    metadata: dict[str, Any] = Field(default_factory=dict)


class CorpusVersion(BaseModel):
    """A versioned, immutable corpus ready for training consumption."""
    corpus_id: str = Field(default_factory=lambda: f"CORPUS-{uuid4().hex[:8]}")
    version: str
    stage: TrainingStage
    format: str = "arrayrecord"  # arrayrecord, parquet
    gcs_path: str = ""
    gold_bundle_count: int = 0
    silver_bundle_count: int = 0
    total_examples: int = 0
    mixture_ratios: dict[str, float] = Field(default_factory=dict)
    dedup_stats: dict[str, int] = Field(default_factory=dict)
    decontam_stats: dict[str, int] = Field(default_factory=dict)
    checksum: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ReleaseManifest(BaseModel):
    """Immutable release record tying corpora to training."""
    manifest_id: str = Field(default_factory=lambda: f"REL-{uuid4().hex[:8]}")
    version: str
    corpora: list[CorpusVersion] = Field(default_factory=list)
    gold_bundles_included: list[str] = Field(default_factory=list)
    silver_bundles_included: list[str] = Field(default_factory=list)
    approval_status: str = "pending"  # pending, approved, rejected
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    notes: str = ""
    rollback_ref: Optional[str] = None  # Previous manifest ID for rollback
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ── Training & Eval Models ────────────────────────────────────────────────────


class TrainingRunSpec(BaseModel):
    """Complete specification for a single training run."""
    run_id: str = Field(default_factory=lambda: f"RUN-{uuid4().hex[:8]}")
    stage: TrainingStage
    student_model: str  # brahmx-sci-3.5b, brahmx-sci-8b, etc.
    corpus_version: str
    release_manifest_id: str
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    maxtext_config: dict[str, Any] = Field(default_factory=dict)
    tpu_topology: str = "v5e-256"
    num_steps: int = 0
    checkpoint_interval: int = 1000
    eval_interval: int = 500
    base_checkpoint: Optional[str] = None  # For fine-tuning / recovery stages
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class EvalReport(BaseModel):
    """Evaluation results for a checkpoint."""
    eval_id: str = Field(default_factory=lambda: f"EVAL-{uuid4().hex[:8]}")
    run_id: str
    checkpoint_step: int
    checkpoint_path: str
    metrics: dict[str, float] = Field(default_factory=dict)
    benchmark_results: dict[str, dict[str, float]] = Field(default_factory=dict)
    scientific_factuality: Optional[dict[str, float]] = None
    tool_use_accuracy: Optional[dict[str, float]] = None
    multilingual_scores: Optional[dict[str, float]] = None
    long_context_scores: Optional[dict[str, float]] = None
    pass_criteria_met: bool = False
    notes: str = ""
    evaluated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ModelRelease(BaseModel):
    """A released student model package."""
    release_id: str = Field(default_factory=lambda: f"MODEL-{uuid4().hex[:8]}")
    model_name: str
    version: str
    checkpoint_path: str
    tokenizer_path: str
    serving_config: dict[str, Any] = Field(default_factory=dict)
    eval_report_id: str
    training_run_id: str
    release_manifest_id: str
    approved_by: str = ""
    approved_at: Optional[datetime] = None
    rollback_ready: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
