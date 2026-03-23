"""
Gold Generator — Orchestrates the full gold generation path.

State machine:
  RETRIEVE_EVIDENCE → BUILD_PROMPT → GENERATE_CANDIDATES (Teacher A + B + C)
  → DEAN_SCORE_ALL → SELECT_BEST → AUDITOR_REVIEW
  → AUTO_PATCH (max 2 rounds, same teacher) → HUMAN_REVIEW_GATE → PUBLISH_BUNDLE
  → EMIT_SLICES → DONE

Key design choices aligned with the final architecture:
- Three independent teachers are called separately (not n=3 from one model).
- Teachers return structured JSON with embedded claim_ledger.
- Repairs go back to the same teacher that produced the failing candidate.
- Auditor receives NO evidence pack (FM-09 bias mitigation).
- Max 2 repair rounds enforced here; no round-3 teacher call is ever made.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import structlog

from brahmx_sdg.schemas import (
    AuditorReport, ClaimLedger, DeanScore, EvidencePack,
    GoldRecordBundle, InferenceRuntime, PromptSpec, TaskSpec,
    TeacherCandidate, ValidationReport,
)

logger = structlog.get_logger()

MAX_REPAIR_ROUNDS = 2
DEFAULT_ROUTING_CONFIG = "configs/routing/models.yaml"


class GoldState(Enum):
    RETRIEVE_EVIDENCE = auto()
    BUILD_PROMPT = auto()
    GENERATE_CANDIDATES = auto()
    DEAN_SCORE = auto()
    SELECT_CANDIDATE = auto()
    AUDITOR_REVIEW = auto()
    AUTO_PATCH = auto()
    HUMAN_REVIEW_GATE = auto()
    PUBLISH_BUNDLE = auto()
    EMIT_SLICES = auto()
    DONE = auto()
    BLOCKED = auto()


@dataclass
class GoldContext:
    task_spec: TaskSpec
    kb_path: Path
    output_dir: Path
    evidence_pack: Optional[EvidencePack] = None
    prompt_spec: Optional[PromptSpec] = None
    candidates: list[TeacherCandidate] = field(default_factory=list)
    dean_scores: list[DeanScore] = field(default_factory=list)
    selected_candidate: Optional[TeacherCandidate] = None
    auditor_report: Optional[AuditorReport] = None
    repair_rounds: int = 0
    # track which candidates failed so repair targets the right teacher
    failed_candidate: Optional[TeacherCandidate] = None
    failed_dean_score: Optional[DeanScore] = None
    human_approved: bool = False
    bundle: Optional[GoldRecordBundle] = None
    block_reason: str = ""
    state: GoldState = GoldState.RETRIEVE_EVIDENCE


@dataclass
class GoldResult:
    success: bool
    bundle_id: str = ""
    reason: str = ""
    output_path: str = ""


class GoldGenerator:
    """
    Orchestrates all stages from evidence retrieval to gold record publishing.
    Each stage is a pure function over GoldContext.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        routing_config: str = DEFAULT_ROUTING_CONFIG,
    ) -> None:
        self.config = config or {}
        self.routing_config = routing_config
        self._evidence_builder = None
        self._prompt_constructor = None
        self._teacher_router = None
        self._dean = None
        self._auditor = None
        self._bundle_assembler = None
        self._slice_emitter = None

    def run(
        self,
        spec_path: Path,
        kb_path: Path,
        output_dir: Path,
    ) -> GoldResult:
        """Run the full gold pipeline for a single task spec."""
        output_dir.mkdir(parents=True, exist_ok=True)
        task_spec = TaskSpec.model_validate_json(spec_path.read_text())
        ctx = GoldContext(task_spec=task_spec, kb_path=kb_path, output_dir=output_dir)

        handlers = {
            GoldState.RETRIEVE_EVIDENCE: self._retrieve_evidence,
            GoldState.BUILD_PROMPT: self._build_prompt,
            GoldState.GENERATE_CANDIDATES: self._generate_candidates,
            GoldState.DEAN_SCORE: self._dean_score,
            GoldState.SELECT_CANDIDATE: self._select_candidate,
            GoldState.AUDITOR_REVIEW: self._auditor_review,
            GoldState.AUTO_PATCH: self._auto_patch,
            GoldState.HUMAN_REVIEW_GATE: self._human_review_gate,
            GoldState.PUBLISH_BUNDLE: self._publish_bundle,
            GoldState.EMIT_SLICES: self._emit_slices,
        }

        while ctx.state not in (GoldState.DONE, GoldState.BLOCKED):
            handler = handlers.get(ctx.state)
            if handler is None:
                ctx.state = GoldState.BLOCKED
                break
            logger.info(
                "gold_state",
                task=ctx.task_spec.task_id,
                state=ctx.state.name,
                repair_round=ctx.repair_rounds,
            )
            ctx = handler(ctx)

        if ctx.state == GoldState.DONE and ctx.bundle:
            out = ctx.output_dir / f"{ctx.bundle.record_id}.json"
            return GoldResult(
                success=True,
                bundle_id=ctx.bundle.record_id,
                output_path=str(out),
            )
        return GoldResult(
            success=False,
            reason=ctx.block_reason or f"Blocked at {ctx.state.name}",
        )

    # ── Stage handlers ────────────────────────────────────────────────────────

    def _retrieve_evidence(self, ctx: GoldContext) -> GoldContext:
        from brahmx_sdg.kb.evidence_pack_builder import EvidencePackBuilder
        builder = self._evidence_builder or EvidencePackBuilder()
        pack = builder.build(ctx.task_spec.model_dump(), ctx.kb_path)
        ctx.evidence_pack = EvidencePack(**pack)
        if ctx.evidence_pack.missing_required_claims:
            logger.error("kb_gap", missing=ctx.evidence_pack.missing_required_claims)
            ctx.block_reason = f"KB gap: {ctx.evidence_pack.missing_required_claims}"
            ctx.state = GoldState.BLOCKED
            return ctx
        ctx.state = GoldState.BUILD_PROMPT
        return ctx

    def _build_prompt(self, ctx: GoldContext) -> GoldContext:
        from brahmx_sdg.prompt.constructor import PromptConstructor
        constructor = self._prompt_constructor or PromptConstructor()
        ctx.prompt_spec = constructor.build(ctx.task_spec, ctx.evidence_pack)
        ctx.state = GoldState.GENERATE_CANDIDATES
        return ctx

    def _generate_candidates(self, ctx: GoldContext) -> GoldContext:
        """Call Teacher A, B, and C separately; parse JSON response per candidate."""
        from brahmx_sdg.routing import TeacherRouter, ModelRole, WorkloadClass

        router = self._teacher_router or TeacherRouter.from_config(self.routing_config)

        messages = [
            {"role": "system", "content": ctx.prompt_spec.system_prompt},
            {"role": "user", "content": ctx.prompt_spec.user_prompt},
        ]

        ctx.candidates = []
        roles = [ModelRole.TEACHER_A, ModelRole.TEACHER_B, ModelRole.TEACHER_C]
        workloads = [WorkloadClass.BULK, WorkloadClass.FRONTIER, WorkloadClass.BULK]

        for role, workload in zip(roles, workloads):
            try:
                results = router.generate(
                    role=role,
                    messages=messages,
                    n=1,
                    max_tokens=4096,
                    workload_class=workload,
                    response_format={"type": "json_object"},
                )
                for r in results:
                    candidate = self._parse_candidate(r, ctx.task_spec, ctx.prompt_spec)
                    ctx.candidates.append(candidate)
                    logger.info(
                        "candidate_generated",
                        teacher=role.value,
                        candidate_id=candidate.candidate_id,
                        claim_count=len(candidate.claim_ledger),
                    )
            except Exception as e:
                logger.warning("teacher_skipped", role=role.value, error=str(e))

        if not ctx.candidates:
            ctx.block_reason = "All three teachers failed to generate a candidate."
            ctx.state = GoldState.BLOCKED
            return ctx

        ctx.state = GoldState.DEAN_SCORE
        return ctx

    def _dean_score(self, ctx: GoldContext) -> GoldContext:
        from brahmx_sdg.verification.dean import Dean
        dean = self._dean or Dean()
        ctx.dean_scores = []
        for candidate in ctx.candidates:
            score = dean.score(candidate, ctx.task_spec, ctx.evidence_pack)
            ctx.dean_scores.append(score)
            logger.info(
                "dean_scored",
                candidate=candidate.candidate_id,
                verdict=score.verdict,
                score=score.composite_score,
            )
        ctx.state = GoldState.SELECT_CANDIDATE
        return ctx

    def _select_candidate(self, ctx: GoldContext) -> GoldContext:
        passing = [
            (c, s) for c, s in zip(ctx.candidates, ctx.dean_scores)
            if s.verdict in ("PASS", "PASS_WITH_EDITS")
        ]

        if passing:
            best = max(passing, key=lambda t: t[1].composite_score)
            ctx.selected_candidate = best[0]
            ctx.state = GoldState.AUDITOR_REVIEW
        elif ctx.repair_rounds < MAX_REPAIR_ROUNDS:
            # Pick the best-scoring candidate for repair (even if FAIL)
            best_pair = max(
                zip(ctx.candidates, ctx.dean_scores),
                key=lambda t: t[1].composite_score,
            )
            ctx.failed_candidate = best_pair[0]
            ctx.failed_dean_score = best_pair[1]
            ctx.state = GoldState.AUTO_PATCH
        else:
            ctx.block_reason = (
                f"No passing candidate after {ctx.repair_rounds} repair rounds. "
                f"Best score: {max(s.composite_score for s in ctx.dean_scores):.3f}"
            )
            logger.error("no_passing_candidate", block_reason=ctx.block_reason)
            ctx.state = GoldState.BLOCKED

        return ctx

    def _auditor_review(self, ctx: GoldContext) -> GoldContext:
        from brahmx_sdg.verification.auditor import Auditor
        auditor = self._auditor or Auditor(routing_config=self.routing_config)
        ctx.auditor_report = auditor.review(ctx.selected_candidate, ctx.task_spec)
        logger.info(
            "auditor_reviewed",
            candidate=ctx.selected_candidate.candidate_id,
            status=ctx.auditor_report.status,
        )

        if ctx.auditor_report.status == "FAIL" and ctx.repair_rounds < MAX_REPAIR_ROUNDS:
            ctx.failed_candidate = ctx.selected_candidate
            ctx.failed_dean_score = None
            ctx.state = GoldState.AUTO_PATCH
        elif ctx.auditor_report.status == "FAIL":
            ctx.block_reason = f"Auditor FAIL after {ctx.repair_rounds} repair rounds."
            ctx.state = GoldState.BLOCKED
        else:
            ctx.state = GoldState.HUMAN_REVIEW_GATE
        return ctx

    def _auto_patch(self, ctx: GoldContext) -> GoldContext:
        """Targeted repair: call the same teacher that produced the failing candidate."""
        ctx.repair_rounds += 1
        logger.info("auto_patch_start", round=ctx.repair_rounds)

        from brahmx_sdg.routing import TeacherRouter, ModelRole, WorkloadClass

        router = self._teacher_router or TeacherRouter.from_config(self.routing_config)

        # Determine which teacher to call based on the failing candidate's generation metadata
        failed = ctx.failed_candidate
        teacher_role_str = (failed.generation_metadata.get("role", "teacher_a") if failed else "teacher_a")
        try:
            repair_role = ModelRole(teacher_role_str)
        except ValueError:
            repair_role = ModelRole.TEACHER_A

        # Build a repair prompt that includes the failure feedback
        failure_summary = self._build_failure_summary(ctx)
        repair_messages = [
            {"role": "system", "content": ctx.prompt_spec.system_prompt},
            {"role": "user", "content": ctx.prompt_spec.user_prompt},
            {
                "role": "assistant",
                "content": (failed.content if failed else "(no previous content)"),
            },
            {
                "role": "user",
                "content": (
                    f"The previous response was rejected. Please fix the following issues "
                    f"and return an improved JSON response:\n\n{failure_summary}"
                ),
            },
        ]

        try:
            results = router.generate(
                role=repair_role,
                messages=repair_messages,
                n=1,
                max_tokens=4096,
                workload_class=WorkloadClass.BULK,
                response_format={"type": "json_object"},
            )
            for r in results:
                patched = self._parse_candidate(r, ctx.task_spec, ctx.prompt_spec)
                patched.generation_metadata["repair_round"] = ctx.repair_rounds
                # Replace the failing candidate in the list
                ctx.candidates = [
                    patched if c.candidate_id == (failed.candidate_id if failed else "") else c
                    for c in ctx.candidates
                ]
                if patched.candidate_id not in [c.candidate_id for c in ctx.candidates]:
                    ctx.candidates.append(patched)
                logger.info(
                    "repair_generated",
                    teacher=repair_role.value,
                    round=ctx.repair_rounds,
                    candidate_id=patched.candidate_id,
                )
        except Exception as e:
            logger.error("repair_failed", round=ctx.repair_rounds, error=str(e))

        ctx.state = GoldState.DEAN_SCORE
        return ctx

    def _human_review_gate(self, ctx: GoldContext) -> GoldContext:
        """
        In automated mode: log and proceed. In production, this would block
        until a human reviewer approves via the review UI.
        """
        logger.info(
            "human_review_gate",
            task=ctx.task_spec.task_id,
            candidate=ctx.selected_candidate.candidate_id if ctx.selected_candidate else "none",
            note="auto-approving in non-interactive mode",
        )
        ctx.state = GoldState.PUBLISH_BUNDLE
        return ctx

    def _publish_bundle(self, ctx: GoldContext) -> GoldContext:
        from brahmx_sdg.packaging.bundle_assembler import BundleAssembler
        assembler = self._bundle_assembler or BundleAssembler()
        ctx.bundle = assembler.assemble_gold(
            task_spec=ctx.task_spec,
            evidence_pack=ctx.evidence_pack,
            candidates=ctx.candidates,
            selected=ctx.selected_candidate,
            dean_scores=ctx.dean_scores,
            auditor_report=ctx.auditor_report,
            human_approved=ctx.human_approved,
        )
        output_path = ctx.output_dir / f"{ctx.bundle.record_id}.json"
        output_path.write_text(ctx.bundle.model_dump_json(indent=2))
        logger.info(
            "bundle_published",
            bundle_id=ctx.bundle.record_id,
            path=str(output_path),
        )
        ctx.state = GoldState.EMIT_SLICES
        return ctx

    def _emit_slices(self, ctx: GoldContext) -> GoldContext:
        from brahmx_sdg.packaging.slice_emitter import SliceEmitter
        emitter = self._slice_emitter or SliceEmitter()
        if ctx.bundle:
            slices = emitter.emit(ctx.bundle)
            slices_dir = ctx.output_dir / "slices"
            slices_dir.mkdir(exist_ok=True)
            for sl in slices:
                path = slices_dir / f"{sl.slice_id}.json"
                path.write_text(json.dumps(sl.model_dump(), indent=2, ensure_ascii=False))
            logger.info(
                "slices_emitted",
                count=len(slices),
                bundle=ctx.bundle.record_id,
                dir=str(slices_dir),
            )
        ctx.state = GoldState.DONE
        return ctx

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _parse_candidate(
        self,
        result: dict[str, Any],
        task_spec: TaskSpec,
        prompt_spec: PromptSpec,
    ) -> TeacherCandidate:
        """
        Parse a raw LLM result dict into a TeacherCandidate.
        Extracts content and claim_ledger from JSON response.
        Falls back gracefully if JSON is malformed.
        """
        raw = result["content"]

        # Try JSON parse first
        content = raw
        claim_ledger_raw: list[dict[str, Any]] = []
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                content = parsed.get("content", raw)
                claim_ledger_raw = parsed.get("claim_ledger", [])
                if not isinstance(claim_ledger_raw, list):
                    claim_ledger_raw = []
        except (json.JSONDecodeError, ValueError):
            # Fallback: extract from fenced JSON block if present
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group(1))
                    content = parsed.get("content", raw)
                    claim_ledger_raw = parsed.get("claim_ledger", [])
                except (json.JSONDecodeError, ValueError):
                    pass

        # Normalize claim ledger entries
        claim_ledger = [
            {
                "claim_id": c.get("claim_id", f"CLM-{uuid4().hex[:6].upper()}"),
                "claim_text": str(c.get("claim_text", "")),
                "claim_type": c.get("claim_type", "fact"),
                "verifiability": c.get("verifiability", "KB"),
                "supporting_citations": c.get("supporting_citations", []),
                "is_critical": bool(c.get("is_critical", False)),
            }
            for c in claim_ledger_raw
            if isinstance(c, dict) and c.get("claim_text")
        ]

        return TeacherCandidate(
            task_id=task_spec.task_id,
            prompt_id=prompt_spec.prompt_id,
            teacher_model=result.get("model", "unknown"),
            teacher_runtime=InferenceRuntime(result.get("runtime", "vllm_gpu")),
            content=content,
            claim_ledger=claim_ledger,
            generation_metadata={
                "role": result.get("role", "teacher_a"),
                "finish_reason": result.get("finish_reason", ""),
                "model_name": result.get("model_name", ""),
            },
        )

    def _build_failure_summary(self, ctx: GoldContext) -> str:
        lines = []
        if ctx.failed_dean_score:
            lines.append("Dean verification failures:")
            for reason in ctx.failed_dean_score.failure_reasons:
                lines.append(f"  - {reason}")
            if ctx.failed_dean_score.repair_targets:
                lines.append("Suggested repairs:")
                for target in ctx.failed_dean_score.repair_targets:
                    lines.append(f"  - {target}")
        if ctx.auditor_report and ctx.auditor_report.status == "FAIL":
            lines.append("Auditor findings:")
            for finding in ctx.auditor_report.findings:
                lines.append(f"  - {finding}")
        return "\n".join(lines) if lines else "General quality issues — please improve accuracy and citation coverage."

    @classmethod
    def from_config(cls, config_path: Path) -> "GoldGenerator":
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}
        return cls(config=config)
