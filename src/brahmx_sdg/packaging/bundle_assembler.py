"""Bundle Assembler — assembles Gold/Silver Record Bundles."""
from __future__ import annotations
from typing import Optional
from brahmx_sdg.schemas import (
    AuditorReport, ClaimLedger, DeanScore, EvidencePack,
    GoldRecordBundle, PromptSpec, TaskSpec, TeacherCandidate, ValidationReport,
)


class BundleAssembler:
    def assemble_gold(
        self,
        task_spec: TaskSpec,
        evidence_pack: Optional[EvidencePack],
        candidates: list[TeacherCandidate],
        selected: Optional[TeacherCandidate],
        dean_scores: list[DeanScore],
        auditor_report: Optional[AuditorReport],
        human_approved: bool = False,
    ) -> GoldRecordBundle:
        best_dean = max(dean_scores, key=lambda d: d.composite_score) if dean_scores else DeanScore(candidate_id="")
        bundle = GoldRecordBundle(
            task_spec=task_spec,
            evidence_pack_hash=evidence_pack.cft_snapshot_hash if evidence_pack else "",
            prompt_spec_hash="",
            candidates=candidates,
            selected_candidate_id=selected.candidate_id if selected else "",
            claim_ledger=ClaimLedger(candidate_id=selected.candidate_id if selected else ""),
            dean_score=best_dean,
            auditor_report=auditor_report or AuditorReport(candidate_id="", auditor_model=""),
            validation_report=ValidationReport(candidate_id=selected.candidate_id if selected else ""),
            human_approved=human_approved,
        )
        bundle.bundle_hash = bundle.compute_hash()
        return bundle
