"""
Dean — Primary Verifier.

Verification chain (in order):
  1. Structural: JSON-parseable, claim ledger present, no CJK leakage
  2. Citation: BM25/TF-IDF coverage + precision vs evidence pack
  3. Symbolic: SymPy/Pint checks on any embedded tool_checks
  4. LLM rubric: pedagogy, coverage, difficulty calibration via Dean model
  5. Composite score → PASS / PASS_WITH_EDITS / FAIL

Thresholds read from configs/gates/gold_gates.yaml.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

import yaml
import structlog

from brahmx_sdg.schemas import DeanScore, EvidencePack, TaskSpec, TeacherCandidate

logger = structlog.get_logger()

_CJK_PATTERN = re.compile(r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]")
DEFAULT_CONFIG = Path("configs/gates/gold_gates.yaml")
DEFAULT_ROUTING_CONFIG = "configs/routing/models.yaml"


class Dean:
    """Primary verifier. Scores a Teacher candidate against gold gate thresholds."""

    def __init__(
        self,
        config_path: Path = DEFAULT_CONFIG,
        routing_config: str = DEFAULT_ROUTING_CONFIG,
        use_llm_scoring: bool = True,
    ) -> None:
        self.routing_config = routing_config
        self.use_llm_scoring = use_llm_scoring
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
            self.gates = config.get("tiers", {}).get("gold", {}).get("gates", {})
        except FileNotFoundError:
            self.gates = {}

        # Thresholds (with defaults matching gold_gates.yaml)
        self._citation_coverage_min = self._gate_value("citation_coverage", 0.98)
        self._citation_precision_min = self._gate_value("citation_precision", 0.95)

    def _gate_value(self, key: str, default: float) -> float:
        g = self.gates.get(key, {})
        if isinstance(g, dict):
            return float(g.get("value", default))
        return default

    def score(
        self,
        candidate: TeacherCandidate,
        task_spec: TaskSpec,
        evidence_pack: Optional[EvidencePack] = None,
    ) -> DeanScore:
        gate_results: dict[str, Any] = {}
        failures: list[str] = []
        repair_targets: list[str] = []

        # 1. CJK leakage
        leaked = _CJK_PATTERN.findall(candidate.content)
        gate_results["cjk_leakage"] = bool(leaked)
        if leaked:
            failures.append(f"cjk_leakage: {len(leaked)} characters found")
            repair_targets.append("Remove CJK characters from content")

        # 2. Claim ledger presence
        has_ledger = bool(candidate.claim_ledger)
        gate_results["claim_ledger_present"] = has_ledger
        if not has_ledger:
            failures.append("claim_ledger_missing")
            repair_targets.append("Add a Claim Ledger with at least one cited claim")
            return DeanScore(
                candidate_id=candidate.candidate_id,
                composite_score=0.0,
                gate_results=gate_results,
                verdict="FAIL",
                failure_reasons=failures,
                repair_targets=repair_targets,
            )

        # 3. Citation check (BM25/TF-IDF lexical)
        if evidence_pack and evidence_pack.top_chunks:
            citation_metrics = self._check_citations(candidate, evidence_pack)
            gate_results.update(citation_metrics)

            cov = citation_metrics.get("citation_coverage", 0.0)
            prec = citation_metrics.get("citation_precision", 0.0)

            if cov < self._citation_coverage_min:
                failures.append(
                    f"citation_coverage {cov:.2f} < {self._citation_coverage_min:.2f}"
                )
                repair_targets.append(
                    f"Add citations to more claims (need coverage >= {self._citation_coverage_min})"
                )
            if prec < self._citation_precision_min:
                failures.append(
                    f"citation_precision {prec:.2f} < {self._citation_precision_min:.2f}"
                )
                repair_targets.append(
                    "Cite only evidence sources that genuinely support each claim"
                )
        else:
            # No evidence pack — mark citations as self-evident, no penalty
            gate_results["citation_coverage"] = 1.0
            gate_results["citation_precision"] = 1.0

        # 4. Symbolic / numeric checks
        symbolic_ok, symbolic_failures = self._check_symbolic(candidate)
        gate_results["symbolic_checks_pass"] = symbolic_ok
        if not symbolic_ok:
            failures.extend(symbolic_failures)
            repair_targets.append("Fix symbolic/numeric inconsistencies in content")

        # 5. LLM-based rubric scoring (pedagogy, coverage, difficulty)
        llm_score: Optional[float] = None
        llm_findings: list[str] = []
        if self.use_llm_scoring:
            llm_score, llm_findings = self._llm_rubric_score(candidate, task_spec)
            gate_results["llm_rubric_score"] = llm_score
            if llm_score is not None and llm_score < 0.5:
                failures.append(f"llm_rubric_score {llm_score:.2f} < 0.50")
                repair_targets.extend(llm_findings)

        # 6. Composite score
        dims = [
            1.0 if has_ledger else 0.0,
            0.0 if leaked else 1.0,
            gate_results.get("citation_coverage", 1.0),
            gate_results.get("citation_precision", 1.0),
            1.0 if symbolic_ok else 0.0,
        ]
        if llm_score is not None:
            dims.append(llm_score)
        composite = sum(dims) / len(dims)

        if not failures:
            verdict = "PASS"
        elif len(failures) <= 2:
            verdict = "PASS_WITH_EDITS"
        else:
            verdict = "FAIL"

        return DeanScore(
            candidate_id=candidate.candidate_id,
            composite_score=round(composite, 4),
            gate_results=gate_results,
            verdict=verdict,
            failure_reasons=failures,
            repair_targets=repair_targets,
        )

    # ── Citation check ────────────────────────────────────────────────────────

    def _check_citations(
        self, candidate: TeacherCandidate, pack: EvidencePack
    ) -> dict[str, float]:
        from brahmx_sdg.verification.citation_checker import CitationChecker

        checker = CitationChecker()
        # Build chunks dict: "[1]" → text, "[2]" → text, …
        chunks = {
            f"[{i + 1}]": c.get("text", "")
            for i, c in enumerate(pack.top_chunks)
        }
        metrics = checker.check(
            claim_ledger=candidate.claim_ledger,
            chunks=chunks,
            section_text=candidate.content,
        )
        return {
            "citation_coverage": metrics.coverage,
            "citation_precision": metrics.precision,
            "citation_specificity": metrics.specificity,
        }

    # ── Symbolic check ────────────────────────────────────────────────────────

    def _check_symbolic(
        self, candidate: TeacherCandidate
    ) -> tuple[bool, list[str]]:
        if not candidate.tool_checks_required:
            return True, []

        from brahmx_sdg.verification.symbolic_numeric_validator import (
            SymbolicNumericValidator,
        )
        validator = SymbolicNumericValidator()
        results = validator.validate_all(candidate.tool_checks_required)
        failures = [
            f"symbolic_fail [{r.check_type}]: {r.expression} → {r.explanation or r.actual}"
            for r in results
            if not r.passed
        ]
        return len(failures) == 0, failures

    # ── LLM rubric scoring ────────────────────────────────────────────────────

    def _llm_rubric_score(
        self, candidate: TeacherCandidate, task_spec: TaskSpec
    ) -> tuple[Optional[float], list[str]]:
        """
        Ask the Dean LLM to evaluate pedagogy, coverage, and accuracy.
        Returns (score 0-1, list_of_improvement_suggestions).
        Falls back to None on error (won't penalise the candidate).
        """
        import json
        try:
            from brahmx_sdg.routing import TeacherRouter, ModelRole, WorkloadClass

            router = TeacherRouter.from_config(self.routing_config)

            rubric_prompt = _build_rubric_prompt(candidate, task_spec)
            results = router.generate(
                role=ModelRole.DEAN,
                messages=[
                    {"role": "system", "content": DEAN_SYSTEM_PROMPT},
                    {"role": "user", "content": rubric_prompt},
                ],
                n=1,
                temperature=0.1,
                max_tokens=512,
                workload_class=WorkloadClass.BULK,
                response_format={"type": "json_object"},
            )
            if not results:
                return None, []

            raw = results[0]["content"]
            parsed = json.loads(raw)
            score = float(parsed.get("overall_score", 0.5))
            findings = parsed.get("improvement_suggestions", [])
            if not isinstance(findings, list):
                findings = []
            return min(1.0, max(0.0, score)), findings

        except Exception as e:
            logger.warning("dean_llm_score_failed", error=str(e))
            return None, []


# ── Dean LLM prompts ──────────────────────────────────────────────────────────

DEAN_SYSTEM_PROMPT = """\
You are a scientific content reviewer. Evaluate the provided content against \
the task specification and return a JSON assessment. Be strict but fair.

Return ONLY valid JSON in this schema:
{
  "overall_score": <float 0.0-1.0>,
  "pedagogy_score": <float 0.0-1.0>,
  "coverage_score": <float 0.0-1.0>,
  "accuracy_score": <float 0.0-1.0>,
  "improvement_suggestions": ["<suggestion 1>", "<suggestion 2>"]
}
"""


def _build_rubric_prompt(
    candidate: TeacherCandidate, task_spec: TaskSpec
) -> str:
    objectives = "\n".join(f"- {o}" for o in task_spec.objectives) or "(none)"
    content_preview = candidate.content[:2000]
    claim_count = len(candidate.claim_ledger)
    return (
        f"## Task Objectives\n{objectives}\n\n"
        f"## Domain\n{task_spec.domain}\n\n"
        f"## Content to Evaluate (first 2000 chars)\n{content_preview}\n\n"
        f"## Claim Ledger\n{claim_count} claims found.\n\n"
        "Evaluate: Does this content fully address the objectives? "
        "Is it pedagogically sound? Is it factually accurate based on the content itself? "
        "Score 0-1 and list any improvement suggestions."
    )
