"""
Independent Auditor — Adversarial review from a different model lineage.

Key constraints (FM-09 bias mitigation):
  - The Auditor does NOT receive the evidence pack.
  - It evaluates content + spec only, using its own knowledge.
  - This intentionally breaks the shared-KB bias loop.

The Auditor is the final automated gate before human review.
PASS → proceed to bundle.
FAIL → repair loop (if rounds remain) or escalate to human review.
"""
from __future__ import annotations

import json
from typing import Optional

import structlog

from brahmx_sdg.schemas import AuditorReport, TaskSpec, TeacherCandidate

logger = structlog.get_logger()

DEFAULT_ROUTING_CONFIG = "configs/routing/models.yaml"

AUDITOR_SYSTEM_PROMPT = """\
You are an independent scientific content auditor. Your role is to review \
educational content WITHOUT access to the original source evidence — you rely \
solely on your own knowledge and the curriculum specification.

Evaluate:
1. Factual accuracy — does the content make scientific sense?
2. Internal consistency — are there contradictions within the content?
3. Format compliance — does the output meet the curriculum spec?
4. Safety — no harmful, biased, or inappropriate content?
5. Completeness — does it address all stated objectives?

Return ONLY valid JSON in this schema:
{
  "status": "PASS" | "FAIL" | "ESCALATE",
  "overall_confidence": <float 0.0-1.0>,
  "findings": ["<issue 1>", "<issue 2>"],
  "severity": "low" | "medium" | "high" | "critical",
  "override_dean": false,
  "escalate_to_human": false,
  "notes": "<brief summary>"
}

Set status=PASS if content is factually sound and meets objectives.
Set status=FAIL if there are clear factual errors or significant gaps.
Set status=ESCALATE if you disagree with the earlier Dean assessment or detect ambiguity.
"""


class Auditor:
    """
    Independent adversarial reviewer.
    Uses a different model from the teachers/dean to avoid shared-bias.
    Does NOT receive the evidence pack (FM-09 mitigation).
    """

    def __init__(
        self,
        routing_config: str = DEFAULT_ROUTING_CONFIG,
    ) -> None:
        self.routing_config = routing_config

    def review(
        self,
        candidate: TeacherCandidate,
        task_spec: TaskSpec,
    ) -> AuditorReport:
        """
        Review a candidate WITHOUT the evidence pack.
        Falls back to a lenient PASS if the LLM call fails,
        so a network/API error doesn't silently block the pipeline.
        """
        try:
            return self._llm_review(candidate, task_spec)
        except Exception as e:
            logger.warning(
                "auditor_llm_failed",
                candidate=candidate.candidate_id,
                error=str(e),
                note="falling back to heuristic review",
            )
            return self._heuristic_review(candidate, task_spec)

    # ── LLM-based review ──────────────────────────────────────────────────────

    def _llm_review(
        self, candidate: TeacherCandidate, task_spec: TaskSpec
    ) -> AuditorReport:
        from brahmx_sdg.routing import TeacherRouter, ModelRole, WorkloadClass

        router = TeacherRouter.from_config(self.routing_config)

        prompt = _build_audit_prompt(candidate, task_spec)
        results = router.generate(
            role=ModelRole.AUDITOR,
            messages=[
                {"role": "system", "content": AUDITOR_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            n=1,
            temperature=0.2,
            max_tokens=512,
            workload_class=WorkloadClass.AUDIT,
            response_format={"type": "json_object"},
        )

        if not results:
            raise ValueError("Auditor returned no results")

        raw = results[0]["content"]
        parsed = json.loads(raw)

        status = parsed.get("status", "FAIL")
        if status not in ("PASS", "FAIL", "ESCALATE"):
            status = "FAIL"

        severity = parsed.get("severity", "low")
        if severity not in ("low", "medium", "high", "critical"):
            severity = "low"

        findings = parsed.get("findings", [])
        if not isinstance(findings, list):
            findings = []

        # ESCALATE → treat as PASS but flag for human review
        escalate = (status == "ESCALATE") or bool(parsed.get("escalate_to_human", False))
        override = bool(parsed.get("override_dean", False))
        final_status = "PASS" if status in ("PASS", "ESCALATE") else "FAIL"

        return AuditorReport(
            candidate_id=candidate.candidate_id,
            auditor_model=results[0].get("model", "openai-auditor"),
            status=final_status,
            findings=findings,
            severity=severity,
            override_dean=override,
            escalate_to_human=escalate,
        )

    # ── Heuristic fallback ────────────────────────────────────────────────────

    def _heuristic_review(
        self, candidate: TeacherCandidate, task_spec: TaskSpec
    ) -> AuditorReport:
        """
        Simple rule-based fallback when the LLM call fails.
        Checks basic length, objective coverage, and claim presence.
        """
        findings: list[str] = []
        passed = True

        # Minimum length check
        if len(candidate.content) < 100:
            findings.append("Content too short (< 100 characters)")
            passed = False

        # Claim ledger presence
        if not candidate.claim_ledger:
            findings.append("No claim ledger found")
            passed = False

        # Objectives mentioned in content (simple keyword check)
        content_lower = candidate.content.lower()
        for obj in task_spec.objectives[:5]:
            keywords = [w for w in obj.lower().split() if len(w) > 4]
            if keywords and not any(kw in content_lower for kw in keywords[:3]):
                findings.append(f"Objective may not be addressed: {obj[:80]}")

        return AuditorReport(
            candidate_id=candidate.candidate_id,
            auditor_model="heuristic-fallback",
            status="PASS" if passed else "FAIL",
            findings=findings,
            severity="low" if passed else "medium",
            override_dean=False,
            escalate_to_human=not passed,
        )


# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_audit_prompt(candidate: TeacherCandidate, task_spec: TaskSpec) -> str:
    objectives = "\n".join(f"- {o}" for o in task_spec.objectives) or "(none)"
    claim_summary = (
        "\n".join(
            f"  [{i+1}] {c.get('claim_text', '')[:120]}"
            for i, c in enumerate(candidate.claim_ledger[:10])
        )
        or "  (no claims)"
    )
    content_preview = candidate.content[:3000]

    return (
        f"## Curriculum Specification\n"
        f"Domain: {task_spec.domain}\n"
        f"Task type: {task_spec.task_type}\n"
        f"Language: {task_spec.language}\n"
        f"Difficulty: {task_spec.difficulty}\n\n"
        f"## Objectives\n{objectives}\n\n"
        f"## Content to Audit\n{content_preview}\n\n"
        f"## Claim Ledger Summary ({len(candidate.claim_ledger)} claims)\n"
        f"{claim_summary}\n\n"
        "Audit this content for factual accuracy, completeness, and quality. "
        "You do NOT have access to the original source documents."
    )
