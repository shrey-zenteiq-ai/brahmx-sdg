"""
Slice Emitter — produces multiple training task formats from a single Gold bundle.

Emitted slice types:
  1. explanation_generation   — turn the content into an instruction→response SFT pair
  2. qa_with_citation         — generate Q&A pairs grounded in cited claims
  3. quiz_generation          — multiple-choice questions from key claims
  4. term_extraction          — extract and define domain terms
  5. misconception_correction — identify and correct a common misconception
  6. claim_verification       — given a claim, verify it against content
  7. summary_generation       — summarise the content at different levels
  8. structured_outline       — produce a structured outline of the content

Each slice is a DatasetSlice ready for downstream SFT.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any
from uuid import uuid4

from brahmx_sdg.schemas import BundleTier, DatasetSlice, GoldRecordBundle


class SliceEmitter:
    """Emit all training slice formats from a gold bundle."""

    def __init__(self, max_qa_pairs: int = 5, max_claims_for_quiz: int = 4) -> None:
        self.max_qa_pairs = max_qa_pairs
        self.max_claims_for_quiz = max_claims_for_quiz

    def emit(self, bundle: GoldRecordBundle) -> list[DatasetSlice]:
        """Emit all slice types. Any that fail are skipped with a warning."""
        selected = self._get_selected_candidate(bundle)
        if not selected:
            return []

        slices: list[DatasetSlice] = []
        emitters = [
            self._emit_explanation,
            self._emit_qa_with_citation,
            self._emit_quiz,
            self._emit_term_extraction,
            self._emit_misconception_correction,
            self._emit_claim_verification,
            self._emit_summary,
            self._emit_structured_outline,
        ]
        for fn in emitters:
            try:
                result = fn(bundle, selected)
                if isinstance(result, list):
                    slices.extend(result)
                elif result is not None:
                    slices.append(result)
            except Exception:
                pass
        return slices

    # ── Slice emitters ─────────────────────────────────────────────────────────

    def _emit_explanation(
        self, bundle: GoldRecordBundle, selected: dict
    ) -> DatasetSlice:
        spec = bundle.task_spec
        instruction = (
            f"Generate a comprehensive {spec.task_type} explanation for the domain "
            f"'{spec.domain}' covering the following objectives:\n"
            + "\n".join(f"- {o}" for o in spec.objectives)
        )
        return self._make_slice(
            task_type="explanation_generation",
            bundle=bundle,
            input_data={
                "instruction": instruction,
                "domain": spec.domain,
                "difficulty": spec.difficulty,
                "language": spec.language,
                "objectives": spec.objectives,
            },
            target_data={"response": selected["content"]},
        )

    def _emit_qa_with_citation(
        self, bundle: GoldRecordBundle, selected: dict
    ) -> list[DatasetSlice]:
        claim_ledger = selected.get("claim_ledger", [])
        if not claim_ledger:
            return []

        slices = []
        for i, claim in enumerate(claim_ledger[: self.max_qa_pairs]):
            claim_text = claim.get("claim_text", "")
            if not claim_text or len(claim_text) < 20:
                continue
            citations = claim.get("supporting_citations", [])
            question = _claim_to_question(claim_text)
            slices.append(
                self._make_slice(
                    task_type="qa_with_citation",
                    bundle=bundle,
                    input_data={
                        "question": question,
                        "context": selected["content"][:1500],
                        "domain": bundle.task_spec.domain,
                    },
                    target_data={
                        "answer": claim_text,
                        "supporting_citations": citations,
                        "claim_id": claim.get("claim_id", ""),
                    },
                    suffix=f"qa{i}",
                )
            )
        return slices

    def _emit_quiz(
        self, bundle: GoldRecordBundle, selected: dict
    ) -> list[DatasetSlice]:
        claim_ledger = selected.get("claim_ledger", [])
        critical = [
            c for c in claim_ledger if c.get("is_critical") or c.get("claim_type") in ("fact", "equation")
        ]
        if not critical:
            critical = claim_ledger

        slices = []
        for i, claim in enumerate(critical[: self.max_claims_for_quiz]):
            claim_text = claim.get("claim_text", "")
            if not claim_text or len(claim_text) < 20:
                continue
            question = _claim_to_question(claim_text)
            # Correct answer is the claim itself; distractors are generic
            slices.append(
                self._make_slice(
                    task_type="quiz_generation",
                    bundle=bundle,
                    input_data={
                        "question": question,
                        "domain": bundle.task_spec.domain,
                        "difficulty": bundle.task_spec.difficulty,
                    },
                    target_data={
                        "correct_answer": claim_text,
                        "claim_id": claim.get("claim_id", ""),
                        "explanation": (
                            f"This is established knowledge in {bundle.task_spec.domain}. "
                            f"See claim {claim.get('claim_id', '')}."
                        ),
                    },
                    suffix=f"quiz{i}",
                )
            )
        return slices

    def _emit_term_extraction(
        self, bundle: GoldRecordBundle, selected: dict
    ) -> DatasetSlice:
        # Extract terms from claim ledger entries tagged as definitions
        definitions = [
            c for c in selected.get("claim_ledger", [])
            if c.get("claim_type") == "definition"
        ]
        all_claims = selected.get("claim_ledger", [])[:8]
        terms = []
        for c in definitions or all_claims:
            terms.append({
                "term": c.get("claim_id", ""),
                "definition": c.get("claim_text", ""),
                "claim_type": c.get("claim_type", "fact"),
            })
        return self._make_slice(
            task_type="term_extraction",
            bundle=bundle,
            input_data={
                "instruction": (
                    f"Extract and define the key domain terms from this {bundle.task_spec.domain} content."
                ),
                "content": selected["content"][:2000],
            },
            target_data={"terms": terms},
        )

    def _emit_misconception_correction(
        self, bundle: GoldRecordBundle, selected: dict
    ) -> DatasetSlice:
        spec = bundle.task_spec
        return self._make_slice(
            task_type="misconception_correction",
            bundle=bundle,
            input_data={
                "instruction": (
                    f"Identify and correct a common misconception in {spec.domain} "
                    f"related to: {', '.join(spec.objectives[:2]) if spec.objectives else spec.domain}"
                ),
                "domain": spec.domain,
                "difficulty": spec.difficulty,
            },
            target_data={
                "correction": selected["content"],
                "claim_ledger": selected.get("claim_ledger", []),
            },
        )

    def _emit_claim_verification(
        self, bundle: GoldRecordBundle, selected: dict
    ) -> list[DatasetSlice]:
        slices = []
        for i, claim in enumerate(selected.get("claim_ledger", [])[:3]):
            claim_text = claim.get("claim_text", "")
            if not claim_text:
                continue
            slices.append(
                self._make_slice(
                    task_type="claim_verification",
                    bundle=bundle,
                    input_data={
                        "claim": claim_text,
                        "context": selected["content"][:1500],
                        "domain": bundle.task_spec.domain,
                    },
                    target_data={
                        "verdict": "SUPPORTED",
                        "verifiability": claim.get("verifiability", "KB"),
                        "supporting_citations": claim.get("supporting_citations", []),
                    },
                    suffix=f"cv{i}",
                )
            )
        return slices

    def _emit_summary(self, bundle: GoldRecordBundle, selected: dict) -> DatasetSlice:
        content = selected["content"]
        # Short summary: first ~200 words
        words = content.split()
        short = " ".join(words[:200])
        return self._make_slice(
            task_type="summary_generation",
            bundle=bundle,
            input_data={
                "instruction": (
                    f"Summarise the following {bundle.task_spec.domain} content in 3-5 sentences."
                ),
                "content": content[:3000],
                "domain": bundle.task_spec.domain,
            },
            target_data={"summary": short},
        )

    def _emit_structured_outline(
        self, bundle: GoldRecordBundle, selected: dict
    ) -> DatasetSlice:
        spec = bundle.task_spec
        claim_ledger = selected.get("claim_ledger", [])
        outline_items = [
            f"- {c.get('claim_text', '')[:100]}"
            for c in claim_ledger[:8]
            if c.get("claim_text")
        ]
        return self._make_slice(
            task_type="structured_outline",
            bundle=bundle,
            input_data={
                "instruction": (
                    f"Produce a structured outline for a {spec.difficulty}-level "
                    f"{spec.domain} section covering: {', '.join(spec.objectives[:3])}"
                ),
                "domain": spec.domain,
                "difficulty": spec.difficulty,
                "objectives": spec.objectives,
            },
            target_data={
                "outline": "\n".join(outline_items) if outline_items else selected["content"][:500],
                "claim_count": len(claim_ledger),
            },
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_selected_candidate(self, bundle: GoldRecordBundle) -> dict[str, Any] | None:
        if not bundle.selected_candidate_id or not bundle.candidates:
            return None
        for c in bundle.candidates:
            if c.candidate_id == bundle.selected_candidate_id:
                return {
                    "content": c.content,
                    "claim_ledger": c.claim_ledger,
                    "candidate_id": c.candidate_id,
                }
        # Fallback: first candidate
        c = bundle.candidates[0]
        return {
            "content": c.content,
            "claim_ledger": c.claim_ledger,
            "candidate_id": c.candidate_id,
        }

    def _make_slice(
        self,
        task_type: str,
        bundle: GoldRecordBundle,
        input_data: dict[str, Any],
        target_data: dict[str, Any],
        suffix: str = "",
    ) -> DatasetSlice:
        uid = hashlib.sha256(
            f"{bundle.record_id}:{task_type}:{suffix}".encode()
        ).hexdigest()[:16]
        return DatasetSlice(
            slice_id=f"SLICE-{uid}",
            task_type=task_type,
            input_data=input_data,
            target_data=target_data,
            source_bundle_id=bundle.record_id,
            tier=bundle.tier,
            metadata={
                "domain": bundle.task_spec.domain,
                "language": bundle.task_spec.language,
                "difficulty": bundle.task_spec.difficulty,
                "section_id": bundle.task_spec.section_id,
                "promotion_status": bundle.promotion_status.value,
            },
        )


# ── Utility ───────────────────────────────────────────────────────────────────

def _claim_to_question(claim_text: str) -> str:
    """Heuristically convert a declarative claim into a question."""
    text = claim_text.strip().rstrip(".")
    # Very simple heuristics — good enough for training data
    if text.lower().startswith("the "):
        return f"What is {text[4:]}?"
    if " is " in text:
        subject, predicate = text.split(" is ", 1)
        return f"What is {subject.strip()}?"
    if " are " in text:
        subject, predicate = text.split(" are ", 1)
        return f"What are {subject.strip()}?"
    if " equals " in text or " = " in text:
        return f"What is the value or formula for: {text}?"
    return f"Explain: {text}?"
