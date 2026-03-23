"""
Deterministic Prompt Constructor.

Builds reproducible prompts that ask teachers to return structured JSON
containing both the educational content and a machine-auditable claim ledger.
"""
from __future__ import annotations

import json
from typing import Optional

from brahmx_sdg.schemas import EvidencePack, PromptSpec, TaskSpec
from brahmx_sdg.common import deterministic_hash

# JSON schema shown to the teacher in the prompt
_RESPONSE_SCHEMA = {
    "content": "<string: full educational content — explanations, worked examples, exercises>",
    "claim_ledger": [
        {
            "claim_id": "CLM-001",
            "claim_text": "<atomic factual assertion>",
            "claim_type": "fact | equation | definition | threshold | procedure | policy",
            "verifiability": "KB | self-evident | assumption",
            "supporting_citations": ["[1]", "[3]"],
            "is_critical": True,
        }
    ],
}

_SYSTEM_TEMPLATE = """\
You are a scientific content generator for an AI training dataset.

Your task: generate high-quality educational content that is STRICTLY grounded \
in the evidence provided. Every non-trivial factual claim MUST be cited to a \
numbered evidence source.

Domain: {domain}
Language: {language}
Task type: {task_type}
Difficulty: {difficulty}

RULES:
1. Do NOT assert any claim that cannot be supported by the numbered evidence sources.
2. Include a complete Claim Ledger listing every factual claim.
3. Each claim must reference the evidence source(s) that support it using [N] notation.
4. Claims with verifiability="self-evident" need no citation (e.g. definitions of terms).
5. Claims with verifiability="assumption" are explicit unknowns — mark them clearly.
6. Output ONLY valid JSON matching the schema below. No prose outside the JSON.

REQUIRED OUTPUT SCHEMA:
{schema}
"""

_USER_TEMPLATE = """\
## Task Objective
{task_type}: {objectives}

## Required Claims to Cover
{required_claims}

## Evidence Sources (cite using [N] notation)
{evidence_sources}

{constraints_block}
Generate the complete response as a single JSON object matching the schema.
"""


class PromptConstructor:
    def build(self, task_spec: TaskSpec, evidence_pack: Optional[EvidencePack] = None) -> PromptSpec:
        system_prompt = self._build_system_prompt(task_spec)
        user_prompt = self._build_user_prompt(task_spec, evidence_pack)
        must_cover = [rc.get("statement", rc.get("claim_id", "")) for rc in task_spec.required_claims]
        must_not_contradict = []
        if evidence_pack:
            must_not_contradict = [
                c.get("statement", "") for c in evidence_pack.known_constraints
            ]
        content = system_prompt + user_prompt
        return PromptSpec(
            task_id=task_spec.task_id,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            must_cover=must_cover,
            must_not_contradict=must_not_contradict,
            evidence_pack_hash=evidence_pack.cft_snapshot_hash if evidence_pack else "",
            prompt_hash=deterministic_hash(content),
        )

    # ── Private builders ───────────────────────────────────────────────────────

    def _build_system_prompt(self, spec: TaskSpec) -> str:
        schema_str = json.dumps(_RESPONSE_SCHEMA, indent=2)
        return _SYSTEM_TEMPLATE.format(
            domain=spec.domain or "general science",
            language=spec.language,
            task_type=spec.task_type,
            difficulty=spec.difficulty,
            schema=schema_str,
        )

    def _build_user_prompt(
        self, spec: TaskSpec, pack: Optional[EvidencePack]
    ) -> str:
        # Objectives
        obj_str = "\n".join(f"- {o}" for o in spec.objectives) if spec.objectives else "(none)"

        # Required claims
        if spec.required_claims:
            rc_lines = "\n".join(
                f"- [{i+1}] {rc.get('statement', rc.get('claim_id', ''))}"
                for i, rc in enumerate(spec.required_claims)
            )
        else:
            rc_lines = "(no specific required claims)"

        # Evidence sources
        if pack and pack.top_chunks:
            ev_lines = []
            for i, chunk in enumerate(pack.top_chunks[:15]):
                text = chunk.get("text", "")[:600].replace("\n", " ")
                source = chunk.get("source", chunk.get("doc_id", ""))
                ev_lines.append(f"[{i+1}] {text}" + (f"  (source: {source})" if source else ""))
            evidence_str = "\n".join(ev_lines)
        else:
            evidence_str = "(no evidence pack provided — generate from general knowledge and mark all claims as self-evident)"

        # Constraints
        constraints_block = ""
        if pack and pack.known_constraints:
            lines = "\n".join(f"- {c.get('statement', '')}" for c in pack.known_constraints)
            constraints_block = f"## MUST NOT CONTRADICT\n{lines}\n\n"

        return _USER_TEMPLATE.format(
            task_type=spec.task_type,
            objectives=obj_str,
            required_claims=rc_lines,
            evidence_sources=evidence_str,
            constraints_block=constraints_block,
        )
