# BrahmX SDG — HLD Critique & Improvement Suggestions

## Strong Parts of the Design

1. **Two-plane separation** is architecturally clean and operationally sound. The hard artifact boundary prevents the most common MLOps failure mode (training coupled to inference).
2. **Evidence-first gold path** with structured claim ledgers, deterministic prompts, and multi-stage verification is genuinely production-grade trust architecture.
3. **Model selection table** is realistic — routing by validated runtime rather than theoretical performance.
4. **Scientific lane fan-out** correctly treats each source as a multi-product asset rather than single-use data.
5. **vLLM-first, JetStream-as-exception** is operationally pragmatic and avoids overengineering.

## Weak Points & Under-Specified Areas

### Missing Services
- **No explicit data quality monitoring service**: Need continuous quality metrics over generated bundles (drift detection, quality decay, coverage gaps).
- **No cost accounting service**: No mechanism to track $/sample, $/bundle, or budget controls per lane/tier.
- **No model versioning registry**: HLD mentions model roles but lacks a formal model version lifecycle (promote, deprecate, retire).
- **No A/B testing framework** for teacher models or verification thresholds.

### Missing Contracts
- **Teacher-to-claim-ledger extraction** is underspecified. How exactly does structured claim extraction work? Is it a separate model call or embedded in the teacher prompt?
- **Human review → provenance feedback loop** is mentioned but the data contract for human review decisions is undefined.
- **Rollback contract**: What happens to downstream corpora and trained checkpoints when a gold bundle is rolled back?

### Missing Observability
- No structured logging standard defined
- No metrics/alerting architecture for pipeline health
- No SLOs defined for generation latency, verification throughput, or corpus freshness
- No dashboard design for trust stack health (claim coverage, citation precision trends)

### Missing Human-in-the-Loop
- Human review is described as a gate but the UI/workflow is unspecified
- No escalation routing rules (which samples auto-escalate, which are batch-reviewed)
- No annotation interface for KB gap resolution
- No inter-annotator agreement measurement

### Missing Data Quality Loops
- No feedback from downstream eval to upstream generation (eval failures should trigger upstream investigation)
- No automatic KB gap ticket → CFT enrichment → re-generation loop
- No quality regression detection across corpus versions

### Missing Dataset Economics Controls
- No per-lane budget caps
- No cost/quality Pareto tracking
- No automatic scaling of cheap vs. expensive teacher allocation based on difficulty signals

### Missing Security/Compliance
- No data access controls or audit logging
- No PII detection pipeline
- No content policy enforcement beyond "safety_compliance: PASS"
- No differential privacy considerations for generated data

## Concrete HLD Improvements

### Must-Have (Build Now)
1. **Add a DataQualityMonitor service** that continuously computes and alerts on: citation precision trends, claim coverage ratio, dean pass rate, auditor agreement rate, bundle rejection rate per lane.
2. **Formalize the claim ledger extraction contract**: Define whether extraction is in-prompt (structured output mode) or a separate model call with its own retry/validation.
3. **Add structured logging and OpenTelemetry tracing** to every pipeline stage with correlation IDs from task spec through to training slice.
4. **Define rollback semantics**: When a gold bundle is revoked, all downstream corpus versions referencing it must be flagged, and any training runs using those versions must be logged.

### Should-Have (Build in Phase 2)
5. **Cost accounting per bundle**: Track inference cost, verification cost, human review cost. Enable per-lane cost caps.
6. **Eval-to-generation feedback loop**: When a trained checkpoint fails eval on a specific domain, automatically surface the relevant gold/silver bundles for review.
7. **Human review UI spec**: Define the annotation interface, routing rules, and IAA measurement.

### Nice-to-Have (Phase 3+)
8. **A/B testing framework** for teacher models and verification thresholds.
9. **Automatic KB gap resolution** pipeline that identifies gaps and triggers enrichment.

## Where the HLD is Overengineered
- **7 specialized lanes from day one**: Start with 2-3 lanes (simulation JSON, curriculum, code/tool-use) and add the rest incrementally. Each lane is a significant engineering investment.
- **Reasoning-budget lane** is premature until the base pipeline is stable. The truncation policy is complex and the training impact is uncertain.

## Where the HLD is Underengineered
- **Tokenizer freeze process** is mentioned as a policy but the operational procedure (how to test, validate, and lock the tokenizer) is missing.
- **Corpus versioning semantics** need formal definition: semantic versioning? Content-hash versioning? What constitutes a breaking change?
- **MaxText training configuration** is mentioned but the integration pattern (how to parameterize MaxText from run specs) is unspecified.
