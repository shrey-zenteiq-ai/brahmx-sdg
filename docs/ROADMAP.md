# BrahmX SDG — Implementation Roadmap

## Phase 0: Spike / Proof-of-Concept (2-3 weeks)

**Goal**: Validate the end-to-end gold path with a single source, single teacher, single domain.

**Components to Build**:
- Minimal CFT with 50-100 hand-curated claims
- Evidence Pack Builder (BM25 retrieval only)
- Deterministic Prompt Constructor (1 template)
- Single teacher call via vLLM (Qwen3-32B on GPU)
- Claim Ledger extraction (structured output parsing)
- Dean scoring (3 gates: schema, claim ledger, citation coverage)
- Bundle assembler (minimal fields)
- Slice emitter (2 formats: explanation + QA)
- Local artifact store (filesystem)

**Dependencies**: vLLM running locally or on a single GPU node

**Team**: 1 senior ML engineer + 1 platform engineer

**Blockers**: vLLM TPU validation for Qwen3-32B, CFT seed data

**Exit Criteria**:
- Can generate a gold bundle from a task spec end-to-end
- Dean scoring produces meaningful pass/fail decisions
- Generated bundles contain valid claim ledgers with citations
- 50+ gold bundles generated with >70% dean pass rate

---

## Phase 1: MVP (6-8 weeks)

**Goal**: Gold + silver pipeline with 2 lanes, corpus assembly, and first student training run.

**Components to Build**:
- Full gold pipeline with multi-candidate generation (n=3)
- Auditor service (Llama-3.3-70B on GPU)
- Citation checker (BM25 + TF-IDF)
- Symbolic/numeric validator (SymPy + Pint)
- Silver generator with 2 lanes:
  - Simulation JSON lane
  - Curriculum difficulty lane
- Teacher router with vLLM TPU + GPU pools
- Corpus assembler (dedup + basic decontam + Parquet export)
- Provenance registry (in-memory → PostgreSQL)
- Release manifest creation
- MaxText training launcher (CPT + SFT-gold)
- Basic eval runner
- Kubeflow pipeline definitions (gold + training)
- CI/CD with gold regression tests

**Dependencies**: TPU allocation, MaxText cluster setup, KB enrichment (500+ claims)

**Team**: 2 ML engineers + 1 platform engineer + 1 data engineer + 0.5 KB curator

**Blockers**: TPU quota, MaxText integration testing, KB enrichment velocity

**Exit Criteria**:
- 500+ gold bundles, 2000+ silver bundles generated
- First student (3.5B) trained on gold-heavy corpus
- Eval shows meaningful signal vs. baseline
- Kubeflow pipelines running in staging
- Provenance traceable from training example to source

---

## Phase 2: Production Hardening (8-12 weeks)

**Goal**: All 7 lanes, full trust stack, production Kubeflow, human review.

**Components to Build**:
- Remaining 5 scientific lanes:
  - Bounded dialogue
  - LaTeX artifact (with compile loop)
  - Code/tool-use (with sandbox)
  - Multilingual (IndicTrans2 integration)
  - Reasoning-budget
- Code execution sandbox (nsjail)
- LaTeX compile validator
- Human review UI/backend
- Full release governance workflow
- Rollback manager
- Cost accounting per bundle
- Data quality monitoring dashboard
- All training stages: preference, RL, distill recovery, LC recovery
- Structured logging + OpenTelemetry
- Alert rules for pipeline health
- Production Kubeflow with retry/caching/approval gates
- 8B student training with full stage progression

**Dependencies**: Human review team, nsjail setup, IndicTrans2 deployment

**Team**: 3 ML engineers + 2 platform engineers + 1 frontend engineer + 1 data engineer + 1 KB curator

**Blockers**: Human review team availability, nsjail security review, IndicTrans2 quality validation

**Exit Criteria**:
- All 7 lanes producing validated data
- 5000+ gold bundles, 20000+ silver bundles
- Human review processing 50+ samples/day
- 8B student competitive on science benchmarks
- Full provenance and rollback capability operational
- Production alerts firing correctly

---

## Phase 3: Scale-Out (12-16 weeks)

**Goal**: Multi-tenant, cost-optimized, 35B student.

**Components to Build**:
- Multi-tenant pipeline isolation
- Cost/quality Pareto optimization
- Automatic lane scaling based on demand
- A/B testing framework for teachers/thresholds
- Eval-to-generation feedback loop
- Advanced decontamination (model-based)
- KB gap auto-detection and enrichment triggers
- 35B dense student training
- Production serving pipeline (vLLM)
- Dataset economics dashboard

**Team**: 4 ML engineers + 2 platform + 1 data + 1 KB curator + 1 SRE

**Exit Criteria**:
- 35B student trained and evaluated
- Cost per gold bundle tracked and optimized
- Multi-tenant isolation validated
- Automated scaling operational

---

## Phase 4: Advanced Research Extensions (Ongoing)

**Goal**: 109B MoE, advanced lanes, continuous improvement.

**Components to Build**:
- 109B MoE student (research target)
- Lab-protocol generation lane
- Simulation-to-report generation lane
- Literature review synthesis lane
- Structured theorem/proof lane
- Continuous learning from eval feedback
- Advanced preference optimization
- Multi-modal scientific data (diagrams, plots)

**Team**: Research team + infrastructure support

**Exit Criteria**: N/A (ongoing research program)

---

## Build Order Summary

```
Week 1-3:   Spike — single gold path end-to-end
Week 4-6:   MVP foundations — multi-candidate, auditor, routing
Week 7-10:  MVP complete — 2 lanes, corpus, first training
Week 11-14: Trust stack hardening — all validators, human review
Week 15-18: All lanes operational
Week 19-22: 8B student full training progression
Week 23-26: Production hardening, monitoring, alerts
Week 27-30: Scale-out, cost optimization
Week 31+:   35B training, advanced research
```
