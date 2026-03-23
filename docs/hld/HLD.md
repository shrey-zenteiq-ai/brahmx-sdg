# BrahmX SDG — High-Level Design (HLD)

**Version**: 1.0.0  
**Status**: Architecture Review Ready  
**Last Updated**: 2025-06-01

---

## 1. System Overview

BrahmX SDG is a two-plane architecture for sovereign, science-first foundation model training data generation and model distillation.

### 1.1 Design Principles

1. **Two-plane isolation**: Data Factory and Model Factory communicate only via static, versioned artifacts
2. **Hard distillation only**: No soft KD, no token-logit matching — teacher outputs become canonical text
3. **Evidence-first gold path**: Every gold sample is evidence-bounded and verification-heavy
4. **Silver on top of gold truth**: Silver expands breadth but never replaces gold's trust basis
5. **vLLM-first inference**: vLLM on TPU for validated bulk; GPU + vLLM for frontier models
6. **MaxText for training**: MaxText is the Model Factory runtime, not the teacher-serving stack
7. **Trust over scale**: Claim ledgers, dean scoring, citations, provenance, and governance are mandatory
8. **Production realism**: Route around unsupported model/runtime combos; no speculative design

### 1.2 Top-Level Architecture Diagram (ASCII)

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                               SOURCE PLANE                                       │
│   textbooks | papers | reports | HF datasets | code | web | simulation JSON      │
└────────────────────────────────────┬─────────────────────────────────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                        INGESTION + POLICY LAYER                                  │
│   license tagging | normalization | decontam | tokenizer freeze | script policy   │
└──────────┬──────────────────────────────────────────────────────────┬────────────┘
           │                                                          │
           ▼                                                          ▼
┌─────────────────────────────────┐   ┌───────────────────────────────────────────┐
│       GOLD DATA FACTORY         │   │          SILVER DATA FACTORY              │
│ CFT → Evidence Pack → Prompt    │   │  Seed/Persona/Difficulty Graph            │
│ → Teacher A/B/C → Claim Ledger  │   │  → Specialized Scientific Lanes          │
│ → Dean + symbolic checks        │   │  → Lane Validators → Silver Bundle       │
│ → Auditor → Human Review        │   │                                           │
│ → Gold Record Bundle            │   │                                           │
└──────────────┬──────────────────┘   └──────────────────┬────────────────────────┘
               │                                          │
               └─────────────────┬────────────────────────┘
                                 ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                    CANONICALIZATION + DATA PRODUCT FAN-OUT                        │
│  pretraining rewrites | SFT pairs | QA | traces | tools | preference | LaTeX    │
└────────────────────────────────────┬─────────────────────────────────────────────┘
                                     ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                 TRUST + CORPUS ASSEMBLY + RELEASE GOVERNANCE                     │
│  dedup | decontam | policy gates | provenance registry | manifests | mixtures    │
└────────────────────────────────────┬─────────────────────────────────────────────┘
                                     ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│              MODEL FACTORY (STATIC VERSIONED CORPORA ONLY)                       │
│  CPT → SFT → Preference/Ranking → RL/Alignment → Distill Recovery → LC Recovery │
│  MaxText on TPU — no live teacher dependency                                     │
└────────────────────────────────────┬─────────────────────────────────────────────┘
                                     ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                          DEPLOYMENT + SERVING                                    │
│  student packaging | eval | vLLM default | JetStream rare exception              │
└──────────────────────────────────────────────────────────────────────────────────┘
```

## 2. Data Factory vs Model Factory Split

### 2.1 Data Factory

**Responsibility**: Ingest scientific sources, generate evidence-bounded gold and silver data, verify, canonicalize, and emit static versioned corpora.

**Key subsystems**:
- Source Ingestion & Normalization
- Canonical Fact Table (CFT)
- Evidence Pack Builder
- Deterministic Prompt Constructor
- Teacher Generation (multi-model)
- Claim Ledger Extraction
- Dean Scoring + Symbolic/Numeric Validators
- Independent Auditor
- Human Review Gate
- Specialized Scientific Lanes (7 lanes)
- Silver Data Factory
- Canonicalization & Fan-out
- Corpus Assembly & Release

**Outputs**: ArrayRecord/Parquet versioned corpora with manifests, provenance, and release metadata.

### 2.2 Model Factory

**Responsibility**: Consume static corpora and produce trained student checkpoints through a multi-stage training pipeline.

**Key stages**:
1. Tokenizer freeze & corpus replay validation
2. Continued Pretraining (CPT) — science-heavy
3. Gold-heavy SFT
4. Silver breadth SFT
5. Preference / ranking
6. RL / alignment
7. Hard distillation recovery
8. Long-context recovery
9. Deployment packaging

**Runtime**: MaxText on TPU for all training stages. No live teacher dependency.

### 2.3 Artifact Handoff Boundary

The **only** interface between planes is:
- Static versioned corpora (ArrayRecord/Parquet)
- Release manifests (JSON)
- Provenance registry entries
- Dataset version IDs

The Model Factory **never** calls a live teacher. All training data is pre-generated, canonicalized, re-tokenized, and sealed before training begins.

## 3. Trust & Governance Layer

### 3.1 Gold Path Acceptance Chain

```
CFT → Evidence Pack → Deterministic Prompt → Teacher A/B/C
  → Claim Ledger → Dean + citations + symbolic/tool checks
  → Auditor → Human Review → Gold Record Bundle
```

### 3.2 Trust Components

| Component | Purpose | Accept/Reject Logic |
|-----------|---------|---------------------|
| Canonical Fact Table | Atomic scientific claims with provenance | Reject unverifiable/contradictory atoms |
| Evidence Pack Builder | Task-local evidence bundles | Block if required evidence missing |
| Claim Ledger | Machine-auditable assertions from teacher | Reject missing/orphan/unsupported claims |
| Dean Scoring | Rubric-based structured acceptance | Pass/repair/reject on thresholds |
| Citation Checker | Lexical citation verification (BM25/TF-IDF) | Reject false/hallucinated citations |
| Symbolic/Numeric | Formula, unit, numerical consistency | Reject unit mismatches, impossible values |
| Code Sandbox | Compile/execute generated code | Reject compile errors, failing tests |
| LaTeX Sandbox | Compile LaTeX artifacts | Reject compile failures after retry |
| Independent Auditor | Adversarial review (different model lineage) | Override dean or escalate to human |
| Human Review | High-risk/disagreement samples | Promote, downgrade, or reject |
| Provenance Registry | Manifests, lineage, promotion status | No training use without manifest + approval |

## 4. Serving & Infrastructure Routing

| Workload | Default Route | Keep Out Of |
|----------|---------------|-------------|
| Bulk evidence-grounded generation | vLLM TPU single-host pool | JetStream/MaxText inference |
| Frontier science / code / audits | GPU + vLLM | Speculative TPU vLLM |
| Routine dean scoring | vLLM TPU or GPU | Multi-host TPU inference |
| Rare multi-host TPU need | JetStream + MaxText + Pathways | General teacher-serving |
| Student training (CPT/SFT/RL) | MaxText on TPU | Live teacher coupling |
| Production student serving | vLLM on GPU or validated TPU | Research-only stacks |

### 4.1 TPU Allocation Policy

- 70-85% of TPU fabric → MaxText training jobs
- 10-20% → single-host vLLM TPU bulk generation
- 0-10% → controlled experiments / exceptional serving

## 5. Model Selection

| Role | Primary Model | Runtime |
|------|--------------|---------|
| Teacher A (bulk science) | Qwen3-32B-Instruct | vLLM TPU single-host |
| Teacher B (difficult science) | Kimi K2.5 | vLLM GPU |
| Teacher C (alternate lineage) | Llama-3.3-70B-Instruct | vLLM TPU or GPU |
| Dean / Verifier | Qwen3-32B-Instruct | vLLM TPU or GPU |
| Auditor | Llama-3.3-70B-Instruct | vLLM GPU |
| Translation | IndicTrans2 | HF Transformers GPU |
| LaTeX/Code specialist | Qwen2.5-Coder-32B-Instruct | vLLM GPU |
| Student (pilot) | BrahmX-Sci-3.5B dense | MaxText TPU / vLLM serving |
| Student (primary) | BrahmX-Sci-8B dense | MaxText TPU / vLLM serving |
| Student (future) | BrahmX-Sci-35B dense | MaxText TPU |

## 6. Specialized Scientific Lanes

| Lane | Input | Key Output | Primary Validator |
|------|-------|------------|-------------------|
| Structured Simulation JSON | PINNs/FEM/PDE JSON | Scientific QA, SFT, counterfactuals | JSON schema + numeric checks |
| Bounded Multi-LLM Dialogue | Gold/silver chunks | Dialogues, tutoring transcripts | Topic-bound + hallucination drift |
| LaTeX Artifact | Scientific chunks | LaTeX SFT, compiler-repair pairs | XeLaTeX sandbox compile gate |
| Code / Tool-Use | APIs, notebooks | Executable code pairs, tool trajectories | nsjail sandbox + tests |
| Multilingual Sovereign | Accepted docs | Indic pretraining, bilingual QA | Script policy + glossary + alignment |
| Reasoning-Budget | Reasoning-rich tasks | Budgeted-thinking SFT, public traces | Trace truncation + answer preservation |
| Curriculum Difficulty | Gold exemplars | Curriculum ladders, difficulty-tagged SFT | Dup detection + solvability |

## 7. Deployment Topology

```
┌─────────────────────────────────────────────────────────┐
│                    GKE / GCE Cluster                     │
│                                                          │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────────┐ │
│  │ vLLM TPU    │  │ vLLM GPU    │  │ MaxText Training │ │
│  │ Pool        │  │ Pool        │  │ Pods (TPU)       │ │
│  │ (bulk gen)  │  │ (frontier)  │  │ (CPT/SFT/RL)    │ │
│  └─────────────┘  └─────────────┘  └──────────────────┘ │
│                                                          │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────────┐ │
│  │ Kubeflow    │  │ Batch       │  │ Object Store     │ │
│  │ Pipelines   │  │ Workers     │  │ (GCS buckets)    │ │
│  └─────────────┘  └─────────────┘  └──────────────────┘ │
│                                                          │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────────┐ │
│  │ Provenance  │  │ Human Review│  │ Monitoring /     │ │
│  │ Registry    │  │ UI          │  │ Observability    │ │
│  └─────────────┘  └─────────────┘  └──────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

## 8. Data & Control Flow

1. **Ingest**: Sources → normalization → license tagging → tokenizer preview → Source Graph (GCS)
2. **Gold Generate**: Source atoms → CFT → Evidence Pack → Prompt → Teacher → Claim Ledger → Dean → Auditor → Gold Bundle (GCS)
3. **Silver Generate**: Source Graph + Gold exemplars → Seed/Persona → Lanes → Validators → Silver Bundle (GCS)
4. **Canonicalize**: Gold + Silver → Fan-out (pretraining/SFT/QA/traces/etc.) → Tokenizer-safe text
5. **Assemble**: Canonical data → Dedup → Decontam → Mixture → ArrayRecord/Parquet → Release Manifest
6. **Train**: Static corpora → MaxText CPT → SFT → Preference → RL → Recovery → Checkpoint
7. **Deploy**: Checkpoint → Eval → Package → vLLM serving

## 9. Environment Separation

| Environment | Purpose | Infra |
|-------------|---------|-------|
| **dev** | Local development, unit tests | Docker Compose, mock services |
| **staging** | Integration testing, pipeline validation | Small GKE cluster, subset TPU |
| **prod** | Full-scale generation and training | Production GKE, full TPU allocation |
