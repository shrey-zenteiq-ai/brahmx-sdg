# BrahmX SDG — Sovereign Science-Focused Synthetic Data Generation Platform

BrahmX SDG is ZenteiQ Aitech's **Data Factory** for generating verified gold training artifacts.
It ingests scientific source documents, builds an evidence-grounded knowledge base, and produces
**Gold Record Bundles** — structured training examples that carry full provenance: claim ledgers,
citation trails, Dean verification scores, and Auditor reports.

The pipeline is inference-backend agnostic. Today it runs against the OpenAI API. When TPU vLLM
pods are ready the swap is a single file edit (`configs/routing/models.yaml`).

---

## Table of Contents

1. [Repository Structure](#repository-structure)
2. [Concepts](#concepts)
3. [Pipeline Data Flow](#pipeline-data-flow)
   - [Full Path — with Dean LLM Scoring](#full-path--with-dean-llm-scoring)
   - [Fast Path — without Dean LLM Scoring](#fast-path--without-dean-llm-scoring)
4. [Stage-by-Stage Breakdown](#stage-by-stage-breakdown)
5. [Installation](#installation)
6. [Configuration](#configuration)
7. [Running the Pipeline](#running-the-pipeline)
   - [Step 1 — Ingest Source Documents](#step-1--ingest-source-documents)
   - [Step 2 — Generate a Gold Bundle (single spec)](#step-2--generate-a-gold-bundle-single-spec)
   - [Step 3 — Batch Generation](#step-3--batch-generation)
   - [Step 4 — Inspect Bundles](#step-4--inspect-bundles)
   - [Step 5 — Assemble JSONL Corpus](#step-5--assemble-jsonl-corpus)
8. [Output File Formats](#output-file-formats)
   - [Gold Record Bundle](#gold-record-bundle-gold-json)
   - [Dataset Slice](#dataset-slice-json)
   - [JSONL Corpus](#jsonl-corpus)
9. [Task Spec Schema](#task-spec-schema)
10. [Config Reference](#config-reference)
11. [Switching to TPU vLLM](#switching-to-tpu-vllm)
12. [API Cost Guide](#api-cost-guide)
13. [Architecture Principles](#architecture-principles)
14. [Running on Vertex AI Pipelines](#running-on-vertex-ai-pipelines)
    - [How it fits together](#how-it-fits-together)
    - [Vertex Pipeline DAG](#vertex-pipeline-dag)
    - [Prerequisites](#prerequisites)
    - [Step 1 — GCP Setup](#step-1--gcp-setup-iam-apis-bucket-secrets)
    - [Step 2 — Build and Push the Component Image](#step-2--build-and-push-the-component-image)
    - [Step 3 — Upload Assets to GCS](#step-3--upload-assets-to-gcs)
    - [Step 4 — Set Environment Variables](#step-4--set-environment-variables)
    - [Step 5 — Compile the Pipeline](#step-5--compile-the-pipeline)
    - [Step 6 — Submit to Vertex AI](#step-6--submit-to-vertex-ai)
    - [Pipeline Parameters Reference](#pipeline-parameters-reference)
    - [Monitoring](#monitoring)
    - [Common Errors](#common-errors)

---

## Repository Structure

```
brahmx-sdg-platform/
├── pyproject.toml                     # Package + dependencies
├── .env.example                       # Environment variable template
├── configs/
│   ├── routing/
│   │   └── models.yaml                # Model endpoints + routing rules
│   ├── gates/
│   │   └── gold_gates.yaml            # Acceptance thresholds (gold/silver/bronze)
│   ├── specs/                         # Task spec JSON files
│   │   ├── thermodynamics_first_law.json
│   │   ├── cell_biology_mitosis.json
│   │   └── linear_algebra_eigenvalues.json
│   └── corpus_manifest.yaml           # Corpus assembly manifest
├── data/
│   ├── kb/
│   │   └── chunks/                    # Pre-chunked knowledge base atoms (JSON)
│   ├── gold/
│   │   ├── GOLD-*.json                # Gold Record Bundles
│   │   └── slices/                    # Individual training slice JSON files
│   └── corpus/
│       └── v0.1/                      # Assembled JSONL corpus
├── pipelines/
│   └── vertex/
│       ├── Dockerfile                 # Component image (brahmx_sdg pre-installed)
│       ├── components.py              # KFP component definitions (GCS I/O wrappers)
│       ├── pipeline.py                # Pipeline DAG + compile + submit
│       ├── requirements.txt           # Host-side deps (kfp, google-cloud-aiplatform)
│       ├── setup_iam.sh               # GCP IAM + API enablement script
│       ├── build_and_push.sh          # Build + push Docker image to Artifact Registry
│       └── .env.example               # Vertex-specific environment variables
├── scripts/
│   └── upload_to_gcs.sh              # Upload specs / KB / configs to GCS
└── src/brahmx_sdg/
    ├── ingestion/                     # Source document ingestion
    ├── kb/                            # Knowledge base retrieval (BM25)
    ├── prompt/                        # Deterministic prompt construction
    ├── routing/                       # Model endpoint routing
    ├── generation/                    # Gold generator orchestration
    ├── verification/                  # Dean + Auditor + citation + symbolic checks
    ├── packaging/                     # Bundle assembly + slice emission (8 types)
    ├── corpus/                        # JSONL corpus assembly
    └── cli.py                         # CLI entry point
```

---

## Concepts

| Term | Description |
|------|-------------|
| **Task Spec** | JSON file describing what to generate: section ID, objectives, required claims, domain, difficulty |
| **Knowledge Base (KB)** | Set of pre-chunked source document atoms. Retrieval is BM25 against the task's objectives |
| **Evidence Pack** | Top-k KB chunks retrieved for a specific task. Forms the citation context for teachers |
| **Teacher A / B / C** | Three independent LLM roles. A = bulk (gpt-4o-mini), B = frontier (gpt-4o), C = alternate perspective (gpt-4o-mini). Each called separately to ensure lineage diversity |
| **Claim Ledger** | JSON array embedded in every teacher response. Each entry has `claim_id`, `claim_text`, `claim_type`, `verifiability`, `supporting_citations` (e.g. `["[1]","[3]"]`), and `is_critical` |
| **Dean** | Verification engine: citation coverage/precision/specificity (BM25 + TF-IDF), CJK leakage check, symbolic/numeric validation (SymPy + Pint), and optional LLM rubric scoring |
| **Auditor** | Independent LLM review with **no** evidence pack (FM-09 bias mitigation). Evaluates only using its own knowledge |
| **Repair Loop** | On Dean/Auditor failure, a multi-turn repair prompt is sent back to **the same teacher** that produced the failing candidate (identified via `generation_metadata.role`). Max 2 rounds |
| **Gold Record Bundle** | Atomic provenance unit: task spec + evidence pack + 3 candidates + Dean scores + Auditor report + 8 training slices |
| **Slice** | One training example in a specific format (explanation, QA, quiz, etc.) derived from the selected candidate |
| **JSONL Corpus** | Flat file of slices, one per line, assembled from many bundles. One file per slice type + `all_slices.jsonl` |

---

## Pipeline Data Flow

### Full Path — with Dean LLM Scoring

This is the default path. Six LLM API calls are made per task spec (3 teachers + 1 Dean rubric + 1 Auditor + possibly 1 repair).

```
Task Spec JSON
      │
      ▼
┌─────────────────────────────┐
│  EvidencePackBuilder        │  BM25 search over data/kb/chunks/*.json
│  (BM25 retrieval, top-k=6)  │  Query = section_id + objectives tokens
└──────────────┬──────────────┘
               │  evidence_pack  (citations [1]...[k])
               ▼
┌─────────────────────────────┐
│  PromptConstructor          │  Deterministic. Injects:
│                             │    - system role + JSON schema
└──────────────┬──────────────┘    - objectives list
               │                   - evidence sources as [1] text...
               │                   - required claims
               │  system_prompt + user_prompt
               ▼
      ┌────────┴─────────┐
      │                  │
      ▼                  ▼                 ▼
┌──────────┐      ┌──────────┐     ┌──────────┐
│ Teacher A│      │ Teacher B│     │ Teacher C│   3 independent LLM calls
│gpt-4o-mini│    │  gpt-4o  │     │gpt-4o-mini│  with different temperatures
│ T=0.7   │      │ T=0.8    │     │ T=0.9    │
└────┬─────┘      └────┬─────┘     └────┬─────┘
     │                 │                │
     └────────┬────────┘                │
              └────────────┬────────────┘
                           │  3 × {content, claim_ledger}
                           ▼
              ┌────────────────────────┐
              │  Dean  (for each cand) │
              │  1. CitationChecker    │  BM25 + TF-IDF: coverage, precision,
              │     (no LLM call)      │  specificity vs evidence pack
              │  2. CJK leakage check  │  Regex scan for Han/Kana/Hangul chars
              │  3. SymbolicValidator  │  SymPy equation parse / Pint unit check
              │     (no LLM call)      │
              │  4. LLM rubric score   │  gpt-4o-mini (T=0.1) rates 0-100
              │     (1 LLM call)       │  on 5 axes: accuracy, clarity, depth,
              │                        │  citations, pedagogy
              └────────────┬───────────┘
                           │  dean_score per candidate
                           │  → select best passing candidate
                           │  (or best-scoring for repair if all fail)
                           ▼
              ┌────────────────────────┐
              │  Repair Loop (if fail) │
              │  Max 2 rounds          │  Multi-turn prompt sent to SAME teacher
              │                        │  with aggregated failure reasons
              └────────────┬───────────┘
                           │  selected_candidate_id
                           ▼
              ┌────────────────────────┐
              │  Auditor               │  gpt-4o-mini (T=0.2)
              │  (NO evidence pack)    │  Receives: spec + content + claim summary
              │  FM-09 mitigation      │  Returns: status, findings, severity,
              │                        │  override_dean, escalate_to_human
              └────────────┬───────────┘
                           │  auditor_report
                           ▼
              ┌────────────────────────┐
              │  BundleAssembler       │  Assembles GoldRecordBundle:
              │                        │    record_id (SHA-256 prefix)
              │                        │    task_spec, evidence_pack
              │                        │    candidates[], claim_ledger
              │                        │    dean_report, auditor_report
              └────────────┬───────────┘
                           │  bundle.json  →  data/gold/GOLD-{id}.json
                           ▼
              ┌────────────────────────┐
              │  SliceEmitter          │  8 slice types from selected candidate
              └────────────┬───────────┘
                           │
         ┌─────────────────┼──────────────────┐
         ▼                 ▼                  ▼
  explanation_      qa_with_citation    quiz_generation
  generation        term_extraction     misconception_
  claim_            summary_            structured_
  verification      generation          outline
         │
         └──► data/gold/slices/{task_id}_{type}_{slice_id}.json
```

### Fast Path — without Dean LLM Scoring

Add `--no-llm-dean` to skip the LLM rubric scoring step inside Dean. This removes one API call per
candidate and makes the pipeline significantly faster and cheaper. Citation coverage, CJK leakage,
and symbolic validation still run — only the subjective rubric scoring is skipped.

```
Task Spec JSON
      │
      ▼
┌─────────────────────────────┐
│  EvidencePackBuilder        │  (same as full path)
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  PromptConstructor          │  (same as full path)
└──────────────┬──────────────┘
               │
      ┌────────┴─────────┐
      ▼                  ▼                 ▼
┌──────────┐      ┌──────────┐     ┌──────────┐
│ Teacher A│      │ Teacher B│     │ Teacher C│
└────┬─────┘      └────┬─────┘     └────┬─────┘
     └────────┬─────────┘────────────────┘
              │
              ▼
┌────────────────────────┐
│  Dean  (for each cand) │
│  1. CitationChecker    │  BM25/TF-IDF (no LLM call)
│  2. CJK leakage check  │  Regex (no LLM call)
│  3. SymbolicValidator  │  SymPy/Pint (no LLM call)
│  ~~LLM rubric skipped~~│  <-- removed by --no-llm-dean
└────────────┬───────────┘
             │  dean_score (citation + symbolic only)
             ▼
┌────────────────────────┐
│  Repair Loop (if fail) │  (same as full path)
└────────────┬───────────┘
             │
             ▼
┌────────────────────────┐
│  Auditor               │  (same as full path — always runs)
└────────────┬───────────┘
             │
             ▼
┌────────────────────────┐
│  BundleAssembler       │
└────────────┬───────────┘
             │
             ▼
┌────────────────────────┐
│  SliceEmitter (8 types)│
└────────────────────────┘
```

**API calls comparison:**

| Step | Full path | Fast path (`--no-llm-dean`) |
|------|-----------|------------------------------|
| Teacher A | 1 | 1 |
| Teacher B | 1 | 1 |
| Teacher C | 1 | 1 |
| Dean LLM rubric (×3 candidates) | up to 3 | 0 |
| Auditor | 1 | 1 |
| Repair rounds (if needed) | 0–2 | 0–2 |
| **Typical total** | **6–9** | **4–6** |

---

## Stage-by-Stage Breakdown

### Stage 1 — Evidence Pack Builder

**Input:** task spec (`section_id`, `objectives[]`, `domain`)

**What happens:**
- Loads all JSON files from `data/kb/chunks/`
- Tokenizes each chunk using whitespace splitting
- Builds a BM25Okapi corpus index
- Forms a query string from `section_id` + all objective text
- Retrieves top-k chunks (default k=6) ranked by BM25 score
- Numbers the retrieved chunks as `[1]`, `[2]`, ..., `[k]`

**Output:** `EvidencePack` — list of `EvidenceSource` objects each with `citation_key`, `text`, `chunk_id`, `source_file`, `domain`

If the KB has no chunks for the domain, generation proceeds with an empty evidence pack (teachers generate from parametric knowledge only, which will reduce citation scores).

---

### Stage 2 — Prompt Constructor

**Input:** task spec + evidence pack

**What happens:**
- Assembles a fixed-format system prompt that includes:
  - Role description ("You are an expert science educator...")
  - The required JSON response schema (`{"content": "...", "claim_ledger": [...]}`)
  - Evidence sources formatted as `[1] <text>`, `[2] <text>`, ...
  - Response constraints (from `task_spec.constraints`)
- Assembles a user prompt with:
  - Section ID and domain
  - Full objectives list
  - Required claims that MUST appear

**Output:** `(system_prompt: str, user_prompt: str)` — identical for all three teachers.

---

### Stage 3 — Three-Teacher Generation

**Input:** system prompt + user prompt

**What happens (per teacher):**
- Calls the model via the HTTP router using `response_format={"type": "json_object"}`
- Each teacher uses a different model and temperature to ensure response diversity:
  - **Teacher A** (`openai-teacher-a`, gpt-4o-mini, T=0.7): bulk evidence-grounded generation
  - **Teacher B** (`openai-teacher-b`, gpt-4o, T=0.8): frontier-quality, difficult/nuanced content
  - **Teacher C** (`openai-teacher-c`, gpt-4o-mini, T=0.9): alternate perspective, different framing
- Parses the JSON response to extract `content` (the generated text) and `claim_ledger` (array of claims)
- Tags each candidate with `generation_metadata.role` (`"teacher_a"`, `"teacher_b"`, `"teacher_c"`)

**Claim ledger format per entry:**
```json
{
  "claim_id": "CLM-001",
  "claim_text": "The First Law states ΔU = Q - W",
  "claim_type": "factual",
  "verifiability": "citation_supported",
  "supporting_citations": ["[1]", "[3]"],
  "is_critical": true
}
```

**Output:** 3 `TeacherCandidate` objects

---

### Stage 4 — Dean Verification

**Input:** candidate + evidence pack + task spec

Dean runs 4 checks on each candidate. The first 3 are deterministic (no LLM calls):

**4a. Citation Coverage/Precision (CitationChecker)**
- Builds a BM25 index from evidence pack chunks keyed as `[1]`, `[2]`, etc.
- For each claim in the claim ledger, checks whether its `supporting_citations` actually appear in the evidence pack
- Computes `citation_coverage` (fraction of claims with at least one valid citation), `citation_precision` (fraction of cited sources that are genuinely relevant to the claim), and `citation_specificity`
- Thresholds from `configs/gates/gold_gates.yaml`

**4b. CJK Leakage Check**
- Regex scan for Chinese/Japanese/Korean characters (Han, Hiragana, Katakana, Hangul unicode ranges)
- Hard fail if any CJK characters are found in an English-language task

**4c. Symbolic/Numeric Validation (SymbolicNumericValidator)**
- Runs on `candidate.tool_checks_required` items
- Parses mathematical equations with SymPy
- Validates physical units with Pint
- Flags equations that fail to parse or produce unit mismatches

**4d. LLM Rubric Scoring** (skipped with `--no-llm-dean`)
- Calls `openai-dean` (gpt-4o-mini, T=0.1) with the candidate content
- Dean rates 0-100 on 5 axes: factual accuracy, clarity, depth, citation quality, pedagogical value
- Returns `(score: float 0-1, suggestions: list[str])`
- Failure to call Dean returns `None` (no penalty applied)

**Selection logic:**
- If ≥1 candidate passes all gates → select highest-scoring passing candidate
- If all fail → select best-scoring candidate, mark it as `failed_candidate` → trigger repair loop

**Output:** `DeanReport` per candidate with `passed: bool`, `scores: dict`, `gate_failures: list[str]`

---

### Stage 5 — Repair Loop

**Triggered when:** all 3 candidates fail Dean gates

**What happens:**
- Identifies which candidate scored best (least failures)
- Reads `generation_metadata["role"]` to find the originating teacher model
- Builds a multi-turn repair conversation:
  - Original system prompt
  - Original user prompt
  - Assistant turn: the failing candidate's content
  - User turn: failure summary listing specific gate failures and Dean/Auditor suggestions
- Calls the **same teacher** that produced the failing candidate
- Runs the full Dean verification loop again on the repaired candidate
- Repeats up to **2 rounds**

**Output:** Repaired `TeacherCandidate` or original best candidate if repair doesn't help

---

### Stage 6 — Auditor Review

**Input:** task spec + selected candidate content + claim ledger summary (NO evidence pack)

**What happens:**
- Calls `openai-auditor` (gpt-4o-mini, T=0.2)
- System prompt explicitly states the Auditor has **no access to source documents** (FM-09 mitigation against confirmation bias)
- Auditor evaluates using only its own parametric knowledge:
  - Is the content factually accurate?
  - Are any claims misleading or oversimplified?
  - Does the content meet the stated objectives?
- Returns structured JSON:
  ```json
  {
    "status": "PASS",
    "findings": ["Minor: sign convention not explained for W"],
    "severity": "low",
    "override_dean": false,
    "escalate_to_human": false
  }
  ```
- Fallback: if LLM call fails, a heuristic review checks length, claim ledger presence, and keyword coverage

**Output:** `AuditorReport`

---

### Stage 7 — Bundle Assembly

**Input:** task spec + evidence pack + 3 candidates + Dean reports + Auditor report

**What happens:**
- Assigns `record_id` = first 12 hex chars of `SHA-256(section_id + timestamp)`
- Merges all claim ledger entries from all candidates into a unified `claim_ledger` at bundle level
- Sets `selected_candidate_id` to the passing/best-repaired candidate
- Writes `data/gold/GOLD-{record_id}.json`

**Output:** `GoldRecordBundle` JSON file

---

### Stage 8 — Slice Emission

**Input:** gold bundle (selected candidate's content + claim ledger + evidence pack + task spec)

**What happens:** Generates 8 training slice types from the selected candidate:

| Slice type | Description |
|------------|-------------|
| `explanation_generation` | Full explanation text as SFT completion |
| `qa_with_citation` | Question–answer pairs where answers cite evidence sources |
| `quiz_generation` | Multiple-choice questions derived from critical claims |
| `term_extraction` | Key term → definition pairs from the content |
| `misconception_correction` | Common misconception → corrected explanation |
| `claim_verification` | Individual claims formatted as verify/reject tasks |
| `summary_generation` | Condensed summary of the explanation |
| `structured_outline` | Hierarchical outline of topics covered |

Each slice is written to `data/gold/slices/{task_id}_{slice_type}_{slice_id}.json`

**Output:** Up to 8 slice files per bundle

---

### Stage 9 — Corpus Assembly

**Input:** all `GOLD-*.json` files in `data/gold/`

**What happens:**
- Loads and validates each bundle as a `GoldRecordBundle` Pydantic object
- Re-runs `SliceEmitter.emit()` on each bundle to collect all slices
- Content-hash deduplication (SHA-256 of slice content) removes exact duplicates
- Writes one `.jsonl` file per slice type + `all_slices.jsonl`

**Output:** `data/corpus/v0.1/*.jsonl`

---

## Installation

```bash
# 1. Clone / enter repo
cd /path/to/brahmx-sdg-platform

# 2. Install (Python 3.11+)
pip install -e ".[dev]" --break-system-packages

# 3. Set API key
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...

# 4. Load env vars
export $(grep -v '^#' .env | xargs)
# or: source <(grep -v '^#' .env | sed 's/^/export /')
```

**Dependencies (automatically installed):**
- `typer`, `rich` — CLI
- `pydantic>=2` — data models
- `openai`, `httpx` — API calls
- `rank-bm25` — knowledge base retrieval
- `sympy`, `pint` — symbolic/numeric validation
- `python-dotenv` — env var loading

---

## Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes | — | OpenAI API key |
| `BRAHMX_ENV` | No | `dev` | Environment tag |
| `BRAHMX_GCS_BUCKET` | No | (empty) | GCS bucket for prod artifact storage |
| `BRAHMX_GCS_PREFIX` | No | `brahmx-sdg` | Path prefix inside GCS bucket |
| `BRAHMX_LOG_LEVEL` | No | `INFO` | Log level |

### Model Routing — `configs/routing/models.yaml`

Each model entry:
```yaml
- model_id: "openai-teacher-b"      # Internal identifier
  model_name: "gpt-4o"             # Model name passed to API
  base_url: "https://api.openai.com"
  api_key_env: "OPENAI_API_KEY"    # Env var to read for Bearer token
  roles: ["teacher_b"]             # Roles this model serves
  temperature: 0.8                 # Per-model temperature
  enabled: true
```

When `api_key_env` is set, the router automatically injects `Authorization: Bearer <value>`.
When empty (vLLM), no auth header is sent.

### Gold Gates — `configs/gates/gold_gates.yaml`

Three tiers: `gold`, `silver`, `bronze`. Each tier lists required gate thresholds.
The pipeline uses the `gold` tier by default. Key thresholds:

| Gate | Gold threshold |
|------|---------------|
| `citation_coverage` | ≥ 0.98 |
| `citation_precision` | ≥ 0.95 |
| `citation_specificity` | ≥ 0.85 |
| `objective_coverage` | ≥ 0.95 |
| `tool_checks_pass` | = 1.00 |
| `auditor_status` | PASS |

---

## Running the Pipeline

### Step 1 — Ingest Source Documents

Skip this step if `data/kb/chunks/` already has JSON files for your domain.

```bash
# Ingest a single file
brahmx-sdg ingest run \
  --source /path/to/your/document.pdf \
  --domain thermodynamics \
  --kb-dir data/kb \
  --chunk-size 2048 \
  --license cc-by-4.0

# Ingest an entire directory
brahmx-sdg ingest run \
  --source /path/to/textbook/chapters/ \
  --domain physics \
  --kb-dir data/kb
```

**Supported source formats:** `.txt`, `.md`, `.py`, `.tex`, `.json`, `.jsonl`, `.pdf`

**Output:** `data/kb/chunks/{stem}.json` — array of chunk atoms with `chunk_id`, `text`, `domain`, `source_file`

To pre-populate the KB without running ingestion, create JSON files manually in `data/kb/chunks/`:
```json
[
  {
    "chunk_id": "thermo-001",
    "text": "The First Law of Thermodynamics states that...",
    "domain": "thermodynamics",
    "source_file": "textbook.pdf",
    "license": "cc-by-4.0"
  }
]
```

---

### Step 2 — Generate a Gold Bundle (single spec)

```bash
# Full pipeline (default — includes Dean LLM rubric scoring)
brahmx-sdg gold run \
  --spec configs/specs/thermodynamics_first_law.json \
  --kb-path data/kb \
  --output-dir data/gold

# Fast path — skip Dean LLM rubric (cheaper, faster, citation/symbolic checks still run)
brahmx-sdg gold run \
  --spec configs/specs/thermodynamics_first_law.json \
  --kb-path data/kb \
  --output-dir data/gold \
  --no-llm-dean

# Custom routing config
brahmx-sdg gold run \
  --spec configs/specs/cell_biology_mitosis.json \
  --routing-config configs/routing/models.yaml \
  --output-dir data/gold
```

**Successful output:**
```
✓ Gold bundle: GOLD-386b9437f254
  Bundle:  data/gold/GOLD-386b9437f254.json
  Slices:  data/gold/slices/
```

**Failed output:**
```
✗ Generation blocked: all candidates failed dean gates after 2 repair rounds
```

---

### Step 3 — Batch Generation

```bash
# Generate bundles for all *.json specs in configs/specs/
brahmx-sdg gold batch-run \
  --specs-dir configs/specs/ \
  --kb-path data/kb \
  --output-dir data/gold \
  --no-llm-dean

# Stop immediately on first failure
brahmx-sdg gold batch-run \
  --specs-dir configs/specs/ \
  --stop-on-failure
```

Prints a Rich summary table:
```
                  Batch Gold Generation — 3 specs
┌─────────────────────────────────────┬────────┬──────────────────────┐
│ Spec                                │ Result │ Bundle ID / Reason   │
├─────────────────────────────────────┼────────┼──────────────────────┤
│ thermodynamics_first_law            │  PASS  │ GOLD-386b9437f254    │
│ cell_biology_mitosis                │  PASS  │ GOLD-7c1a9e20f381    │
│ linear_algebra_eigenvalues          │  PASS  │ GOLD-b3d02e88c1f9    │
└─────────────────────────────────────┴────────┴──────────────────────┘
3 succeeded, 0 failed out of 3 specs.
```

---

### Step 4 — Inspect Bundles

```bash
# Summary table of all gold bundles
brahmx-sdg corpus stats --gold-dir data/gold
```

Output:
```
                 Gold Bundles in data/gold
┌──────────────────────┬──────────────────────┬───────────────┬────────────┬───────┬─────────┐
│ Bundle ID            │ Section ID           │ Domain        │ Candidates │ Claims│ Auditor │
├──────────────────────┼──────────────────────┼───────────────┼────────────┼───────┼─────────┤
│ GOLD-386b9437f254    │ PHYS-C03-M02-S04     │ thermodynamics│     3      │   27  │  PASS   │
└──────────────────────┴──────────────────────┴───────────────┴────────────┴───────┴─────────┘

Total: 1 bundles
Slices: 17 files in data/gold/slices
```

---

### Step 5 — Assemble JSONL Corpus

```bash
# Assemble all gold bundles into JSONL corpus
brahmx-sdg corpus assemble \
  --manifest configs/corpus_manifest.yaml \
  --output data/corpus
```

Output:
```
✓ Corpus v0.1 — 136 examples
  Output:   data/corpus
  Manifest: configs/corpus_manifest.yaml

        Slices by type
┌────────────────────────────┬───────┐
│ Task type                  │ Count │
├────────────────────────────┼───────┤
│ claim_verification         │    17 │
│ explanation_generation     │     1 │
│ misconception_correction   │     9 │
│ qa_with_citation           │     9 │
│ quiz_generation            │     9 │
│ structured_outline         │     1 │
│ summary_generation         │     1 │
│ term_extraction            │    17 │
└────────────────────────────┴───────┘
```

JSONL files are written to `data/corpus/v0.1/`:
```
data/corpus/v0.1/
├── explanation_generation.jsonl
├── qa_with_citation.jsonl
├── quiz_generation.jsonl
├── term_extraction.jsonl
├── misconception_correction.jsonl
├── claim_verification.jsonl
├── summary_generation.jsonl
├── structured_outline.jsonl
└── all_slices.jsonl
```

---

## Output File Formats

### Gold Record Bundle (`GOLD-*.json`)

```json
{
  "record_id": "GOLD-386b9437f254",
  "schema_version": "1.0.0",
  "created_at": "2026-03-22T14:31:07Z",

  "task_spec": {
    "task_id": "TASK-thermo-001",
    "section_id": "PHYS-C03-M02-S04",
    "task_type": "gold_explanation",
    "domain": "thermodynamics",
    "language": "en",
    "difficulty": "medium",
    "objectives": ["Explain the First Law...", "Derive ΔU = Q - W..."],
    "required_claims": [
      {"claim_id": "CFT-PHYS-0042", "statement": "ΔU = Q - W ..."}
    ]
  },

  "evidence_pack": {
    "sources": [
      {
        "citation_key": "[1]",
        "text": "The First Law of Thermodynamics states...",
        "chunk_id": "thermo-001",
        "source_file": "thermodynamics.json",
        "domain": "thermodynamics"
      }
    ]
  },

  "candidates": [
    {
      "candidate_id": "cand-a-xxxx",
      "content": "The First Law of Thermodynamics...",
      "claim_ledger": [
        {
          "claim_id": "CLM-001",
          "claim_text": "The First Law states ΔU = Q - W",
          "claim_type": "factual",
          "verifiability": "citation_supported",
          "supporting_citations": ["[1]"],
          "is_critical": true
        }
      ],
      "generation_metadata": {
        "model_id": "openai-teacher-a",
        "model_name": "gpt-4o-mini",
        "role": "teacher_a",
        "temperature": 0.7
      }
    },
    { "candidate_id": "cand-b-xxxx", "generation_metadata": {"role": "teacher_b"}, "..." : "..." },
    { "candidate_id": "cand-c-xxxx", "generation_metadata": {"role": "teacher_c"}, "..." : "..." }
  ],

  "selected_candidate_id": "cand-b-xxxx",

  "claim_ledger": {
    "claims": [ ]
  },

  "dean_report": {
    "passed": true,
    "selected_candidate_id": "cand-b-xxxx",
    "scores": {
      "cand-b-xxxx": {
        "citation_coverage": 1.0,
        "citation_precision": 0.97,
        "llm_rubric_score": 0.88
      }
    },
    "gate_failures": []
  },

  "auditor_report": {
    "status": "PASS",
    "findings": [],
    "severity": "none",
    "override_dean": false,
    "escalate_to_human": false
  },

  "slices": [
    "data/gold/slices/TASK-thermo-001_explanation_generation_s1a2b3.json",
    "data/gold/slices/TASK-thermo-001_qa_with_citation_s4c5d6.json"
  ]
}
```

---

### Dataset Slice (JSON)

Each slice file in `data/gold/slices/`:

```json
{
  "slice_id": "s1a2b3c4",
  "record_id": "GOLD-386b9437f254",
  "task_id": "TASK-thermo-001",
  "task_type": "qa_with_citation",
  "domain": "thermodynamics",
  "language": "en",

  "input": "What does the First Law of Thermodynamics state?",
  "output": "The First Law states that energy is conserved: ΔU = Q - W, where ΔU is the change in internal energy, Q is heat added, and W is work done by the system. [1]",

  "metadata": {
    "source_candidate_id": "cand-b-xxxx",
    "source_model": "gpt-4o",
    "section_id": "PHYS-C03-M02-S04",
    "difficulty": "medium",
    "citations_used": ["[1]"]
  }
}
```

---

### JSONL Corpus

Each line in `all_slices.jsonl` is one slice JSON object (same structure as above, no line breaks within each record):

```
{"slice_id": "s1a2b3c4", "task_type": "qa_with_citation", "input": "...", "output": "...", ...}
{"slice_id": "s5e6f7g8", "task_type": "explanation_generation", "input": "...", "output": "...", ...}
```

---

## Task Spec Schema

```json
{
  "task_id": "TASK-<domain>-<number>",
  "section_id": "SUBJ-C<chapter>-M<module>-S<section>",
  "task_type": "gold_explanation",
  "domain": "thermodynamics",
  "language": "en",
  "difficulty": "easy | medium | hard",

  "objectives": [
    "One sentence describing what the generated content must cover"
  ],

  "required_claims": [
    {
      "claim_id": "CFT-<SUBJ>-<number>",
      "statement": "Exact claim that must appear in the content"
    }
  ],

  "constraints": {
    "max_word_count": 1200,
    "include_worked_example": true,
    "include_exercises": 2
  },

  "metadata": {
    "curriculum_module": "Classical Thermodynamics",
    "target_student": "undergraduate physics year 1"
  }
}
```

See `configs/specs/` for working examples across thermodynamics, cell biology, and linear algebra.

---

## Config Reference

### `configs/routing/models.yaml`

Controls which LLM is used for each role. Full field reference:

```yaml
models:
  - model_id: string          # Internal ID, referenced in code
    model_name: string        # Passed directly to the API (e.g. "gpt-4o")
    runtime: string           # Informational only ("vllm_tpu", "vllm_gpu", etc.)
    base_url: string          # API base URL
    api_key_env: string       # Env var holding the Bearer token (empty = no auth)
    roles: [string]           # Which roles this model serves
    workload_classes: [string]# "bulk", "frontier", "audit"
    max_context_length: int
    validated_on_tpu: bool
    cost_per_1k_tokens: float # Informational
    avg_latency_ms: int       # Informational
    quality_score: float      # Informational
    temperature: float        # Per-model default temperature
    enabled: bool

routing:
  default_workload: "bulk"
  fallback_enabled: true
  max_fallback_attempts: 2
```

### `configs/gates/gold_gates.yaml`

All acceptance thresholds in one place. Bump `pipeline_version` after any change.

### `configs/corpus_manifest.yaml`

```yaml
version: "v0.1"
gold_paths:
  - "data/gold"             # Directories to scan for GOLD-*.json
silver_paths: []
mixture:
  task_types: []            # Empty = include all 8 slice types
  max_per_type: 0           # 0 = unlimited
format: "jsonl"
```

---

## Switching to TPU vLLM

When TPU vLLM pods are available, the only file that needs editing is
`configs/routing/models.yaml`. No code changes required.

```yaml
# Replace the OpenAI entries with vLLM pod entries:
models:
  - model_id: "llama4-scout-teacher-a"
    model_name: "meta-llama/Llama-4-Scout"
    runtime: "vllm_tpu"
    base_url: "http://pod0-vllm:8000"   # Pod internal address
    api_key_env: ""                      # Empty = no auth header
    roles: ["teacher_a"]
    workload_classes: ["bulk"]
    max_context_length: 10000000
    temperature: 0.7
    enabled: true

  - model_id: "llama4-maverick-teacher-b"
    model_name: "meta-llama/Llama-4-Maverick"
    runtime: "vllm_tpu"
    base_url: "http://pod1-vllm:8000"
    api_key_env: ""
    roles: ["teacher_b"]
    workload_classes: ["bulk", "frontier"]
    max_context_length: 10000000
    temperature: 0.8
    enabled: true

  # ... repeat for teacher_c, dean, auditor
```

The router reads `api_key_env` at call time. When empty, no `Authorization` header is
injected and the request goes straight to the vLLM OpenAI-compatible endpoint.

**Checklist before swap:**
- [ ] Pods are up and passing `/health`
- [ ] Model names match what vLLM loaded (`/v1/models` endpoint)
- [ ] `base_url` is reachable from the machine running the pipeline
- [ ] `OPENAI_API_KEY` can be unset (or leave it; it's simply unused when `api_key_env: ""`)
- [ ] Run `brahmx-sdg gold run --spec configs/specs/thermodynamics_first_law.json --no-llm-dean` as smoke test

---

## API Cost Guide

Costs are approximate and based on OpenAI pricing as of March 2026.

| Step | Model | Tokens (est.) | Cost per run |
|------|-------|---------------|--------------|
| Teacher A | gpt-4o-mini | ~2,000 in + ~1,500 out | ~$0.0008 |
| Teacher B | gpt-4o | ~2,000 in + ~1,500 out | ~$0.015 |
| Teacher C | gpt-4o-mini | ~2,000 in + ~1,500 out | ~$0.0008 |
| Dean rubric (×3) | gpt-4o-mini | ~1,500 in + ~200 out | ~$0.0006 each |
| Auditor | gpt-4o-mini | ~1,200 in + ~300 out | ~$0.0005 |
| **Full path total** | | | **~$0.019/spec** |
| **Fast path (`--no-llm-dean`)** | | | **~$0.017/spec** |

For batch runs across 100 specs the full path costs ~$1.90. Use `--no-llm-dean` during
development and enable LLM scoring for final gold production runs.

---

## Architecture Principles

| Principle | Rule |
|-----------|------|
| **Evidence-first gold** | Scientific facts, citations, and claim ledgers are first-class objects — not optional metadata |
| **Three-teacher diversity** | Teachers A, B, C are called separately with different models and temperatures to ensure lineage independence |
| **FM-09 bias mitigation** | Auditor receives no evidence pack, preventing anchoring bias from the same sources used in generation |
| **Repair to same teacher** | Repair prompts go back to the originating teacher so the student-teacher distillation chain stays clean |
| **Single swap point** | `configs/routing/models.yaml` is the only file that changes when switching inference backends |
| **Trust over scale** | Claim ledgers, citation checks, symbolic validation, and dean scoring are mandatory — not optional quality checks |
| **Two-plane separation** | Data Factory (this repo) exchanges only versioned artifacts with the Model Factory. Training never depends on live teacher inference |

---

## Running on Vertex AI Pipelines

This section covers running the full pipeline on **Google Cloud Vertex AI Pipelines** — the
managed, serverless KFP execution environment. Each pipeline stage runs in its own container,
scales automatically, and writes all intermediate data to GCS.

### How it fits together

```
Your laptop / CI
  │
  ├── 1. Build Docker image (brahmx_sdg pre-installed) → push to Artifact Registry
  ├── 2. Upload specs + KB chunks + routing config → GCS bucket
  ├── 3. Store OPENAI_API_KEY → Secret Manager
  ├── 4. python pipelines/vertex/pipeline.py        → compiles YAML + submits job
  │
  └── Vertex AI Pipelines runs the DAG:
        [ParallelFor over spec_gcs_uris (max 4 concurrent)]
          generate_gold_bundle  ──►  write_to_human_review_queue (Firestore)
          generate_gold_bundle  ──►  write_to_human_review_queue
          ...
        [After all bundles done]
          assemble_corpus  ──►  JSONL files in GCS
```

Each `generate_gold_bundle` step:
- Downloads the task spec and KB chunks from GCS into `/tmp`
- Runs the full `GoldGenerator` (EvidencePack → 3 teachers → Dean → Auditor → Bundle → Slices)
- Uploads `GOLD-*.json` and `slices/` back to GCS
- Fetches `OPENAI_API_KEY` from Secret Manager at runtime — key never appears in logs

---

### Vertex Pipeline DAG

```
ParallelFor(spec_gcs_uris, max_parallel=4)
┌─────────────────────────────────────────────────┐
│  generate_gold_bundle (spec_uri_0)              │
│    download spec + KB from GCS                  │
│    → EvidencePackBuilder (BM25, local)           │
│    → PromptConstructor (deterministic)           │
│    → Teacher A call  (OpenAI API)               │
│    → Teacher B call  (OpenAI API)               │
│    → Teacher C call  (OpenAI API)               │
│    → Dean (citations + symbolic + LLM rubric)   │
│    → Auditor (independent LLM review)           │
│    → BundleAssembler + SliceEmitter             │
│    upload GOLD-*.json + slices/ to GCS          │
│    output: {"success": true, "bundle_id": "..."}│
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
     write_to_human_review_queue
       download bundle from GCS
       write PENDING_REVIEW doc to Firestore

[Loop repeats for each spec in parallel]
[All loops complete]
         │
         ▼
    assemble_corpus
      download all GOLD-*.json from GCS
      re-emit 8 slice types per bundle
      dedup by content hash
      write *.jsonl files to GCS corpus prefix
```

---

### Prerequisites

| Requirement | Notes |
|-------------|-------|
| GCP project | Billing enabled |
| `gcloud` CLI | Authenticated (`gcloud auth login`) |
| Docker | Running locally for image build |
| Python 3.11+ venv | With `kfp` + `google-cloud-aiplatform` installed |
| GCS bucket | For pipeline root, specs, KB, gold bundles, corpus |
| Artifact Registry repo | Docker type, to push the component image |
| Secret Manager secret | Holds `OPENAI_API_KEY` |
| Firestore database | For human review queue (native mode) |

---

### Step 1 — GCP Setup (IAM, APIs, Bucket, Secrets)

Run the provided script once per project. Edit `PROJECT_ID` at the top first:

```bash
# Review and edit the project ID and bucket name, then:
bash pipelines/vertex/setup_iam.sh
```

The script does:
1. Enables: `aiplatform`, `storage`, `artifactregistry`, `secretmanager`, `firestore` APIs
2. Creates a service account `brahmx-vertex-sa`
3. Grants it the required roles (Vertex AI User, Storage Object Admin, Secret Accessor,
   Firestore User, Artifact Registry Reader, Log Writer)
4. Creates the GCS pipeline bucket
5. Creates the Artifact Registry Docker repository

**Create the OpenAI API key secret manually** (do this once):

```bash
export PROJECT_ID=your-project-id

# Create the secret
echo -n "sk-..." | gcloud secrets create openai-api-key \
    --data-file=- \
    --project=$PROJECT_ID

# Grant the pipeline SA access to it
gcloud secrets add-iam-policy-binding openai-api-key \
    --member="serviceAccount:brahmx-vertex-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor" \
    --project=$PROJECT_ID
```

**Create Firestore database** (one-time, native mode):

```bash
gcloud firestore databases create \
    --location=nam5 \
    --project=$PROJECT_ID
```

---

### Step 2 — Build and Push the Component Image

The Docker image pre-installs `brahmx_sdg` so every KFP component can import it without
a per-component `packages_to_install` download at runtime.

```bash
# Set your project and preferred region
export PROJECT_ID=your-project-id
export LOCATION=us-central1           # must match Vertex AI location

# Build and push (run from repo root — Dockerfile copies src/ and pyproject.toml)
bash pipelines/vertex/build_and_push.sh

# The script prints the image URI and saves it to pipelines/vertex/.image_uri.env
# Source it so pipeline.py picks it up:
source pipelines/vertex/.image_uri.env
echo $BRAHMX_IMAGE_URI
# → us-central1-docker.pkg.dev/your-project-id/brahmx/sdg:latest
```

To rebuild after code changes:

```bash
bash pipelines/vertex/build_and_push.sh v2    # tags as :v2
```

---

### Step 3 — Upload Assets to GCS

Task specs, KB chunks, and the routing config must be in GCS before the pipeline runs.

```bash
export PIPELINE_BUCKET=your-pipeline-bucket

# Upload everything from local data/ and configs/ to GCS
bash scripts/upload_to_gcs.sh
```

The script uploads:
- `configs/specs/*.json`           → `gs://$PIPELINE_BUCKET/specs/`
- `data/kb/chunks/*.json`          → `gs://$PIPELINE_BUCKET/kb/chunks/`
- `configs/routing/models.yaml`    → `gs://$PIPELINE_BUCKET/configs/models.yaml`

To add your own source documents and ingest them into the KB locally first:

```bash
# Ingest locally (creates data/kb/chunks/*.json)
brahmx-sdg ingest run --source /path/to/docs/ --domain physics --kb-dir data/kb

# Then re-upload KB
gsutil -m cp data/kb/chunks/*.json gs://$PIPELINE_BUCKET/kb/chunks/
```

To add new task specs:

```bash
# Copy one of the example specs and edit it
cp configs/specs/thermodynamics_first_law.json configs/specs/my_new_topic.json
# ... edit it ...
gsutil cp configs/specs/my_new_topic.json gs://$PIPELINE_BUCKET/specs/
```

---

### Step 4 — Set Environment Variables

Copy and fill in the environment file:

```bash
cp pipelines/vertex/.env.example pipelines/vertex/.env
```

Edit `pipelines/vertex/.env`:

```bash
# GCP project
PROJECT_ID=your-project-id

# Service account created by setup_iam.sh
SA_NAME=brahmx-vertex-sa
SA_EMAIL=brahmx-vertex-sa@your-project-id.iam.gserviceaccount.com

# Vertex AI region — must match Artifact Registry location
LOCATION=us-central1

# GCS bucket (created by setup_iam.sh)
PIPELINE_BUCKET=your-pipeline-bucket

# Docker image URI (printed by build_and_push.sh)
BRAHMX_IMAGE_URI=us-central1-docker.pkg.dev/your-project-id/brahmx/sdg:latest
```

Load it before running the pipeline:

```bash
source pipelines/vertex/.env
# Or for a one-liner:
export $(grep -v '^#' pipelines/vertex/.env | xargs)
```

---

### Step 5 — Compile the Pipeline

Compilation is a local Python step — it validates the DAG and produces a YAML spec that
Vertex AI executes. No GCP calls are made.

```bash
# Activate the vertex venv first
source .venv-vertex/bin/activate

# Compile only (produces brahmx_sdg_pipeline.yaml in the current directory)
python pipelines/vertex/pipeline.py --compile-only

# Compile to a specific path
python pipelines/vertex/pipeline.py --compile-only --output pipelines/vertex/brahmx_sdg_pipeline.yaml
```

Successful output:

```
Pipeline compiled → brahmx_sdg_pipeline.yaml
```

---

### Step 6 — Submit to Vertex AI

```bash
# Compile and submit in one command
python pipelines/vertex/pipeline.py

# Submit without LLM Dean scoring (faster / cheaper for development runs)
NO_LLM_DEAN=true python pipelines/vertex/pipeline.py

# Disable step caching (forces all steps to re-run even if inputs haven't changed)
python pipelines/vertex/pipeline.py --no-cache
```

The submit step:
1. Calls `aiplatform.init(project=PROJECT_ID, location=LOCATION)`
2. Creates a `PipelineJob` pointing at the compiled YAML in `gs://$PIPELINE_BUCKET/pipeline-root`
3. Authenticates via Application Default Credentials — run `gcloud auth application-default login` first
4. Submits and prints a console URL to monitor the run

**Authenticate ADC before submitting:**

```bash
gcloud auth application-default login
gcloud config set project $PROJECT_ID
```

**Outputs written to GCS after a successful run:**

```
gs://$PIPELINE_BUCKET/
├── gold/
│   ├── GOLD-<id>.json          # Gold Record Bundles
│   └── slices/                 # Individual training slice JSONs
└── corpus/
    └── v0.1/
        ├── explanation_generation.jsonl
        ├── qa_with_citation.jsonl
        ├── quiz_generation.jsonl
        ├── term_extraction.jsonl
        ├── misconception_correction.jsonl
        ├── claim_verification.jsonl
        ├── summary_generation.jsonl
        ├── structured_outline.jsonl
        └── all_slices.jsonl
```

Download the corpus locally:

```bash
gsutil -m rsync -r gs://$PIPELINE_BUCKET/corpus/v0.1/ data/corpus/v0.1/
```

---

### Pipeline Parameters Reference

All parameters can be overridden at submit time by editing `_default_parameter_values()` in
[pipelines/vertex/pipeline.py](pipelines/vertex/pipeline.py) or passing them via the Vertex AI
console when re-running a job.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `project_id` | `str` | `$PROJECT_ID` | GCP project ID |
| `spec_gcs_uris` | `list[str]` | 3 example specs | `gs://` URIs of task spec JSON files to process |
| `kb_gcs_prefix` | `str` | `gs://.../kb/` | GCS prefix where KB chunk JSON files live |
| `gold_gcs_prefix` | `str` | `gs://.../gold/` | GCS prefix for gold bundle output |
| `corpus_gcs_prefix` | `str` | `gs://.../corpus/v0.1/` | GCS prefix for JSONL corpus output |
| `routing_config_gcs_uri` | `str` | `gs://.../configs/models.yaml` | GCS URI of the routing config |
| `openai_secret_id` | `str` | `"openai-api-key"` | Secret Manager secret name holding the API key |
| `no_llm_dean` | `bool` | `false` | Skip LLM rubric scoring in Dean (citation/symbolic checks still run) |
| `corpus_version` | `str` | `"v0.1"` | Version tag embedded in the corpus assembly result |

**Parallelism** is hardcoded to `4` concurrent `generate_gold_bundle` tasks. To change it,
edit the `parallelism=4` argument in `dsl.ParallelFor` in [pipeline.py](pipelines/vertex/pipeline.py:116)
and recompile. This must be a compile-time constant, not a runtime parameter.

---

### Monitoring

After submitting, Vertex AI prints a console URL. You can also find the run at:

```
https://console.cloud.google.com/vertex-ai/pipelines/runs?project=YOUR_PROJECT_ID
```

Each step shows:
- Logs (stdout from the component function)
- Input/output parameter values
- Execution time and machine type

**Check gold bundles landed in GCS:**

```bash
gsutil ls gs://$PIPELINE_BUCKET/gold/GOLD-*.json
gsutil ls gs://$PIPELINE_BUCKET/gold/slices/ | head -20
```

**Check the human review queue in Firestore:**

```bash
gcloud firestore documents list \
    --collection-id=human_review_queue \
    --project=$PROJECT_ID
```

**Check corpus JSONL:**

```bash
gsutil cat gs://$PIPELINE_BUCKET/corpus/v0.1/all_slices.jsonl | head -3 | python3 -m json.tool
```

---

### Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `TypeError: Artifacts must have both a schema_title and schema_version` | `from __future__ import annotations` in a KFP component or pipeline file | Remove that import — KFP v2 needs concrete types, not lazy string annotations |
| `ParallelFor parallelism must be >= 0. Got: {{channel:...}}` | Pipeline parameter passed as `parallelism=` | Use a hardcoded `int` — e.g. `parallelism=4` |
| `The pipeline parameter X is not found in the pipeline job input definitions` | `parameter_values` dict contains a key not in the pipeline function signature | Remove the stale key from `_default_parameter_values()` |
| `OPENAI_API_KEY not set` (component log) | Secret Manager call failed or secret name is wrong | Verify `openai_secret_id` matches the Secret Manager secret name and the SA has `secretmanager.secretAccessor` role |
| `google.api_core.exceptions.PermissionDenied` | Service account missing an IAM role | Re-run `setup_iam.sh` or grant the missing role manually |
| `google.api_core.exceptions.NotFound: ... Bucket ... not found` | `PIPELINE_BUCKET` doesn't exist or wrong name | Create the bucket: `gsutil mb gs://$PIPELINE_BUCKET` |
| Component image pull fails | Image not pushed or wrong `BRAHMX_IMAGE_URI` | Re-run `build_and_push.sh` and re-source `.image_uri.env` |
| `No GOLD-*.json chunks found` in corpus assembly | Gold generation step failed silently | Check Vertex AI logs for `generate_gold_bundle` tasks; look for Dean gate failures |

---

## License

Proprietary — ZenteiQ Aitech Innovations Pvt. Ltd.
