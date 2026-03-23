# BrahmX SDG — Low-Level Design (LLD)

**Version**: 1.0.0  
**Status**: Implementation Review Ready

---

## 1. Service Boundaries

### 1.1 Data Factory Services

| Service | Type | Runtime | Responsibility |
|---------|------|---------|----------------|
| `source-ingestion` | Batch job / KFP component | K8s Job | Parse, normalize, tag, chunk source documents |
| `evidence-pack-builder` | Library (in-process) | Python | Build task-local evidence bundles from CFT + chunks |
| `prompt-constructor` | Library (in-process) | Python | Deterministic prompt assembly |
| `teacher-router` | Long-running service | FastAPI/gRPC | Route generation requests to appropriate model pool |
| `gold-generator` | KFP pipeline | Kubeflow | Orchestrate full gold path |
| `silver-generator` | KFP pipeline | Kubeflow | Orchestrate silver expansion |
| `dean-service` | Batch worker | K8s Job / Ray | Rubric-based scoring with citation + symbolic checks |
| `auditor-service` | Batch worker | K8s Job | Independent adversarial review |
| `lane-processors` | Per-lane batch jobs | K8s Job | Specialized scientific generation lanes |
| `corpus-assembler` | KFP pipeline | Kubeflow | Dedup, decontam, mix, package, release |
| `provenance-registry` | Long-running service | FastAPI + DB | Manifest storage, lineage tracking, promotion status |
| `release-governance` | Service + UI backend | FastAPI | Approval gates, rollback, audit trail |
| `human-review-ui` | Web app backend | FastAPI | Review queue for contested samples |

### 1.2 Model Factory Services

| Service | Type | Runtime | Responsibility |
|---------|------|---------|----------------|
| `training-launcher` | KFP pipeline | Kubeflow | Launch MaxText training jobs |
| `checkpoint-manager` | Batch job | K8s Job | Checkpoint conversion, validation |
| `eval-runner` | Batch job / KFP | K8s Job | Run evaluation harnesses |
| `model-packager` | Batch job | K8s Job | Package for serving |

## 2. Module Architecture

### 2.1 Core Package: `brahmx_sdg`

```
src/brahmx_sdg/
├── __init__.py
├── cli.py                          # Typer CLI entrypoint
├── common/
│   ├── __init__.py
│   ├── config.py                   # Config loading (YAML/env)
│   ├── logging.py                  # Structured logging setup
│   ├── storage.py                  # GCS abstraction
│   ├── retry.py                    # Retry policies
│   └── hashing.py                  # Deterministic hashing utilities
├── schemas/
│   ├── __init__.py
│   ├── source.py                   # SourceDocument, SourceAtom, SourceChunk
│   ├── evidence.py                 # EvidencePack, CanonicalFact
│   ├── generation.py               # TaskSpec, PromptSpec, TeacherCandidate
│   ├── verification.py             # Claim, ClaimLedger, DeanScore, AuditorReport
│   ├── bundles.py                  # GoldRecordBundle, SilverBundle
│   ├── corpus.py                   # DatasetSlice, CorpusVersion, ReleaseManifest
│   ├── training.py                 # TrainingRunSpec, EvalReport, ModelRelease
│   └── routing.py                  # ModelEndpoint, RoutingDecision
├── ingestion/
│   ├── __init__.py
│   ├── source_ingestion.py         # Main ingestion pipeline
│   ├── parsers/                    # Format-specific parsers
│   │   ├── pdf_parser.py
│   │   ├── latex_parser.py
│   │   ├── code_parser.py
│   │   └── simulation_json_parser.py
│   ├── normalizer.py               # Text normalization
│   ├── license_tagger.py           # License detection & tagging
│   ├── decontam_scanner.py         # Benchmark contamination check
│   └── tokenizer_previewer.py      # Token inflation preview
├── kb/
│   ├── __init__.py
│   ├── canonical_fact_table.py     # CFT store with CRUD, validity, licensing
│   ├── evidence_pack_builder.py    # Build evidence packs from CFT + chunks
│   └── license_filter.py           # License enforcement
├── prompt/
│   ├── __init__.py
│   ├── constructor.py              # Deterministic prompt constructor
│   ├── format_library.py           # Prompt format templates
│   └── anti_leakage.py             # Prompt decontamination
├── generation/
│   ├── __init__.py
│   ├── teacher.py                  # Teacher abstraction
│   ├── candidate_selector.py       # Multi-candidate selection logic
│   ├── gold_generator.py           # Gold path orchestrator
│   └── silver_generator.py         # Silver expansion orchestrator
├── verification/
│   ├── __init__.py
│   ├── claim_ledger.py             # Claim extraction & validation
│   ├── dean.py                     # Dean scoring service
│   ├── auditor.py                  # Independent auditor
│   ├── citation_checker.py         # BM25/TF-IDF citation verification
│   ├── symbolic_numeric_validator.py  # Formula, unit, numerical checks
│   ├── code_exec_validator.py      # Sandbox code execution
│   ├── latex_compile_validator.py  # LaTeX compile loop
│   └── tool_engine.py              # Tool verification engine
├── lanes/
│   ├── __init__.py
│   ├── base_lane.py                # Abstract lane interface
│   ├── simulation_json_lane.py     # Structured scientific simulation
│   ├── dialogue_lane.py            # Bounded multi-LLM conversation
│   ├── latex_lane.py               # LaTeX artifact generation
│   ├── code_tool_lane.py           # Code / tool-use
│   ├── multilingual_lane.py        # Multilingual sovereign-language
│   ├── reasoning_budget_lane.py    # Reasoning-budget variants
│   └── curriculum_lane.py          # Curriculum difficulty expansion
├── packaging/
│   ├── __init__.py
│   ├── bundle_assembler.py         # Assemble gold/silver bundles
│   ├── slice_emitter.py            # Emit training slices from bundles
│   ├── canonicalizer.py            # Canonicalize text for training
│   └── tool_spec_registry.py       # Tool schema registry
├── corpus/
│   ├── __init__.py
│   ├── corpus_assembler.py         # Full corpus assembly pipeline
│   ├── dedup.py                    # Exact + semantic deduplication
│   ├── decontam.py                 # Eval set contamination scanning
│   ├── mixture_controller.py       # Gold/silver/lane mixture control
│   ├── dataset_exporter.py         # ArrayRecord/Parquet export
│   └── release_manager.py          # Release manifest creation
├── training/
│   ├── __init__.py
│   ├── maxtext_training_launcher.py    # MaxText job launcher
│   ├── distillation_recovery_launcher.py  # Hard distillation recovery
│   ├── long_context_recovery_launcher.py  # LC recovery
│   └── training_run_spec.py        # Run spec builder
├── serving/
│   ├── __init__.py
│   └── model_packager.py           # Package for vLLM serving
├── routing/
│   ├── __init__.py
│   ├── teacher_router.py           # Model routing orchestration
│   ├── model_registry.py           # Model capability registry
│   ├── pool_manager.py             # vLLM pool management
│   └── admission_controller.py     # Queue/rate management
├── governance/
│   ├── __init__.py
│   ├── provenance_registry.py      # Provenance storage & queries
│   ├── release_governance.py       # Release approval workflow
│   └── rollback_manager.py         # Rollback metadata & execution
└── evals/
    ├── __init__.py
    ├── eval_runner.py              # Evaluation harness
    ├── scientific_factuality.py    # Science fact evals
    ├── tool_use_eval.py            # Tool-use evals
    └── multilingual_eval.py        # Multilingual quality evals
```

## 3. Contract Boundaries

### 3.1 Inter-Service Contracts

All inter-service communication uses **versioned Pydantic schemas** serialized as JSON. Artifact handoff uses **GCS paths + manifest JSON**.

| Producer | Consumer | Contract | Medium |
|----------|----------|----------|--------|
| Ingestion | Gold/Silver Generator | `SourceAtom` | GCS + manifest |
| Evidence Pack Builder | Prompt Constructor | `EvidencePack` | In-memory dict |
| Gold Generator | Corpus Assembler | `GoldRecordBundle` | GCS + manifest |
| Silver Generator | Corpus Assembler | `SilverBundle` | GCS + manifest |
| Corpus Assembler | Training Launcher | `CorpusVersion` | GCS + release manifest |
| Training Launcher | Eval Runner | Checkpoint path | GCS path |
| Eval Runner | Model Packager | `EvalReport` | GCS + JSON |

### 3.2 Storage Abstractions

```python
class ArtifactStore(Protocol):
    def put(self, key: str, data: bytes, metadata: dict) -> str: ...
    def get(self, key: str) -> tuple[bytes, dict]: ...
    def list_keys(self, prefix: str) -> list[str]: ...
    def exists(self, key: str) -> bool: ...
```

Implementations: `GCSArtifactStore`, `LocalArtifactStore` (dev), `MockArtifactStore` (test).

## 4. Orchestration Boundaries

### 4.1 Kubeflow Pipelines

| Pipeline | Trigger | Components | Cacheable |
|----------|---------|------------|-----------|
| `source_ingestion_pipeline` | New source batch | parse → normalize → tag → chunk → upload | Yes (deterministic) |
| `gold_generation_pipeline` | Task spec batch | evidence → prompt → generate → verify → bundle | Partially (generation non-deterministic) |
| `silver_generation_pipeline` | Gold batch + lane config | seed → lane fan-out → validate → bundle | Partially |
| `corpus_assembly_pipeline` | Release trigger | dedup → decontam → mix → export → manifest | Yes |
| `training_pipeline` | Corpus release | MaxText launch → checkpoint → eval | No |
| `eval_packaging_pipeline` | Checkpoint ready | eval → package → release sign-off | Yes (eval is deterministic) |

### 4.2 Queue / Job Design

| Queue | Purpose | Concurrency | Priority |
|-------|---------|-------------|----------|
| `gold-generation` | Gold path teacher generation | Per-pool capacity | High |
| `silver-expansion` | Silver breadth generation | Per-pool capacity | Medium |
| `dean-scoring` | Dean verification jobs | High (cheap) | High |
| `auditor-review` | Auditor jobs | Medium | Medium |
| `lane-processing` | Lane-specific generation | Per-lane capacity | Medium |
| `corpus-assembly` | Corpus packaging | Single (serialized) | Low |
| `training-jobs` | MaxText training launches | 1 per stage | Critical |

## 5. Retry & Error Semantics

| Operation | Retry Policy | Max Retries | Backoff | On Exhaustion |
|-----------|-------------|-------------|---------|---------------|
| Teacher generation | Retry with same prompt | 3 | Exponential 2s | Mark BLOCKED, file ticket |
| Dean scoring | Retry (non-deterministic) | 2 | Linear 1s | Use best available score |
| Citation check | No retry (deterministic) | 0 | — | Fail immediately |
| Code sandbox exec | Retry with repair prompt | 5 | None | Reject artifact |
| LaTeX compile | Retry with error feedback | 5 | None | Reject artifact |
| GCS upload | Retry | 5 | Exponential 1s | Fail pipeline step |
| vLLM inference | Retry with fallback pool | 3 | Exponential 2s | Route to fallback pool |

## 6. Artifact Schemas (Summary)

All schemas are defined as Pydantic BaseModel classes in `src/brahmx_sdg/schemas/`. Key artifacts:

- **GoldRecordBundle**: The atomic unit of trust. Contains selected candidate, claim ledger, dean report, auditor report, tool results, human review state, provenance.
- **SilverBundle**: Validator-backed breadth data with source lineage and promotion status.
- **ReleaseManifest**: Immutable record of a corpus version with bundle references, mixture ratios, checksums.
- **TrainingRunSpec**: Complete specification for a training job including corpus version, hyperparameters, infrastructure config.

## 7. Component Interactions (Sequence)

### 7.1 Gold Generation Sequence

```
TaskSpec → EvidencePackBuilder.build(spec, kb)
  → PromptConstructor.build(spec, evidence_pack)
  → TeacherRouter.generate(prompt, role="teacher_a", n=3)
  → [for each candidate] ClaimLedger.extract(candidate)
  → [for each candidate] Dean.score(candidate, spec, evidence_pack)
  → [for each candidate] ToolEngine.verify(candidate.tool_checks)
  → CandidateSelector.select(candidates, dean_reports)
  → Auditor.review(selected, spec)
  → [if contested] HumanReviewGate.submit(selected)
  → BundleAssembler.assemble(all_artifacts)
  → SliceEmitter.emit(bundle) → [8 training slice formats]
  → ArtifactStore.put(bundle_path, bundle)
  → ProvenanceRegistry.register(bundle)
```

### 7.2 Corpus Assembly Sequence

```
ReleaseConfig → CorpusAssembler.collect(gold_manifests, silver_manifests)
  → Dedup.run(all_bundles) → remove exact + near duplicates
  → Decontam.scan(all_bundles, eval_sets) → flag contaminated
  → MixtureController.apply(bundles, mixture_policy)
  → Canonicalizer.canonicalize(bundles, tokenizer)
  → DatasetExporter.export(canonical, format="arrayrecord")
  → ReleaseManager.create_manifest(export_path, metadata)
  → ProvenanceRegistry.register_release(manifest)
  → [Optional] ReleaseGovernance.submit_for_approval(manifest)
```
