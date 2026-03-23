# BrahmX SDG — Documentation Index

## Architecture
- [High-Level Design (HLD)](hld/HLD.md) — System architecture, subsystem responsibilities, deployment topology
- [HLD Critique & Improvements](hld/HLD_CRITIQUE.md) — Critical review and improvement proposals
- [Low-Level Design (LLD)](lld/LLD.md) — Module architecture, interfaces, contracts, storage

## ADRs (Architecture Decision Records)
- [ADR-001: Two-Plane Architecture](adrs/ADR-001-two-plane-architecture.md)
- [ADR-002: Hard Distillation Only](adrs/ADR-002-hard-distillation.md)
- [ADR-003: vLLM-First Inference](adrs/ADR-003-vllm-first.md)
- [ADR-004: Evidence-First Gold Path](adrs/ADR-004-evidence-first.md)
- [ADR-005: Custom Tokenizer Freeze Policy](adrs/ADR-005-tokenizer-freeze.md)
- [ADR-006: MaxText for Training Only](adrs/ADR-006-maxtext-training.md)
- [ADR-007: Claim Ledger as First-Class Object](adrs/ADR-007-claim-ledger.md)
- [ADR-008: Lexical Citation Check over LLM Judges](adrs/ADR-008-lexical-citations.md)

## Guides
- [Repository Guide](guides/repo_guide.md)
- [Local Development](guides/local_development.md)
- [Pipeline Walkthrough](guides/pipeline_walkthrough.md)
- [Kubeflow Deployment](guides/kubeflow_deployment.md)
- [Model Routing Guide](guides/model_routing.md)
- [Trust & Governance Guide](guides/trust_governance.md)
- [Adding a New Scientific Lane](guides/adding_a_lane.md)
- [Adding a New Teacher Model](guides/adding_a_teacher.md)
- [Corpus Release Process](guides/corpus_release.md)
- [Training Process](guides/training_process.md)
- [Ablation Study Workflow](guides/ablation_workflow.md)

## Runbooks
- [Gold Pipeline Failure](runbooks/gold_pipeline_failure.md)
- [Teacher Endpoint Down](runbooks/teacher_endpoint_down.md)
- [Corpus Contamination Detected](runbooks/corpus_contamination.md)
- [Training Job Failure](runbooks/training_failure.md)
- [Rollback Procedure](runbooks/rollback.md)
- [Incident Response](runbooks/incident_response.md)

## Schema Documentation
- [Data Models Reference](lld/LLD.md#6-artifact-schemas)
- [API Contracts](lld/LLD.md#3-contract-boundaries)
