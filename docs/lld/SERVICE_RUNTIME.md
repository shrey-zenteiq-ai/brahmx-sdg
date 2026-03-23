# BrahmX SDG — Service & Job Runtime Architecture

## Runtime Classification

| Component | Runtime Type | Technology | Why |
|-----------|-------------|------------|-----|
| Source Ingestion | Batch job | K8s Job / KFP component | One-time per source batch, no state needed |
| Evidence Pack Builder | Library (in-process) | Python | Deterministic, fast, called within gold generator |
| Prompt Constructor | Library (in-process) | Python | Deterministic, no external deps |
| Teacher Router | Long-running service | FastAPI + gRPC | Shared routing state, pool health tracking |
| Gold Generator | KFP pipeline | Kubeflow component | Multi-stage with retry, caching, human gates |
| Silver Generator | KFP pipeline | Kubeflow component | Fan-out across lanes, batch processing |
| Dean Service | Batch worker | K8s Job / Ray | High-volume, parallelizable scoring |
| Auditor Service | Batch worker | K8s Job | Lower volume, independence-critical |
| Citation Checker | Library (in-process) | Python | Deterministic lexical matching |
| Symbolic Validator | Library (in-process) | Python (SymPy) | Deterministic computation |
| Code Exec Sandbox | Isolated batch job | K8s Job + nsjail | Security isolation required |
| LaTeX Compiler | Isolated batch job | K8s Job + sandbox | Security isolation, retry loop |
| Lane Processors | Per-lane batch jobs | K8s Job | Independent scaling per lane |
| Corpus Assembler | KFP pipeline | Kubeflow | Multi-stage with dedup/decontam |
| Provenance Registry | Long-running service | FastAPI + PostgreSQL | Persistent state, query API |
| Release Governance | Service + UI backend | FastAPI | Approval workflow state |
| Human Review UI | Web app | FastAPI + React | Interactive review queue |
| Training Launcher | KFP pipeline | Kubeflow | Sequential stage orchestration |
| Eval Runner | Batch job | K8s Job | Compute-intensive, parallelizable |
| Model Packager | Batch job | K8s Job | One-time per checkpoint |

## Scaling Strategy

### Bulk Generation (high throughput, cost-sensitive)
- **vLLM TPU pool**: 4-8 replicas, auto-scaled by queue depth
- **Dean workers**: 8-16 replicas on CPU/small-GPU nodes
- **Lane processors**: 1-4 replicas per lane, scaled by backlog

### Frontier Generation (quality-sensitive, lower throughput)
- **vLLM GPU pool**: 2-4 replicas, dedicated A100/H100 nodes
- **Auditor workers**: 2-4 replicas, separate from dean pool

### Training (TPU-bound)
- **MaxText pods**: Dedicated TPU slices (v5e-256 for primary training)
- **No auto-scaling**: TPU allocation is fixed per training run

## Queue Architecture

```
┌─────────────────┐     ┌─────────────────┐
│  gold-gen-queue  │────▶│ Teacher Router   │──▶ vLLM pools
│  (high priority) │     └─────────────────┘
└─────────────────┘

┌─────────────────┐     ┌─────────────────┐
│ silver-gen-queue │────▶│ Lane Processors  │──▶ vLLM pools
│  (med priority)  │     └─────────────────┘
└─────────────────┘

┌─────────────────┐     ┌─────────────────┐
│  dean-queue      │────▶│ Dean Workers     │──▶ vLLM (scoring)
│  (high priority) │     └─────────────────┘
└─────────────────┘

┌─────────────────┐     ┌─────────────────┐
│  audit-queue     │────▶│ Auditor Workers  │──▶ vLLM (GPU only)
│  (med priority)  │     └─────────────────┘
└─────────────────┘
```

## What Should Stay Simple

- **Evidence Pack Builder**: Keep as a library, don't make it a service
- **Prompt Constructor**: Keep as a library, don't make it a service
- **Citation Checker**: Keep as a library, don't make it a service
- **Symbolic Validator**: Keep as a library, don't make it a service
- **Canonicalizer**: Keep as a library, don't make it a service

These are deterministic, stateless functions called within pipeline stages.
Making them services adds latency and failure modes without benefit.
