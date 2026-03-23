# ADR-001: Two-Plane Architecture (Data Factory / Model Factory)

**Status**: Accepted  
**Date**: 2025-06-01  
**Deciders**: ML Platform Team

## Context
We need to design a synthetic data generation and model training system that is auditable, reproducible, and operationally reliable. The key question is whether data generation and model training should be tightly coupled (live teacher-student) or loosely coupled (static artifact handoff).

## Decision
Adopt a strict two-plane architecture:
- **Data Factory**: generates, verifies, and packages training data as static versioned artifacts
- **Model Factory**: consumes only static, versioned corpora for all training stages

The planes communicate ONLY via versioned artifacts in object storage. Training never depends on live teacher inference.

## Consequences
- **Positive**: Training is fully reproducible. Rollback is trivial (point to previous corpus version). No cascading failures from inference into training. Data quality can be measured independently.
- **Negative**: Cannot do online distillation or token-level KD. Must pre-generate all training data before training begins. Corpus lag between generation and training.
- **Mitigation**: Hard distillation recovery stage compensates for quality loss from offline-only distillation.

## Alternatives Considered
1. **Online distillation**: Teacher generates during training. Rejected: fragile, non-reproducible, impossible to audit.
2. **Soft KD with logit matching**: Rejected: custom tokenizer breaks token alignment.
