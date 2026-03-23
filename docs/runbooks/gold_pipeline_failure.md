# Runbook: Gold Pipeline Failure

**Severity**: P1  
**On-Call Team**: Data Factory  
**Escalation**: ML Platform Lead → ML Architect

## Symptoms
- Gold generation pipeline enters BLOCKED state
- Bundle publication rate drops to zero
- KFP dashboard shows failed gold pipeline runs

## Diagnosis Steps

1. **Check pipeline state**
   ```bash
   brahmx-sdg gold status --run-id <RUN_ID>
   ```

2. **Identify failure stage**
   - `RETRIEVE_EVIDENCE` → KB gap (missing claims in CFT)
   - `GENERATE_CANDIDATES` → Teacher endpoint failure or all candidates missing claim ledger
   - `DEAN_SCORE` → Dean model failure or CJK leakage
   - `SELECT_CANDIDATE` → No candidate passes thresholds after max repair rounds
   - `AUDITOR_REVIEW` → Auditor model failure or persistent disagreement

3. **Check teacher endpoints**
   ```bash
   curl http://vllm-tpu-pool:8000/health
   curl http://vllm-gpu-frontier:8000/health
   ```

4. **Check logs**
   ```bash
   kubectl logs -l app=gold-generator --tail=200
   ```

## Resolution

### KB Gap (RETRIEVE_EVIDENCE failure)
1. Check `missing_required_claims` in the evidence pack output
2. File KB enrichment ticket
3. Enrich CFT with missing claims
4. Re-run pipeline

### Teacher Failure (GENERATE_CANDIDATES failure)
1. Check vLLM pool health
2. If pool is down: restart pods, check TPU allocation
3. If all candidates lack claim ledger: check prompt template for claim ledger instruction
4. Route to fallback pool if primary is unhealthy

### Dean/Auditor Failure
1. Check CJK leakage logs (Qwen models)
2. If persistent CJK: update SGLang chat template language constraint
3. If threshold failures: review gate config, check for data distribution shift
4. Lower thresholds temporarily if justified (requires ADR)

### All Candidates Rejected After Repair
1. Inspect dean reports for pattern (same failure across all candidates?)
2. Check if evidence pack is insufficient for the task
3. Consider splitting complex tasks into smaller task specs
4. Escalate to human review for manual override

## Prevention
- Monitor dean pass rate: alert if <60% over rolling 1h window
- Monitor teacher latency P99: alert if >10s
- Monitor evidence pack confidence: alert if <0.6 average
