# BrahmX SDG — Ablation Studies & Experiment Plan

## Prioritized Ablation Matrix

| # | Ablation | Hypothesis | Primary Metric | Cost | Priority |
|---|----------|-----------|---------------|------|----------|
| 1 | Gold/silver mixture ratios | 70/30 gold-heavy beats 50/50 for factuality | SciFactBench F1 | Low | P0 |
| 2 | Teacher model combinations | Multi-teacher diversity > single teacher | Output diversity score | Med | P0 |
| 3 | Dean/auditor calibration thresholds | Tighter thresholds → fewer but higher quality samples | Downstream eval accuracy | Low | P0 |
| 4 | Evidence-pack size | More chunks → better citation coverage but noise | Citation precision | Low | P1 |
| 5 | Claim-ledger strictness | Strict ledger enforcement → better factuality | Hallucination rate | Low | P1 |
| 6 | Bounded dialogue depth | 5-turn beats 10-turn for focus | Topic drift rate | Med | P1 |
| 7 | LaTeX compile retry count | 3 retries sufficient; 5 marginal improvement | Compile pass rate | Low | P2 |
| 8 | Code sandbox repair iterations | 3 repairs > 5 (diminishing returns) | Code pass@1 | Med | P2 |
| 9 | Multilingual translation strategy | IndicTrans2 > LLM translation for Indic | BLEU + semantic alignment | Med | P1 |
| 10 | Reasoning-budget variants | Budget-aware traces improve latency at serving | Accuracy vs. latency curve | Med | P2 |
| 11 | Curriculum expansion aggressiveness | 3x expansion optimal before quality drops | Downstream task diversity | Med | P2 |
| 12 | Preference stage inclusion | Preference stage adds +2-3% on MT-Bench | MT-Bench score | High | P0 |
| 13 | Recovery stage frequency | Recovery after RL critical; after SFT marginal | Post-recovery eval delta | High | P1 |
| 14 | Long-context corpus composition | Mixed LC corpus > retrieval-only corpus | RULER score | Med | P1 |
| 15 | vLLM TPU vs GPU partitioning | 80/20 TPU/GPU split optimal for cost | $/sample at quality threshold | Low | P2 |
| 16 | Bulk vs frontier teacher policy | Reserve frontier for <15% of gold samples | Quality delta on hard tasks | Med | P1 |

## Detailed Ablation Designs

### Ablation 1: Gold/Silver Mixture Ratios
- **Why**: Core design decision — gold is expensive, silver is cheap but noisier
- **Variants**: 90/10, 80/20, 70/30, 50/50, 30/70
- **Data split**: Hold out 10% of gold as eval set; train on rest
- **Primary metric**: SciFactBench F1, MMLU-Science
- **Secondary metric**: Perplexity on held-out gold, style diversity
- **Failure criteria**: >5% drop in factuality at any ratio
- **Expected tradeoff**: Higher gold ratio → better factuality, lower diversity

### Ablation 2: Teacher Model Combinations
- **Why**: Multi-lineage teachers provide diversity; single-teacher risks monoculture
- **Variants**: Qwen-only, Llama-only, Qwen+Llama, Qwen+Llama+Kimi
- **Primary metric**: Output diversity (distinct-4), factuality
- **Failure criteria**: Single-teacher matches multi-teacher within 1%

### Ablation 12: Preference Stage Inclusion/Exclusion
- **Why**: Preference training is expensive — need to verify ROI
- **Variants**: With preference stage, without preference stage
- **Primary metric**: MT-Bench, AlpacaEval
- **Failure criteria**: <1% improvement doesn't justify cost

## Evaluation Suites

### Offline Eval Split
- 10% of gold bundles reserved as held-out eval
- Never used in training, decontam-verified
- Refreshed with each corpus version

### Red-Team Evals
- Adversarial factuality probes
- Citation hallucination attacks
- Tool-use exploitation attempts
- Safety boundary testing

### Domain-Specific Evals
- **Scientific factuality**: SciFactBench, SciFact, custom domain QA
- **Tool-use**: ToolBench, custom sandbox execution evals
- **Multilingual**: IndicNLPBench, FLORES translation quality
- **Long-context**: RULER, Needle-in-Haystack, LongBench
- **Code**: HumanEval, MBPP, custom scientific code evals
