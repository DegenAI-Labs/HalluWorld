# HalluWorld Probe Benchmark Results

**Date:** 2026-05-04  
**Probes:** 529 (extracted from Terminal-Bench trajectories, `extracted_llm_probes_20260504/`)
**Probe types:** causal, cross-tier compound, memory, perceptual, uncertainty

Models marked **Thinking** used extended reasoning (Anthropic adaptive thinking `medium` or OpenAI reasoning effort `medium`).

---

## Overall Leaderboard

| Rank | Model | Org | Thinking | N | Overall Acc |
|-----:|-------|-----|:--------:|--:|:-----------:|
| 1 | GPT-5.5 | OpenAI | ✓ | 529 | **94.1%** |
| 2 | o3 | OpenAI | ✓ | 529 | **89.8%** |
| 3 | o4-mini | OpenAI | ✓ | 529 | **85.8%** |
| 4 | Claude Opus 4.6 | Anthropic | ✓ | 529 | **84.3%** |
| 5 | Claude Sonnet 4.6 | Anthropic | ✓ | 529 | **82.2%** |
| 6 | o3-mini | OpenAI | ✓ | 529 | **79.0%** |
| 7 | GLM-5 | ZhipuAI | | 529 | **73.7%** |
| 8 | GPT-4o | OpenAI | | 529 | **61.6%** |
| 9 | GPT-5.4-mini | OpenAI | | 529 | **49.7%** |
| 10 | GPT-4o-mini | OpenAI | | 529 | **48.2%** |
| 11 | Kimi-K2.6 | Moonshot | | 529 | **43.5%** |

---

## Accuracy by Probe Type

| Model | Overall | Causal | Cross-tier Compound | Memory | Perceptual | Uncertainty |
|-------|--------:|-------:|--------------------:|-------:|-----------:|------------:|
| GPT-5.5 | **94.1%** | 99.1% | 98.1% | 97.2% | 99.1% | 76.9% |
| o3 | **89.8%** | 95.3% | 90.6% | 94.4% | 91.4% | 76.9% |
| o4-mini | **85.8%** | 93.4% | 91.5% | 88.9% | 92.4% | 62.5% |
| Claude Opus 4.6 | **84.3%** | 89.6% | 90.6% | 89.8% | 90.5% | 60.6% |
| Claude Sonnet 4.6 | **82.2%** | 90.6% | 85.9% | 88.9% | 81.9% | 63.5% |
| o3-mini | **79.0%** | 85.9% | 84.0% | 86.1% | 85.7% | 52.9% |
| GLM-5 | **73.7%** | 79.2% | 78.3% | 78.7% | 65.7% | 66.3% |
| GPT-4o | **61.6%** | 64.1% | 60.4% | 65.7% | 64.8% | 52.9% |
| GPT-5.4-mini | **49.7%** | 57.6% | 53.8% | 56.5% | 44.8% | 35.6% |
| GPT-4o-mini | **48.2%** | 60.4% | 53.8% | 42.6% | 38.1% | 46.2% |
| Kimi-K2.6 | **43.5%** | 43.4% | 52.8% | 50.9% | 42.9% | 26.9% |

---

## Notes

- **Uncertainty probes** are consistently the hardest category across all models, even for top performers (GPT-5.5: 76.9%, o3: 76.9%).
- **Thinking models** outperform non-thinking models significantly: top 6 all use extended reasoning. The gap between o3-mini (79.0%, thinking) and GLM-5 (73.7%, no thinking) illustrates this.
- **DeepSeek-V3** and **Qwen3-30B** results are pending — ran into model ID / endpoint issues on Baseten serverless; will be added when resolved.
- All models: 529/529 valid results (1 file in the probe dir is a metadata `summary.json`, not a probe — true probe count is 529).

## Eval Configuration

| Setting | Value |
|---------|-------|
| Max output tokens (thinking models) | 16,384 |
| Max output tokens (standard models) | 256 |
| Concurrency | 8 |
| Timeout | 90–120 s |
| Retries | 3–6 with exponential backoff |
| Eval script | historical `scripts/terminal/tools/evaluate_probes.py` |
| Probe dir | `extracted_llm_probes_20260504/` |
| Run dirs | `runs/probe_eval_<slug>/` |

The 11 × 529 historical response records behind this table are not part of this release.

One provenance caveat is preserved in the manifest rather than hidden: four records in the run
labeled Claude Opus 4.6 report `claude-opus-4-5`; the other 525 report `claude-opus-4-6`.
