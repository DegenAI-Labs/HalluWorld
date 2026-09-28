# Providers

> Deployment IDs below are placeholders. The original documented endpoints were account-scoped
> identifiers for deployments that have since been deactivated; substitute your own. Serverless
> endpoints (`https://inference.baseten.co/v1`) need no deployment ID.

How each LM provider is called, and the behaviors that will surprise you.

## Provider semantics that affect results

**OpenAI reasoning models** (`gpt-5*`, `o1*`, `o3*`, `o4*`) are detected by name prefix. For those,
temperature is forced to `None`, `max_completion_tokens` replaces `max_tokens`, and the cap is
raised to at least 16000. Any table reporting a temperature for these models is reporting a number
that was never sent.

**Anthropic** always sends temperature, including alongside adaptive thinking. The
`thinking={"type":"adaptive"}` path is gated on a substring check for `opus-4-6` / `sonnet-4-6`,
which is brittle to model renames -- check it when adding a model.

**Baseten** deployed endpoints (URL contains `model-`) must be called with `model=""`; serverless
endpoints take the real ID. Qwen models get their cap raised, because they emit reasoning into a
separate field and need headroom to also populate `content`.

**A per-call `max_tokens` is a floor, never a ceiling**, across every provider. A cap that is too
small produces an empty response, which the scoring path would otherwise record as a hallucination
rather than as an infrastructure failure.

**Reasoning models are not deterministic** just because temperature is ignored. Backend
nondeterminism and reasoning-path variation still cause run-to-run drift.

---

# Baseten Model Support

This document explains how to use Baseten-deployed models (GPT-OSS-120B, Qwen, GLM-5) in HalluWorld benchmarks.

## Supported Models

### 1. GPT-OSS-120B (Serverless)
- **Model ID:** `openai/gpt-oss-120b`
- **Endpoint:** Baseten serverless (`https://inference.baseten.co/v1`)
- **Type:** Regular instruct model
- **Token budget:** Default (256) or custom via `--max-tokens`

### 2. Qwen-3-30B-Instruct (Deployed)
- **Model ID:** `qwen-3-30b-instruct`
- **Endpoint:** `https://model-<YOUR-DEPLOYMENT-ID>.api.baseten.co/environments/production/sync/v1`
- **Type:** Instruct model (no separate reasoning field)
- **Token budget:** Auto-scaled to 4096 tokens
- **Characteristics:** Clean, concise answers

### 3. Qwen-3-30B-Thinking (Deployed)
- **Model ID:** `qwen-3-30b-thinking`
- **Endpoint:** `https://model-<YOUR-DEPLOYMENT-ID>.api.baseten.co/environments/production/sync/v1`
- **Type:** Thinking model (generates reasoning internally, returns clean answer in `content` field)
- **Token budget:** Auto-scaled to 4096 tokens
- **Characteristics:** Higher accuracy due to internal reasoning, slower inference

### 4. GLM-5 (Serverless)
- **Model ID:** `zai-org/GLM-5` or `GLM-5`
- **Endpoint:** Baseten serverless (`https://inference.baseten.co/v1`)
- **Type:** Regular instruct model
- **Token budget:** Default (256) or custom via `--max-tokens`

---

## Setup

### 1. Environment Variables

```bash
# Required for all Baseten models
export BASETEN_API_KEY="your_api_key_here"

# Required for deployed models (Qwen)
# For Qwen Instruct:
export BASETEN_BASE_URL="https://model-<YOUR-DEPLOYMENT-ID>.api.baseten.co/environments/production/sync/v1"

# For Qwen Thinking:
export BASETEN_BASE_URL="https://model-<YOUR-DEPLOYMENT-ID>.api.baseten.co/environments/production/sync/v1"

# Not needed for serverless models (GPT-OSS-120B, GLM-5)
```

### 2. Install Dependencies

```bash
pip install openai  # Baseten uses OpenAI-compatible API
```

---

## Usage Examples

### Perception Eval (P1/P2/P3)

```bash
# GPT-OSS-120B (serverless - no BASETEN_BASE_URL needed)
python run_perception_eval.py \
  --models openai/gpt-oss-120b \
  --levels P1_dense_array P2_corridor_gauntlet P3_rotation_challenge \
  --episodes 10 \
  --output results_gptoss.csv

# Qwen Instruct (deployed - needs BASETEN_BASE_URL)
export BASETEN_BASE_URL="https://model-<YOUR-DEPLOYMENT-ID>.api.baseten.co/environments/production/sync/v1"
python run_perception_eval.py \
  --models qwen-3-30b-instruct \
  --levels P1_dense_array P2_corridor_gauntlet P3_rotation_challenge \
  --episodes 10 \
  --output results_qwen_instruct.csv

# Qwen Thinking (deployed - needs BASETEN_BASE_URL)
export BASETEN_BASE_URL="https://model-<YOUR-DEPLOYMENT-ID>.api.baseten.co/environments/production/sync/v1"
python run_perception_eval.py \
  --models qwen-3-30b-thinking \
  --levels P1_dense_array P2_corridor_gauntlet P3_rotation_challenge \
  --episodes 10 \
  --output results_qwen_thinking.csv

# GLM-5 (serverless - no BASETEN_BASE_URL needed)
python run_perception_eval.py \
  --models zai-org/GLM-5 \
  --levels P1_dense_array P2_corridor_gauntlet P3_rotation_challenge \
  --episodes 10 \
  --output results_glm5.csv
```

### Dynamics Eval (Wind Probe)

```bash
# GPT-OSS-120B
python run_dynamics_eval.py \
  --models openai/gpt-oss-120b \
  --hint-modes none hint full \
  --episodes 10 \
  --out results_dynamics_gptoss.csv

# Qwen Instruct
export BASETEN_BASE_URL="https://model-<YOUR-DEPLOYMENT-ID>.api.baseten.co/environments/production/sync/v1"
python run_dynamics_eval.py \
  --models qwen-3-30b-instruct \
  --hint-modes none hint full \
  --episodes 10 \
  --out results_dynamics_qwen_instruct.csv

# Qwen Thinking
export BASETEN_BASE_URL="https://model-<YOUR-DEPLOYMENT-ID>.api.baseten.co/environments/production/sync/v1"
python run_dynamics_eval.py \
  --models qwen-3-30b-thinking \
  --hint-modes none hint full \
  --episodes 10 \
  --out results_dynamics_qwen_thinking.csv
```

### Inventory Eval (Carrying Probe)

```bash
# GPT-OSS-120B
python run_inventory_eval.py \
  --models openai/gpt-oss-120b \
  --episodes 10 \
  --out results_inventory_gptoss.csv

# Qwen Instruct
export BASETEN_BASE_URL="https://model-<YOUR-DEPLOYMENT-ID>.api.baseten.co/environments/production/sync/v1"
python run_inventory_eval.py \
  --models qwen-3-30b-instruct \
  --episodes 10 \
  --out results_inventory_qwen_instruct.csv

# Qwen Thinking
export BASETEN_BASE_URL="https://model-<YOUR-DEPLOYMENT-ID>.api.baseten.co/environments/production/sync/v1"
python run_inventory_eval.py \
  --models qwen-3-30b-thinking \
  --episodes 10 \
  --out results_inventory_qwen_thinking.csv
```

---

## Code Integration

### Using BasetenLM Directly

```python
from halluworld.lm import BasetenLM
import os

# Serverless model (GPT-OSS-120B, GLM-5)
lm = BasetenLM(
    model="openai/gpt-oss-120b",
    base_url="https://inference.baseten.co/v1",
    api_key=os.environ["BASETEN_API_KEY"],
    temperature=0.0,
    max_tokens=256
)

# Deployed model (Qwen)
lm = BasetenLM(
    model="qwen-3-30b-instruct",
    base_url="https://model-<YOUR-DEPLOYMENT-ID>.api.baseten.co/environments/production/sync/v1",
    api_key=os.environ["BASETEN_API_KEY"],
    temperature=0.0,
    # max_tokens auto-scales to 4096 for Qwen models
)

# Query the model
response = lm.query(
    system="You are an agent in a gridworld.",
    user="Is there a red key visible?"
)
print(response.text)  # Clean answer
```

### Auto-Detection in Eval Scripts

All eval scripts (`run_perception_eval.py`, `run_dynamics_eval.py`, `run_inventory_eval.py`) automatically detect Baseten models:

```python
# Auto-detection logic
is_baseten = (model.lower().startswith("glm") or
              model.startswith("zai-org/") or
              model.startswith("openai/") or
              model.lower().startswith("qwen"))

if is_baseten:
    lm = BasetenLM(
        model=model if "/" in model else f"zai-org/{model}",
        base_url=os.environ.get("BASETEN_BASE_URL", "https://inference.baseten.co/v1"),
        api_key=os.environ.get("BASETEN_API_KEY"),
        temperature=0.0,
        max_tokens=max_tokens,
    )
else:
    lm = OpenAILM(...)  # OpenAI models (GPT-4o, GPT-5-mini, etc.)
```

---

## Architecture Notes

### Token Budget Auto-Scaling

Qwen thinking models require large token budgets to:
1. Generate verbose internal reasoning (in `reasoning` field)
2. Populate clean final answer (in `content` field)

BasetenLM automatically scales Qwen models to 4096 tokens:

```python
# In BasetenLM.__init__
if "qwen" in model.lower():
    self.max_tokens = max(max_tokens, 4096)
```

### Response Field Handling

**Qwen Thinking Model:**
- Generates verbose reasoning internally (uses many tokens)
- Returns clean answer in `content` field
- Falls back to `reasoning` field if `content` is None (edge case)

**Other Models:**
- Return answer directly in `content` field
- No separate reasoning field

```python
# In BasetenLM.query()
text = choice.message.content
if text is None and hasattr(choice.message, 'reasoning'):
    # Fallback for edge cases
    text = choice.message.reasoning
    log.warning("Content field was None, falling back to reasoning field")

# Strip whitespace
text = (text or "").strip()
```

---

## Performance Characteristics

Based on mini-test results (3 episodes each):

### P1 Dense Array (Perception)
| Model | Attribute | Count | Presence |
|-------|-----------|-------|----------|
| GPT-OSS-120B | 100% | 100% | 100% |
| Qwen Instruct | 100% | 66.7% | 100% |
| **Qwen Thinking** | **100%** | **100%** | **100%** |

### P3 Rotation Challenge (Allocentric)
| Model | Location (ego) | Allocentric |
|-------|----------------|-------------|
| GPT-OSS-120B | 100% | 0% |
| **Qwen Instruct** | **100%** | **100%** |
| Qwen Thinking | 100% | 50% |

**Key Insights:**
- **Qwen Thinking:** Best overall accuracy (100% on P1)
- **Qwen Instruct:** Best on allocentric reasoning (100% vs 0-50%)
- **GPT-OSS-120B:** Strong baseline, struggles with allocentric
- Each model has unique strengths

---

## Troubleshooting

### "Content field was None" Warning

This is expected for some Qwen responses and is handled automatically via fallback to `reasoning` field. If you see this warning frequently, consider:
1. Increasing `--max-tokens` (though 4096 is already generous)
2. Checking BASETEN_BASE_URL is set correctly for deployed models

### Model Not Found (404)

**For deployed models (Qwen):**
- Ensure `BASETEN_BASE_URL` is set to the correct deployment URL
- Check that API key has access to the deployment

**For serverless models (GPT-OSS, GLM-5):**
- No `BASETEN_BASE_URL` needed (uses default serverless endpoint)
- Verify model ID: `openai/gpt-oss-120b` or `zai-org/GLM-5`

### Slow Inference

**Qwen Thinking** is slower than Instruct/GPT-OSS due to:
- Internal reasoning generation (high token count)
- 4096 token budget
- Deployed model vs serverless (different infrastructure)

Typical speeds:
- GPT-OSS-120B: ~5-10s per episode
- Qwen Instruct: ~10-20s per episode
- Qwen Thinking: ~60-90s per episode (generates reasoning internally)

---

## Files Modified

1. `halluworld/lm/baseten_lm.py` - BasetenLM implementation with auto-scaling
2. `run_perception_eval.py` - Added Baseten model detection
3. `run_dynamics_eval.py` - Added Baseten model detection
4. `run_inventory_eval.py` - Added Baseten model detection

---

## Future Work

- Test with additional Baseten deployments
- Benchmark token usage vs accuracy trade-offs
- Explore other thinking models on Baseten
- Add support for reasoning_effort parameter for Baseten models

---

**Questions?** Contact Varun Gangal (LTI, CMU) or see [baseten.co](https://baseten.co) for deployment docs.


---

# Baseten Models Setup Guide
## Qwen-3-30B (Instruct & Thinking) + GPT-OSS-120B

**Last Updated:** April 25, 2026

---

## Supported Models

| Model | Type | Endpoint | Token Budget | Notes |
|-------|------|----------|--------------|-------|
| **qwen-3-30b-instruct** | Instruct | Deployed | **4096 (auto)** | Clean, concise answers |
| **qwen-3-30b-thinking** | Thinking | Deployed | **4096 (auto)** | Internal reasoning, best accuracy |
| **openai/gpt-oss-120b** | Instruct | Serverless | 256 (default) | 120B params, open-source |
| **zai-org/GLM-5** | Instruct | Serverless | 256 (default) | Open-source |

---

## Environment Setup

```bash
# Required for all Baseten models
export BASETEN_API_KEY="your_baseten_api_key_here"

# For Qwen Instruct (deployed model)
export BASETEN_BASE_URL="https://model-<YOUR-DEPLOYMENT-ID>.api.baseten.co/environments/production/sync/v1"

# For Qwen Thinking (deployed model)
export BASETEN_BASE_URL="https://model-<YOUR-DEPLOYMENT-ID>.api.baseten.co/environments/production/sync/v1"

# For serverless models (GPT-OSS, GLM-5) - no BASETEN_BASE_URL needed
# Uses default: https://inference.baseten.co/v1
```

---

## Quick Start

### Qwen-3-30B-Instruct

```bash
export BASETEN_BASE_URL="https://model-<YOUR-DEPLOYMENT-ID>.api.baseten.co/environments/production/sync/v1"
python run_perception_eval.py \
  --models qwen-3-30b-instruct \
  --levels P1_dense_array \
  --episodes 10 \
  --seed 42 \
  --output results_qwen_instruct.csv
```

### Qwen-3-30B-Thinking

```bash
export BASETEN_BASE_URL="https://model-<YOUR-DEPLOYMENT-ID>.api.baseten.co/environments/production/sync/v1"
python run_perception_eval.py \
  --models qwen-3-30b-thinking \
  --levels P1_dense_array \
  --episodes 10 \
  --seed 42 \
  --output results_qwen_thinking.csv
```

### GPT-OSS-120B (Serverless)

```bash
# No BASETEN_BASE_URL needed
python run_perception_eval.py \
  --models openai/gpt-oss-120b \
  --levels P1_dense_array \
  --episodes 10 \
  --seed 42 \
  --output results_gptoss.csv
```

### GLM-5 (Serverless)

```bash
# No BASETEN_BASE_URL needed
python run_perception_eval.py \
  --models zai-org/GLM-5 \
  --levels P1_dense_array \
  --episodes 10 \
  --seed 42 \
  --output results_glm5.csv
```

---

## Full Experimental Suite

Run all Baseten models on all eval types (perception, dynamics, inventory):

```bash
# Edit run_full_experiments.sh to select models
./run_full_experiments.sh
```

Or manually:

```bash
# Perception (P1/P2/P3)
for MODEL in "qwen-3-30b-instruct" "qwen-3-30b-thinking" "openai/gpt-oss-120b" "zai-org/GLM-5"; do
  for SEED in 42 123 999; do
    # Set BASE_URL for deployed models
    if [[ $MODEL == qwen* ]]; then
      if [[ $MODEL == *instruct ]]; then
        export BASETEN_BASE_URL="https://model-<YOUR-DEPLOYMENT-ID>.api.baseten.co/environments/production/sync/v1"
      else
        export BASETEN_BASE_URL="https://model-<YOUR-DEPLOYMENT-ID>.api.baseten.co/environments/production/sync/v1"
      fi
    fi

    python run_perception_eval.py \
      --models $MODEL \
      --levels P1_dense_array P2_corridor_gauntlet P3_rotation_challenge \
      --episodes 15 \
      --seed $SEED \
      --output results_${MODEL}_seed${SEED}.csv
  done
done
```

---

## Important Features

### 1. Auto-Scaling Token Budget (Qwen Only)

Qwen models automatically get 4096 tokens (vs default 256) to ensure:
- Thinking models have space for internal reasoning
- Content field gets populated with clean final answer

**Code:** `halluworld/lm/baseten_lm.py`
```python
if "qwen" in model.lower():
    self.max_tokens = max(max_tokens, 4096)
```

### 2. Auto-Detection

Models are auto-detected and routed to BasetenLM:

```python
is_baseten = (model.lower().startswith("glm") or
              model.startswith("zai-org/") or
              model.startswith("openai/") or
              model.lower().startswith("qwen"))
```

### 3. Thinking Model Response Handling

Qwen-Thinking separates:
- **reasoning field:** Full internal reasoning (verbose)
- **content field:** Clean final answer (what we use)
- **Fallback:** If content=None, use reasoning field

---

## Comparison: Instruct vs Thinking

Based on comprehensive experiments (45 episodes across 3 seeds):

| Probe | Instruct | Thinking | Winner |
|-------|----------|----------|--------|
| **Count** | 36% | **100%** | Thinking (+64pp) |
| **Attribute** | 90% | **100%** | Thinking (+10pp) |
| **Allocentric** | 9% | **18%** | Thinking (+9pp) |
| **Order** | 100% | 100% | Tie |
| **Inventory** | 82% | **100%** | Thinking (+18pp) |

**Conclusion:** Thinking mode dramatically improves counting, attributes, and multi-turn memory. Use **Thinking for best accuracy**, **Instruct for speed/cost**.

---

## Deployment URLs Reference

**Qwen-3-30B-Instruct:**
```
https://model-<YOUR-DEPLOYMENT-ID>.api.baseten.co/environments/production/sync/v1
```

**Qwen-3-30B-Thinking:**
```
https://model-<YOUR-DEPLOYMENT-ID>.api.baseten.co/environments/production/sync/v1
```

**Serverless (GPT-OSS, GLM-5):**
```
https://inference.baseten.co/v1
(default, no need to set)
```

---

## Troubleshooting

### "Content field was None" Warnings

**Expected for Qwen-Thinking** on complex/multi-turn probes (dynamics). The fallback to reasoning field handles this automatically. No action needed.

### 404 Not Found

- **Deployed models:** Check `BASETEN_BASE_URL` matches deployment URL exactly
- **Serverless models:** Don't set `BASETEN_BASE_URL` (or set to inference.baseten.co/v1)
- **Model parameter:** Deployed models use `model=""` (empty string) internally

### Rate Limits

Baseten has rate limits. Code auto-retries with exponential backoff (3 retries).

### Slow First Request (Qwen)

Deployed models scale down after 30-60 mins of inactivity. First request after scale-down takes 1-2 minutes to cold-start. Subsequent requests are fast.

**Workaround:** Warm up deployment before experiments:
```bash
python test_qwen_deployments.py
```

---

## Performance Characteristics

**Qwen-Instruct:**
- ~10-20s per episode
- Clean, concise responses (2-10 tokens)
- No reasoning overhead

**Qwen-Thinking:**
- ~60-120s per episode
- Verbose internal reasoning (hundreds of tokens)
- Clean final answer in content field
- Best accuracy

**GPT-OSS-120B:**
- ~5-10s per episode (serverless)
- Good baseline, but struggles with order/allocentric

**GLM-5:**
- ~5-10s per episode (serverless)
- Competitive with GPT models
- 100% inventory (strong multi-turn)

---

## Code Integration

See `BASETEN_MODELS.md` for full API usage and integration patterns.

---

**For comprehensive experimental results, see:** `experiment_summary_april25_varun.md`
