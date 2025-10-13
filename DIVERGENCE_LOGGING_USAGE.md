# High-Divergence Token Logging Usage Guide

This guide explains how to use the new linguistic context logging for high-divergence tokens between rollout and actor policies.

## Overview

The implementation logs detailed information about tokens where rollout and actor probabilities significantly diverge, including:
- Token text and ID
- Position information
- Probability values from both backends
- Surrounding context (±5 tokens)
- Full prompt and response text

## Quick Start

### Basic Usage

```python
from verl.utils.debug.metrics import calculate_debug_metrics

# In your training loop, pass the tokenizer to enable decoding
metrics = calculate_debug_metrics(
    data=data_proto,
    tokenizer=tokenizer,  # Pass your tokenizer here
    log_divergence=True,  # Enable divergence logging (default: True)
    divergence_threshold=0.8,  # Only log when max divergence > 0.8 (default: 0.8)
    divergence_top_k=5,  # Log top 5 divergent tokens (default: 5)
    divergence_jsonl_path="./logs/divergence.jsonl",  # Path to save JSONL logs
    iteration=current_iteration,  # Training iteration number
)
```

### Passing the Tokenizer

The tokenizer needs to be accessible in your training code. Here are common patterns:

#### Option 1: Store in DataProto (Recommended)
```python
# In your trainer initialization
data.meta_info['tokenizer'] = self.tokenizer

# In metrics calculation
tokenizer = data.meta_info.get('tokenizer', None)
metrics = calculate_debug_metrics(data, tokenizer=tokenizer, ...)
```

#### Option 2: Pass as Function Parameter
```python
# In your worker/trainer
def update_actor(self, data: DataProto):
    # ... existing code ...
    metrics = calculate_debug_metrics(
        data,
        tokenizer=self.tokenizer,  # Access from self
        log_divergence=True,
        divergence_jsonl_path=self.divergence_log_path,
        iteration=self.current_iteration
    )
```

## Configuration Parameters

### `log_divergence` (bool, default=True)
- Enable/disable detailed divergence logging
- Set to `False` to only compute aggregate metrics without detailed token logging

### `divergence_threshold` (float, default=0.8)
- Minimum max probability divergence to trigger logging
- Only logs when `max(|rollout_prob - actor_prob|) >= threshold`
- Recommended values:
  - `0.5`: Log moderate divergences
  - `0.8`: Log only high divergences (recommended)
  - `0.95`: Log only extreme divergences

### `divergence_top_k` (int, default=5)
- Number of highest-divergence tokens to log per batch
- Recommended values:
  - `5`: Quick overview
  - `10`: More detailed analysis
  - `20`: Comprehensive logging (may be verbose)

### `divergence_jsonl_path` (str, optional)
- Path to save structured JSONL logs
- If `None`, only Python logging is used (human-readable console output)
- Example: `"./logs/divergence_iter_{iteration}.jsonl"`

### `iteration` (int, optional)
- Training iteration number to include in logs
- Helps track divergence evolution over training

## Output Formats

### 1. Python Logging (Console/File)

Human-readable format logged via Python's logging module:

```
================================================================================
High Divergence Tokens (Top 5):
================================================================================

[Rank 0] Divergence: 0.9234
  Token: ' world' (ID: 995)
  Position: 12/50 (from end: 37)
  Batch Index: 3
  Rollout: prob=0.923400, logprob=-0.0795
  Actor:   prob=0.000123, logprob=-8.9034
  LogProb Diff: 8.8239
  Context [7:18]: 'hello world of coding'
  Prompt (45 tokens): 'Write a Python function that...'
  Response (50 tokens): 'Here is the function:\n\ndef hello_world():\n    print("Hello, world")\n\n...'
```

### 2. JSONL Format (Structured Logs)

Each line is a JSON object for easy parsing:

```json
{
  "iteration": 100,
  "rank": 0,
  "batch_idx": 3,
  "position": 12,
  "position_from_end": 37,
  "sequence_length": 50,
  "token_id": 995,
  "token_text": "' world'",
  "prob_divergence": 0.9234,
  "rollout_prob": 0.9234,
  "actor_prob": 0.000123,
  "rollout_logprob": -0.0795,
  "actor_logprob": -8.9034,
  "logprob_diff": 8.8239,
  "is_correct": true,
  "total_score": 1.0,
  "context_text": "'hello world of coding'",
  "context_start_pos": 7,
  "context_end_pos": 18,
  "prompt_text": "'Write a Python function that...'",
  "prompt_num_tokens": 45,
  "full_response": "'Here is the function:\\n\\ndef hello_world():\\n    print(\"Hello, world\")\\n\\n...'",
  "response_num_tokens": 50
}
```

**New Fields (added in correctness logging update):**
- `is_correct` (bool): Whether the response generated a correct answer (based on reward function)
- `total_score` (float): Total reward score for this response (sum of token-level scores)

## Integration Examples

### Example 1: FSDP Actor Worker

```python
# In verl/workers/actor/dp_actor.py

from verl.utils.debug.metrics import calculate_debug_metrics

class DataParallelPPOActor(BasePPOActor):
    def __init__(self, config: ActorConfig, actor_module: nn.Module, ...):
        # ... existing init ...
        self.tokenizer = tokenizer  # Store tokenizer
        self.divergence_log_path = config.get('divergence_log_path', './logs/divergence.jsonl')
        self.current_iteration = 0

    def update_policy(self, data: DataProto):
        # ... existing training code ...

        # Calculate debug metrics with divergence logging
        if self.config.tis_imp_ratio_cap > 0:  # Only when TIS is enabled
            debug_metrics = calculate_debug_metrics(
                data=data,
                tokenizer=self.tokenizer,
                log_divergence=True,
                divergence_threshold=0.8,
                divergence_top_k=5,
                divergence_jsonl_path=self.divergence_log_path,
                iteration=self.current_iteration
            )
            metrics.update(debug_metrics)

        self.current_iteration += 1
        return metrics
```

### Example 2: Ray Trainer

```python
# In your trainer's main loop

from verl.utils.debug.metrics import calculate_debug_metrics

class RayPPOTrainer:
    def train_step(self, iteration: int):
        # ... rollout, compute advantages, etc. ...

        # Before updating actor, compute divergence metrics
        if self.config.log_divergence and iteration % 10 == 0:  # Log every 10 iterations
            debug_metrics = calculate_debug_metrics(
                data=training_data,
                tokenizer=self.tokenizer,
                log_divergence=True,
                divergence_threshold=0.8,
                divergence_top_k=10,
                divergence_jsonl_path=f"{self.log_dir}/divergence_iter{iteration}.jsonl",
                iteration=iteration
            )
            self.logger.log_metrics(debug_metrics)
```

## Analyzing JSONL Logs

### Using pandas

```python
import pandas as pd
import json

# Read JSONL file
logs = []
with open('./logs/divergence.jsonl', 'r') as f:
    for line in f:
        logs.append(json.loads(line))

df = pd.DataFrame(logs)

# Analysis examples
print(f"Average divergence: {df['prob_divergence'].mean():.4f}")
print(f"Max divergence: {df['prob_divergence'].max():.4f}")

# Find most common divergent tokens
print(df['token_text'].value_counts().head(10))

# Analyze divergence by position
import matplotlib.pyplot as plt
plt.hist(df['position'], bins=50)
plt.xlabel('Position in Sequence')
plt.ylabel('Frequency')
plt.title('Distribution of High-Divergence Token Positions')
plt.show()

# Track divergence evolution over iterations
divergence_by_iter = df.groupby('iteration')['prob_divergence'].agg(['mean', 'max', 'count'])
print(divergence_by_iter)

# ===== NEW: Analyze divergence by correctness =====
# Filter by answer correctness
correct_df = df[df['is_correct'] == True]
incorrect_df = df[df['is_correct'] == False]

print(f"\n=== Divergence Analysis by Correctness ===")
print(f"Correct answers - Avg divergence: {correct_df['prob_divergence'].mean():.4f}")
print(f"Incorrect answers - Avg divergence: {incorrect_df['prob_divergence'].mean():.4f}")
print(f"Correct answers - Max divergence: {correct_df['prob_divergence'].max():.4f}")
print(f"Incorrect answers - Max divergence: {incorrect_df['prob_divergence'].max():.4f}")

# Analyze digit tokens by correctness
digit_tokens = df[df['token_text'].str.strip().str.match(r'^[\d\.]+$')]
print(f"\n=== Digit Token Analysis ===")
print(f"Total digit tokens with high divergence: {len(digit_tokens)}")
print(f"  From correct answers: {len(digit_tokens[digit_tokens['is_correct']])} ({len(digit_tokens[digit_tokens['is_correct']])/len(digit_tokens)*100:.1f}%)")
print(f"  From incorrect answers: {len(digit_tokens[~digit_tokens['is_correct']])} ({len(digit_tokens[~digit_tokens['is_correct']])/len(digit_tokens)*100:.1f}%)")

# Visualize divergence distribution by correctness
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(correct_df['prob_divergence'], bins=30, alpha=0.7, label='Correct', color='green')
axes[0].hist(incorrect_df['prob_divergence'], bins=30, alpha=0.7, label='Incorrect', color='red')
axes[0].set_xlabel('Probability Divergence')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Divergence Distribution by Correctness')
axes[0].legend()

# Plot divergence over training for correct vs incorrect
correct_by_iter = correct_df.groupby('iteration')['prob_divergence'].mean()
incorrect_by_iter = incorrect_df.groupby('iteration')['prob_divergence'].mean()
axes[1].plot(correct_by_iter.index, correct_by_iter.values, 'g-', label='Correct', marker='o')
axes[1].plot(incorrect_by_iter.index, incorrect_by_iter.values, 'r-', label='Incorrect', marker='s')
axes[1].set_xlabel('Training Iteration')
axes[1].set_ylabel('Average Divergence')
axes[1].set_title('Divergence Evolution: Correct vs Incorrect')
axes[1].legend()
plt.tight_layout()
plt.show()

# Hypothesis testing: Are digit tokens more divergent in correct answers?
digit_correct = digit_tokens[digit_tokens['is_correct']]
digit_incorrect = digit_tokens[~digit_tokens['is_correct']]
print(f"\n=== Hypothesis Test: Digit Token Divergence ===")
print(f"Digit divergence (correct answers): {digit_correct['prob_divergence'].mean():.4f}")
print(f"Digit divergence (incorrect answers): {digit_incorrect['prob_divergence'].mean():.4f}")
if len(digit_correct) > 0 and len(digit_incorrect) > 0:
    from scipy import stats
    t_stat, p_value = stats.ttest_ind(digit_correct['prob_divergence'], digit_incorrect['prob_divergence'])
    print(f"t-statistic: {t_stat:.4f}, p-value: {p_value:.4e}")
    if p_value < 0.05:
        print("→ Statistically significant difference! ✓")
```

### Using jq (command line)

```bash
# Count tokens by divergence threshold
cat logs/divergence.jsonl | jq 'select(.prob_divergence > 0.9)' | wc -l

# Find most divergent token
cat logs/divergence.jsonl | jq -s 'max_by(.prob_divergence)'

# Extract tokens at specific positions
cat logs/divergence.jsonl | jq 'select(.position < 10) | {token_text, prob_divergence}'

# Group by iteration
cat logs/divergence.jsonl | jq -s 'group_by(.iteration) | map({iteration: .[0].iteration, count: length, avg_divergence: (map(.prob_divergence) | add / length)})'

# ===== NEW: Correctness-based filtering =====
# Count divergent tokens from correct vs incorrect answers
cat logs/divergence.jsonl | jq 'select(.is_correct == true)' | wc -l
cat logs/divergence.jsonl | jq 'select(.is_correct == false)' | wc -l

# Find digit tokens from correct answers with high divergence
cat logs/divergence.jsonl | jq 'select(.is_correct == true and .prob_divergence > 0.8) | select(.token_text | test("^[\"\\047]?[0-9\\.]+[\"\\047]?$"))'

# Average divergence by correctness
cat logs/divergence.jsonl | jq -s 'group_by(.is_correct) | map({is_correct: .[0].is_correct, count: length, avg_div: (map(.prob_divergence) | add / length), max_div: (map(.prob_divergence) | max)})'

# Find cases where actor learned correct answer (high divergence + correct)
cat logs/divergence.jsonl | jq 'select(.is_correct == true and .prob_divergence > 0.9 and .flip_type == "rollout_confident_actor_rejects")'
```

## Performance Considerations

1. **Logging Overhead**: Extracting and decoding context has minimal overhead (~1-2ms per batch) since it only runs when divergence exceeds threshold

2. **Storage**: JSONL files grow over training. Each log entry is ~500-1000 bytes
   - 1000 iterations × 5 tokens/iter ≈ 2.5-5 MB

3. **Recommendations**:
   - Use `divergence_threshold=0.8` to avoid excessive logging
   - Log every N iterations: `if iteration % 10 == 0`
   - Rotate log files by iteration or date
   - Consider disabling after initial debugging phase

## Troubleshooting

### Tokenizer Not Found
```
Error: 'NoneType' object has no attribute 'decode'
```
**Solution**: Pass tokenizer to `calculate_debug_metrics`

### No Divergence Logs
- Check if `max_divergence >= divergence_threshold`
- Verify `log_divergence=True`
- Check if `tis_imp_ratio_cap > 0` (TIS must be enabled)

### JSONL File Permission Error
```
PermissionError: [Errno 13] Permission denied
```
**Solution**: Ensure log directory exists and is writable:
```python
from pathlib import Path
Path("./logs").mkdir(parents=True, exist_ok=True)
```

## Interpreting Correctness Results

### Expected Patterns

**Hypothesis 1: Actor learning correct answers faster than rollout**
- **Expected:** High divergence on digit tokens from **correct** answers
- **Interpretation:** Actor learned right digits, rollout lagging behind
- **Evidence:** `is_correct=true` + high divergence + digit tokens

**Example diagnostic questions:**
```python
# Q1: Are divergent tokens mostly from correct answers?
correct_pct = (df['is_correct'].sum() / len(df)) * 100
print(f"Divergent tokens from correct answers: {correct_pct:.1f}%")
# If > 70%: Strong evidence for Hypothesis 1

# Q2: Do digit tokens diverge more in correct answers?
digit_correct_div = digit_tokens[digit_tokens['is_correct']]['prob_divergence'].mean()
digit_incorrect_div = digit_tokens[~digit_tokens['is_correct']]['prob_divergence'].mean()
print(f"Digit divergence ratio (correct/incorrect): {digit_correct_div/digit_incorrect_div:.2f}x")
# If > 1.5x: Digits diverge more when answer is correct

# Q3: What's the flip pattern?
flip_patterns = df[df['is_correct']].groupby('flip_type').size()
print(flip_patterns)
# "rollout_confident_actor_rejects" suggests actor unlearning rollout's wrong answer
```

### Diagnostic Scenarios

| Observation | Interpretation | Next Steps |
|-------------|----------------|------------|
| 80%+ divergent tokens from **correct** answers | ✅ Actor learning right answers | Confirms Hypothesis 1; normal behavior |
| 80%+ divergent tokens from **incorrect** answers | ⚠️ Actor learning wrong answers | Check reward function; possible training instability |
| Divergence equal on correct/incorrect | 🤔 Random divergence | May indicate TIS not helping; consider disabling |
| Only digit tokens diverge (correct answers) | ✅ Expected for math tasks | Actor concentrating learning on answer-critical tokens |
| Non-digit tokens diverge more | ⚠️ Unexpected pattern | Check prompt formatting; possible tokenization issues |

## Additional Fields to Log (Future Extensions)

Based on your earlier question, you can extend `extract_linguistic_context` to log:
- Attention mask values
- Entropy at divergent positions
- Top-5 token alternatives from each backend
- Advantage/reward values at those positions

See the implementation in `verl/utils/debug/metrics.py` for how to add these fields.
