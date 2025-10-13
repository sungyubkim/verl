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
  "context_text": "'hello world of coding'",
  "context_start_pos": 7,
  "context_end_pos": 18,
  "prompt_text": "'Write a Python function that...'",
  "prompt_num_tokens": 45,
  "full_response": "'Here is the function:\\n\\ndef hello_world():\\n    print(\"Hello, world\")\\n\\n...'",
  "response_num_tokens": 50
}
```

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

## Additional Fields to Log (Future Extensions)

Based on your earlier question, you can extend `extract_linguistic_context` to log:
- Attention mask values
- Entropy at divergent positions
- Top-5 token alternatives from each backend
- Advantage/reward values at those positions

See the implementation in `verl/utils/debug/metrics.py` for how to add these fields.
