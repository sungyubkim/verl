# Weighted Dataset Sampling

## Overview

The Weighted Dataset Sampling feature allows you to control the mixing ratio of multiple datasets when training reinforcement learning models. This is useful for:

- **Balancing datasets of different sizes**: Ensure fair representation from each dataset
- **Emphasizing certain datasets**: Give more weight to higher-quality or more relevant data
- **Curriculum learning**: Adjust dataset ratios over time (can be extended)
- **Over-sampling small datasets**: Prevent small datasets from being underrepresented

## How It Works

When training with multiple datasets, VERL normally concatenates all datasets and samples uniformly. With weighted sampling, you can specify the exact proportion of samples to draw from each dataset per epoch.

### Key Concepts

1. **Dataset Ratios**: A list of floats specifying the proportion of samples from each dataset
2. **Epoch Size**: The total number of samples per epoch (default: size of largest dataset)
3. **Over-sampling**: Small datasets are sampled with replacement to meet the ratio
4. **Under-sampling**: Large datasets are sampled without replacement

### Example

Suppose you have two datasets:
- **GSM8K**: 7,000 training samples
- **MATH**: 1,000 training samples

**Without weighted sampling** (default):
- Each epoch uses all 8,000 samples
- GSM8K: 87.5%, MATH: 12.5%

**With weighted sampling** (`dataset_ratios: [0.5, 0.5]`):
- Each epoch uses 7,000 samples (size of largest dataset)
- GSM8K: 3,500 samples (50%, under-sampled)
- MATH: 3,500 samples (50%, over-sampled with replacement)

## Usage

### 1. Configuration File

Add to your `data` config:

```yaml
data:
  train_files:
    - ~/data/gsm8k/train.parquet
    - ~/data/math/train.parquet

  # Specify mixing ratios (must sum to ~1.0)
  dataset_ratios: [0.7, 0.3]  # 70% GSM8K, 30% MATH

  # Optional: specify epoch size
  epoch_size: null  # default: uses largest dataset size
  # epoch_size: 8000  # or specify a number

  # IMPORTANT: must set num_workers to 0
  dataloader_num_workers: 0
```

### 2. Command Line

```bash
python3 -m verl.trainer.main_ppo \
    data.train_files="['gsm8k.parquet', 'math.parquet']" \
    data.dataset_ratios="[0.7, 0.3]" \
    data.dataloader_num_workers=0 \
    # ... other configs
```

### 3. Shell Script

See `examples/gpg_trainer/run_qwen2-7b_math_weighted.sh` for a complete example.

## Configuration Options

### `dataset_ratios` (required)

- **Type**: `list[float]`
- **Description**: Proportion of samples from each dataset
- **Requirements**:
  - Must have same length as number of datasets in `train_files`
  - Should sum to approximately 1.0 (will be normalized if not)
  - Order corresponds to order in `train_files`

**Examples**:
```yaml
# Equal mixing of 2 datasets
dataset_ratios: [0.5, 0.5]

# Emphasize first dataset (80-20)
dataset_ratios: [0.8, 0.2]

# Three datasets (50-30-20)
dataset_ratios: [0.5, 0.3, 0.2]
```

### `epoch_size` (optional)

- **Type**: `int` or `null`
- **Default**: `null` (uses size of largest dataset)
- **Description**: Total number of samples to use per epoch

**Examples**:
```yaml
# Automatic (default)
epoch_size: null

# Explicit size
epoch_size: 10000
```

### `dataloader_num_workers` (required to be 0)

- **Type**: `int`
- **Required Value**: `0`
- **Why**: Multi-process workers can cache data, interfering with sampling ratios

```yaml
dataloader_num_workers: 0
```

## Implementation Details

### WeightedDatasetSampler

The weighted sampling is implemented via `WeightedDatasetSampler` class in `verl/utils/dataset/weighted_sampler.py`.

**How it works**:

1. **Initialization**:
   - Reads each sample's `data_source` field to identify which dataset it comes from
   - Builds an index mapping for each dataset
   - Validates that number of ratios matches number of datasets

2. **Sampling**:
   - For each epoch, calculates how many samples to draw from each dataset
   - Uses `np.random.choice` with `replace=True` for over-sampling
   - Uses `np.random.choice` with `replace=False` for under-sampling
   - Shuffles all sampled indices together

3. **Integration**:
   - Automatically activated when `dataset_ratios` is specified in config
   - Integrated into `create_rl_sampler()` in `main_ppo.py`

### Data Source Field

Each sample in your dataset must have a `data_source` field indicating which dataset it comes from. This is automatically added when using `RLHFDataset` with multiple parquet files.

Example sample:
```python
{
    'data_source': 'gsm8k',  # or 'math', etc.
    'prompt': 'What is 2+2?',
    'response': '4',
    # ... other fields
}
```

## Examples

### Example 1: Balanced Sampling (Equal Ratios)

```yaml
data:
  train_files:
    - dataset_a.parquet  # 10,000 samples
    - dataset_b.parquet  # 2,000 samples
  dataset_ratios: [0.5, 0.5]
  dataloader_num_workers: 0
```

**Result**:
- Epoch size: 10,000 (largest dataset)
- Dataset A: 5,000 samples (under-sampled)
- Dataset B: 5,000 samples (over-sampled, 2.5x repetition)

### Example 2: Emphasizing One Dataset

```yaml
data:
  train_files:
    - high_quality.parquet  # 5,000 samples
    - low_quality.parquet   # 5,000 samples
  dataset_ratios: [0.9, 0.1]
  dataloader_num_workers: 0
```

**Result**:
- Epoch size: 5,000
- High quality: 4,500 samples (90%)
- Low quality: 500 samples (10%)

### Example 3: Three Datasets

```yaml
data:
  train_files:
    - dataset_1.parquet  # 8,000 samples
    - dataset_2.parquet  # 4,000 samples
    - dataset_3.parquet  # 1,000 samples
  dataset_ratios: [0.5, 0.3, 0.2]
  dataloader_num_workers: 0
```

**Result**:
- Epoch size: 8,000 (largest dataset)
- Dataset 1: 4,000 samples (50%)
- Dataset 2: 2,400 samples (30%)
- Dataset 3: 1,600 samples (20%, over-sampled 1.6x)

### Example 4: Custom Epoch Size

```yaml
data:
  train_files:
    - dataset_a.parquet  # 5,000 samples
    - dataset_b.parquet  # 3,000 samples
  dataset_ratios: [0.6, 0.4]
  epoch_size: 12000  # Explicitly set larger than any dataset
  dataloader_num_workers: 0
```

**Result**:
- Epoch size: 12,000 (custom)
- Dataset A: 7,200 samples (60%, over-sampled 1.44x)
- Dataset B: 4,800 samples (40%, over-sampled 1.6x)

## Reproducibility

Use the `seed` parameter to ensure reproducible sampling:

```yaml
data:
  dataset_ratios: [0.7, 0.3]
  seed: 42
  dataloader_num_workers: 0
```

With the same seed, you'll get identical sampling across runs.

## Backward Compatibility

The weighted sampling feature is **completely backward compatible**:

- If `dataset_ratios` is **not specified** (or set to `null`), the default behavior (simple concatenation) is used
- Existing configs and scripts will continue to work without modification
- No changes to existing dataset classes required

## Testing

Comprehensive tests are available in `tests/utils/dataset/test_weighted_sampler_on_cpu.py`.

Run tests with:
```bash
pytest tests/utils/dataset/test_weighted_sampler_on_cpu.py -v
```

## Limitations and Considerations

1. **Over-sampling can lead to overfitting**: Small datasets will have samples repeated multiple times
2. **Requires `dataloader_num_workers=0`**: Multi-process workers interfere with sampling
3. **Memory usage**: All datasets are still loaded into memory (no streaming support yet)
4. **Static ratios**: Ratios are fixed per training run (dynamic curriculum learning would require extending the sampler)

## Future Extensions

Potential future enhancements:

- **Dynamic ratio adjustment**: Change ratios during training (curriculum learning)
- **Per-batch stratification**: Guarantee specific ratios within each batch
- **Streaming support**: Work with datasets larger than memory
- **Automatic ratio optimization**: Learn optimal ratios during training

## Troubleshooting

### Error: "dataset_ratios must be specified"

Make sure to set `dataset_ratios` in your config when using `WeightedDatasetSampler`.

### Error: "Number of dataset_ratios must match number of unique datasets"

The number of values in `dataset_ratios` must match the number of files in `train_files`.

### Error: "num_workers must be 0"

Set `dataloader_num_workers: 0` in your config.

### Warning: "dataset_ratios sum to X, normalizing to 1.0"

Your ratios don't sum to exactly 1.0. They will be automatically normalized. To avoid the warning, ensure they sum to 1.0.

### Samples seem to repeat too often

This is expected for small datasets with high ratios. The dataset is over-sampled with replacement to meet the specified ratio.

## References

- Implementation: `verl/utils/dataset/weighted_sampler.py`
- Integration: `verl/trainer/main_ppo.py` (see `create_rl_sampler()`)
- Tests: `tests/utils/dataset/test_weighted_sampler_on_cpu.py`
- Example: `examples/gpg_trainer/run_qwen2-7b_math_weighted.sh`
- Config: `verl/trainer/config/data/legacy_data.yaml`
