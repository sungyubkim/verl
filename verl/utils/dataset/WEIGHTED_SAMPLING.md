# Weighted Dataset Sampling

## Overview

The Weighted Dataset Sampling feature allows you to control the mixing ratio of multiple datasets when training reinforcement learning models. This is useful for:

- **Balancing datasets of different sizes**: Ensure fair representation from each dataset
- **Emphasizing certain datasets**: Give more weight to higher-quality or more relevant data
- **Curriculum learning**: Adjust dataset ratios over time (can be extended)
- **Over-sampling small datasets**: Prevent small datasets from being underrepresented

## How It Works

### File-Based Tracking (Automatic)

When you load multiple datasets with RLHFDataset, each sample is **automatically tagged** with:
- `_file_index`: Integer indicating which file it came from (0, 1, 2, ...)
- `_source_file`: Original file path (for debugging)

The WeightedDatasetSampler uses `_file_index` to identify which file each sample belongs to, then applies the corresponding ratio from `dataset_ratios`.

**Important**: The order of `dataset_ratios` **EXACTLY matches** the order of files in `train_files`.

### Key Concepts

1. **Dataset Ratios**: A list of floats specifying the proportion of samples from each **file** (not data_source)
2. **File Order = Ratio Order**: First ratio applies to first file, second ratio to second file, etc.
3. **Epoch Size**: The total number of samples per epoch (default: size of largest file)
4. **Over-sampling**: Small files are sampled with replacement to meet the ratio
5. **Under-sampling**: Large files are sampled without replacement
6. **No Preprocessing**: Files are automatically tracked, no manual setup needed

### Example

Suppose you have two files:
- **gsm8k.parquet**: 7,000 training samples
- **math.parquet**: 1,000 training samples

```yaml
train_files: [gsm8k.parquet, math.parquet]
dataset_ratios: [0.5, 0.5]  # 50% from each FILE
```

**Without weighted sampling** (default):
- Each epoch uses all 8,000 samples
- GSM8K: 87.5%, MATH: 12.5%

**With weighted sampling** (`dataset_ratios: [0.5, 0.5]`):
- Each epoch uses 7,000 samples (size of largest file)
- GSM8K: 3,500 samples (50%, under-sampled) ← File 0
- MATH: 3,500 samples (50%, over-sampled with replacement) ← File 1

**Startup Logs**:
```
================================================================================
WeightedDatasetSampler Configuration
================================================================================
Total samples in dataset: 8000
Epoch size: 7000
Number of files: 2

File mapping:
  File 0: ~/data/gsm8k.parquet
  File 1: ~/data/math.parquet

Per-file sampling:
  File 0 (~/data/gsm8k.parquet):
    - Original size: 7000
    - Ratio: 50.00%
    - Samples per epoch: 3500
    - Mode: under-sampling (0.50x)

  File 1 (~/data/math.parquet):
    - Original size: 1000
    - Ratio: 50.00%
    - Samples per epoch: 3500
    - Mode: over-sampling (with replacement) (3.50x)
================================================================================
```

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
   - Reads each sample's `_file_index` field to identify which file it comes from
   - Builds an index mapping from file index to sample indices
   - Validates that number of ratios matches number of files
   - Logs file mapping and sampling statistics

2. **Sampling**:
   - For each epoch, calculates how many samples to draw from each file based on ratios
   - Uses `np.random.choice` with `replace=True` for over-sampling (small files)
   - Uses `np.random.choice` with `replace=False` for under-sampling (large files)
   - Shuffles all sampled indices together

3. **Integration**:
   - Automatically activated when `dataset_ratios` is specified in config
   - Integrated into `create_rl_sampler()` in `main_ppo.py`

### Automatic File Tracking

**No manual setup required!** When using `RLHFDataset` to load multiple parquet files, each sample is automatically tagged with:

- `_file_index`: Integer indicating which file it came from (0, 1, 2, ...)
- `_source_file`: Original file path (for debugging and logging)

These columns are added automatically by `RLHFDataset._read_files_and_tokenize()` method.

Example sample after loading:
```python
{
    '_file_index': 0,
    '_source_file': '~/data/gsm8k/train.parquet',
    'prompt': 'What is 2+2?',
    'response': '4',
    # ... other fields from your parquet file
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

### Error: "_file_index column is required"

This error means the dataset doesn't have the `_file_index` column. This usually happens if:
- You're not using `RLHFDataset` to load your data
- You're using a custom dataset class that doesn't add `_file_index`

**Solution**: Make sure you're using `RLHFDataset` to load your parquet files. The `_file_index` column is automatically added when loading multiple files.

### Error: "Number of dataset_ratios must match number of files"

The number of values in `dataset_ratios` must match the number of files in `train_files`.

Example:
```yaml
train_files: [file1.parquet, file2.parquet, file3.parquet]
dataset_ratios: [0.5, 0.3, 0.2]  # ✅ Correct: 3 files, 3 ratios
# dataset_ratios: [0.5, 0.5]     # ❌ Error: 3 files, 2 ratios
```

### Error: "num_workers must be 0"

Set `dataloader_num_workers: 0` in your config.

### Warning: "dataset_ratios sum to X, normalizing to 1.0"

Your ratios don't sum to exactly 1.0. They will be automatically normalized. To avoid the warning, ensure they sum to 1.0.

### Samples seem to repeat too often

This is expected for small datasets with high ratios. The dataset is over-sampled with replacement to meet the specified ratio.

---

## Schema Normalization

### Overview

When loading multiple datasets, you may encounter schema mismatch errors such as:
- Missing fields in some datasets
- Type mismatches between datasets
- Null-type inference issues (when a column contains only null values)
- Field ordering differences in struct types

VERL provides automatic schema normalization to handle these cases.

### Enabling Schema Normalization

```yaml
data:
  train_files:
    - dataset1.parquet
    - dataset2.parquet
  normalize_schema: true  # Enable schema normalization
  schema_default_values:  # Default values for missing fields
    dict: {}
    list: []
    str: ''
    int: 0
    float: 0.0
    bool: false
```

### How It Works

1. **Reference Schema**: The first dataset's schema is used as the reference
2. **Missing Fields**: Fields present in the reference but missing in other datasets are added with default values
3. **Type Casting**: All datasets are cast to match the reference schema (strict mode)
4. **Extra Fields**: Fields in subsequent datasets that aren't in the reference are preserved
5. **Error Handling**: Clear error messages if type casting fails

### Configuration Options

#### `normalize_schema` (boolean)

- **Default**: `false`
- **Description**: Enable/disable automatic schema normalization
- **When to use**: When working with datasets that have slight schema differences

#### `schema_default_values` (dict)

- **Description**: Default values to use for missing fields based on type
- **Default values**:
  ```yaml
  dict: {}
  list: []
  str: ''
  int: 0
  float: 0.0
  bool: false
  ```

### Example Scenarios

#### Scenario 1: Missing Field

**Problem**:
```
Dataset 1: {prompt: str, data_source: str, extra_info: dict}
Dataset 2: {prompt: str, data_source: str}  # Missing extra_info
→ Error: Schema mismatch
```

**Solution**:
```yaml
data:
  normalize_schema: true
  schema_default_values:
    dict: {}
```

**Result**: Dataset 2 gets `extra_info: {}` for all rows

#### Scenario 2: Null-Type Inference

**Problem**:
```
Dataset 1: {prompt: str, tags: list<string>}
Dataset 2: {prompt: str, tags: null}  # All null values → inferred as "null" type
→ Error: Type mismatch (list vs null)
```

**Solution**:
```yaml
data:
  normalize_schema: true
  schema_default_values:
    list: []
```

**Result**: Dataset 2's `tags` field is cast to `list<string>` with empty lists

#### Scenario 3: Type Mismatch (Incompatible)

**Problem**:
```
Dataset 1: {prompt: str, value: str}
Dataset 2: {prompt: str, value: int}
→ Error: Cannot cast int to str
```

**Solution**: Preprocess datasets to align types before training, or keep `normalize_schema: false` and fix data manually.

### When to Use Schema Normalization

**Use it when:**
- Datasets have optional fields that may be missing
- Working with null-heavy datasets
- Combining datasets from different sources with slight variations
- You control the datasets and know types are compatible

**Don't use it when:**
- Datasets have fundamentally incompatible types
- You want strict schema validation
- Performance is critical (normalization adds slight overhead)
- Schemas are already guaranteed to match

### Best Practices

1. **First Dataset Matters**: Ensure your first dataset has all required fields with correct types
2. **Check Logs**: Schema normalization logs which fields are being added/modified
3. **Test First**: Try with small samples to verify normalization works correctly
4. **Set Appropriate Defaults**: Customize `schema_default_values` based on your data
5. **Monitor Warnings**: Pay attention to logged warnings about schema differences

### Error Messages

#### "Schema normalization failed during casting"

This means the type casting couldn't be performed. Common causes:
- Incompatible types (e.g., string → int)
- Nested struct differences
- List element type mismatches

**Solution**: Check the detailed error message to identify the problematic field and align types manually.

#### "Missing fields in dataset: {...}"

This is a warning (not an error) indicating which fields were added with default values.

### Performance Considerations

- Schema normalization adds overhead for each dataset after the first
- For very large datasets, this may add 5-10% to loading time
- Consider preprocessing datasets offline if loading time is critical

---

## References

### Weighted Dataset Sampling
- Implementation: `verl/utils/dataset/weighted_sampler.py`
- Integration: `verl/trainer/main_ppo.py` (see `create_rl_sampler()`)
- Tests: `tests/utils/dataset/test_weighted_sampler_on_cpu.py`
- Example: `examples/gpg_trainer/run_qwen2-7b_math_weighted.sh`
- Config: `verl/trainer/config/data/legacy_data.yaml`

### Schema Normalization
- Implementation: `verl/utils/dataset/rl_dataset.py` (see `_normalize_schema()`, `_get_default_value()`)
- Tests: `tests/utils/dataset/test_schema_normalization_on_cpu.py`
- Config: `verl/trainer/config/data/legacy_data.yaml` (see `normalize_schema`, `schema_default_values`)
