# Copyright 2025 Amazon.com Inc and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Test WeightedDatasetSampler for dataset ratio control.
"""

import warnings
from collections import Counter

import datasets
import numpy as np
import pytest
from omegaconf import DictConfig, OmegaConf

from verl.utils.dataset.weighted_sampler import WeightedDatasetSampler


class MockRLHFDataset:
    """Mock dataset that mimics RLHFDataset structure."""

    def __init__(self, dataset_sources: list[tuple[str, int]]):
        """
        Args:
            dataset_sources: List of (source_name, count) tuples.
                Example: [('gsm8k', 7000), ('math', 1000)]
        """
        data_dicts = []
        for source_name, count in dataset_sources:
            for i in range(count):
                data_dicts.append({
                    'data_source': source_name,
                    'prompt': f'{source_name}_prompt_{i}',
                    'index': i,
                })

        self.dataframe = datasets.Dataset.from_list(data_dicts)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        return self.dataframe[idx]


def test_basic_weighted_sampling():
    """Test basic weighted sampling with two datasets."""
    dataset = MockRLHFDataset([
        ('gsm8k', 7000),
        ('math', 1000),
    ])

    data_config = OmegaConf.create({
        'dataset_ratios': [0.7, 0.3],
        'seed': 42,
    })

    sampler = WeightedDatasetSampler(dataset, data_config)

    # Epoch size should be 7000 (largest dataset)
    assert sampler.epoch_size == 7000
    assert len(sampler) == 7000

    # Check that we have correct number of indices
    indices = list(sampler)
    assert len(indices) == 7000


def test_sampling_ratios_over_multiple_epochs():
    """Test that ratios are maintained over multiple epochs."""
    dataset = MockRLHFDataset([
        ('dataset_a', 1000),
        ('dataset_b', 500),
    ])

    data_config = OmegaConf.create({
        'dataset_ratios': [0.6, 0.4],
        'seed': 42,
    })

    sampler = WeightedDatasetSampler(dataset, data_config)

    # Sample 3 epochs and verify ratios
    for epoch in range(3):
        indices = list(sampler)

        # Count sources
        source_counts = Counter()
        for idx in indices:
            source = dataset[idx]['data_source']
            source_counts[source] += 1

        total = sum(source_counts.values())

        # Check ratios (with some tolerance for randomness)
        dataset_a_ratio = source_counts['dataset_a'] / total
        dataset_b_ratio = source_counts['dataset_b'] / total

        assert abs(dataset_a_ratio - 0.6) < 0.01, f"Epoch {epoch}: dataset_a ratio {dataset_a_ratio}"
        assert abs(dataset_b_ratio - 0.4) < 0.01, f"Epoch {epoch}: dataset_b ratio {dataset_b_ratio}"


def test_over_sampling_with_replacement():
    """Test that small datasets are over-sampled with replacement."""
    dataset = MockRLHFDataset([
        ('large', 5000),
        ('small', 100),
    ])

    data_config = OmegaConf.create({
        'dataset_ratios': [0.5, 0.5],  # Equal ratios
        'seed': 42,
    })

    sampler = WeightedDatasetSampler(dataset, data_config)
    indices = list(sampler)

    # Count how many times each index appears
    small_dataset_indices = [i for i in range(100)]  # First 100 are from 'small'
    small_dataset_samples = [idx for idx in indices if idx < 100]

    # Should have ~2500 samples from small dataset (50% of 5000)
    # Since small dataset only has 100 samples, they must be repeated
    assert len(small_dataset_samples) > 100, "Small dataset should be over-sampled"

    # Some indices should appear multiple times
    index_counts = Counter(small_dataset_samples)
    max_count = max(index_counts.values())
    assert max_count > 1, "Some samples from small dataset should be repeated"


def test_under_sampling():
    """Test that large datasets are under-sampled."""
    dataset = MockRLHFDataset([
        ('large', 10000),
        ('small', 1000),
    ])

    data_config = OmegaConf.create({
        'dataset_ratios': [0.3, 0.7],  # More from small dataset
        'epoch_size': 5000,
        'seed': 42,
    })

    sampler = WeightedDatasetSampler(dataset, data_config)
    indices = list(sampler)

    # Count sources
    source_counts = Counter()
    for idx in indices:
        source = dataset[idx]['data_source']
        source_counts[source] += 1

    # Large dataset should be under-sampled (only 1500 out of 10000)
    assert source_counts['large'] == 1500
    # Small dataset should be over-sampled (3500 out of 1000)
    assert source_counts['small'] == 3500


def test_custom_epoch_size():
    """Test custom epoch size setting."""
    dataset = MockRLHFDataset([
        ('dataset_a', 1000),
        ('dataset_b', 500),
    ])

    data_config = OmegaConf.create({
        'dataset_ratios': [0.5, 0.5],
        'epoch_size': 2000,
        'seed': 42,
    })

    sampler = WeightedDatasetSampler(dataset, data_config)

    assert sampler.epoch_size == 2000
    assert len(sampler) == 2000

    indices = list(sampler)
    assert len(indices) == 2000


def test_ratio_normalization():
    """Test that ratios are normalized if they don't sum to 1.0."""
    dataset = MockRLHFDataset([
        ('dataset_a', 1000),
        ('dataset_b', 500),
    ])

    data_config = OmegaConf.create({
        'dataset_ratios': [0.7, 0.2],  # Sum = 0.9, not 1.0
        'seed': 42,
    })

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sampler = WeightedDatasetSampler(dataset, data_config)

        # Should have raised a warning
        assert len(w) == 1
        assert "normalizing to 1.0" in str(w[0].message)

    # Ratios should be normalized
    expected_ratio_a = 0.7 / 0.9
    expected_ratio_b = 0.2 / 0.9

    assert abs(sampler.dataset_ratios[0] - expected_ratio_a) < 0.001
    assert abs(sampler.dataset_ratios[1] - expected_ratio_b) < 0.001


def test_missing_dataset_ratios():
    """Test that missing dataset_ratios raises an error."""
    dataset = MockRLHFDataset([
        ('dataset_a', 1000),
    ])

    data_config = OmegaConf.create({
        'seed': 42,
    })

    with pytest.raises(ValueError, match="dataset_ratios must be specified"):
        WeightedDatasetSampler(dataset, data_config)


def test_mismatched_ratios_and_datasets():
    """Test that mismatched number of ratios and datasets raises an error."""
    dataset = MockRLHFDataset([
        ('dataset_a', 1000),
        ('dataset_b', 500),
    ])

    data_config = OmegaConf.create({
        'dataset_ratios': [0.5, 0.3, 0.2],  # 3 ratios but only 2 datasets
        'seed': 42,
    })

    with pytest.raises(ValueError, match="Number of dataset_ratios.*must match"):
        WeightedDatasetSampler(dataset, data_config)


def test_single_dataset_with_ratio():
    """Test that single dataset works with ratio=1.0."""
    dataset = MockRLHFDataset([
        ('dataset_a', 1000),
    ])

    data_config = OmegaConf.create({
        'dataset_ratios': [1.0],
        'seed': 42,
    })

    sampler = WeightedDatasetSampler(dataset, data_config)
    indices = list(sampler)

    assert len(indices) == 1000


def test_reproducibility_with_seed():
    """Test that sampling is reproducible with the same seed."""
    dataset = MockRLHFDataset([
        ('dataset_a', 1000),
        ('dataset_b', 500),
    ])

    data_config = OmegaConf.create({
        'dataset_ratios': [0.6, 0.4],
        'seed': 123,
    })

    sampler1 = WeightedDatasetSampler(dataset, data_config)
    indices1 = list(sampler1)

    sampler2 = WeightedDatasetSampler(dataset, data_config)
    indices2 = list(sampler2)

    # Should be identical
    assert indices1 == indices2


def test_different_seeds_produce_different_samples():
    """Test that different seeds produce different sampling orders."""
    dataset = MockRLHFDataset([
        ('dataset_a', 1000),
        ('dataset_b', 500),
    ])

    data_config1 = OmegaConf.create({
        'dataset_ratios': [0.6, 0.4],
        'seed': 42,
    })

    data_config2 = OmegaConf.create({
        'dataset_ratios': [0.6, 0.4],
        'seed': 999,
    })

    sampler1 = WeightedDatasetSampler(dataset, data_config1)
    indices1 = list(sampler1)

    sampler2 = WeightedDatasetSampler(dataset, data_config2)
    indices2 = list(sampler2)

    # Should be different
    assert indices1 != indices2


def test_three_datasets():
    """Test with three datasets."""
    dataset = MockRLHFDataset([
        ('dataset_a', 5000),
        ('dataset_b', 2000),
        ('dataset_c', 500),
    ])

    data_config = OmegaConf.create({
        'dataset_ratios': [0.5, 0.3, 0.2],
        'seed': 42,
    })

    sampler = WeightedDatasetSampler(dataset, data_config)
    indices = list(sampler)

    # Count sources
    source_counts = Counter()
    for idx in indices:
        source = dataset[idx]['data_source']
        source_counts[source] += 1

    total = sum(source_counts.values())

    # Check ratios
    assert abs(source_counts['dataset_a'] / total - 0.5) < 0.01
    assert abs(source_counts['dataset_b'] / total - 0.3) < 0.01
    assert abs(source_counts['dataset_c'] / total - 0.2) < 0.01


def test_missing_data_source_field():
    """Test that missing data_source field raises an error."""
    # Create dataset without data_source field
    data_dicts = [{'prompt': f'prompt_{i}'} for i in range(100)]
    dataset = datasets.Dataset.from_list(data_dicts)

    class MockDatasetWithoutDataSource:
        def __init__(self):
            self.dataframe = dataset

        def __len__(self):
            return len(self.dataframe)

    mock_dataset = MockDatasetWithoutDataSource()

    data_config = OmegaConf.create({
        'dataset_ratios': [1.0],
        'seed': 42,
    })

    with pytest.raises(ValueError, match="must have a 'data_source' column"):
        WeightedDatasetSampler(mock_dataset, data_config)
