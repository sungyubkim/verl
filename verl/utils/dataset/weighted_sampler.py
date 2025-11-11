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
"""Weighted dataset sampler for controlling mixing ratios of multiple datasets."""

import warnings
from collections import defaultdict
from collections.abc import Sized
from typing import Iterator

import numpy as np
from omegaconf import DictConfig

from verl.experimental.dataset.sampler import AbstractSampler


class WeightedDatasetSampler(AbstractSampler):
    """Sampler that controls the mixing ratio of multiple datasets.

    This sampler allows you to specify the proportion of samples to draw from each
    dataset when training on multiple datasets simultaneously. It supports over-sampling
    (with replacement) for smaller datasets and under-sampling (without replacement)
    for larger datasets to achieve the desired ratios.

    Args:
        data_source: The dataset to sample from. Must have a 'data_source' field
            indicating which dataset each sample comes from.
        data_config: Configuration object containing:
            - dataset_ratios (list[float]): Proportion of samples from each dataset.
                Should sum to ~1.0. Order corresponds to the order datasets appear.
            - epoch_size (int, optional): Total number of samples per epoch.
                If None, uses the size of the largest dataset.
            - seed (int, optional): Random seed for reproducibility.

    Example:
        With two datasets A (7000 samples) and B (1000 samples):

        dataset_ratios: [0.7, 0.3]
        epoch_size: 8000

        Each epoch will sample:
        - 5600 samples from dataset A (under-sampling)
        - 2400 samples from dataset B (over-sampling with replacement)
    """

    def __init__(
        self,
        data_source: Sized,
        data_config: DictConfig,
    ):
        super().__init__(data_source, data_config)

        self.data_source = data_source
        self.data_config = data_config

        # Get configuration
        self.dataset_ratios = data_config.get("dataset_ratios", None)
        if self.dataset_ratios is None:
            raise ValueError(
                "dataset_ratios must be specified in data_config when using WeightedDatasetSampler. "
                "Example: dataset_ratios: [0.7, 0.3]"
            )

        self.dataset_ratios = list(self.dataset_ratios)

        # Validate and normalize ratios
        ratio_sum = sum(self.dataset_ratios)
        if abs(ratio_sum - 1.0) > 0.01:
            warnings.warn(
                f"dataset_ratios sum to {ratio_sum:.4f}, normalizing to 1.0. "
                f"Original ratios: {self.dataset_ratios}"
            )
            self.dataset_ratios = [r / ratio_sum for r in self.dataset_ratios]

        # Build index mapping for each dataset
        self.dataset_indices = self._build_dataset_indices()

        # Validate number of datasets matches number of ratios
        if len(self.dataset_indices) != len(self.dataset_ratios):
            raise ValueError(
                f"Number of dataset_ratios ({len(self.dataset_ratios)}) must match "
                f"number of unique datasets ({len(self.dataset_indices)}). "
                f"Found datasets: {list(self.dataset_indices.keys())}"
            )

        # Determine epoch size
        self.epoch_size = data_config.get("epoch_size", None)
        if self.epoch_size is None:
            # Use the size of the largest dataset
            max_dataset_size = max(len(indices) for indices in self.dataset_indices.values())
            self.epoch_size = max_dataset_size

        # Setup random number generator
        self.seed = data_config.get("seed", None)
        self.generator = np.random.default_rng(self.seed)

        # Log sampling information
        print("\n" + "=" * 80)
        print("WeightedDatasetSampler Configuration")
        print("=" * 80)
        print(f"Epoch size: {self.epoch_size}")
        print(f"Total samples in concatenated dataset: {len(data_source)}")
        print("\nPer-dataset sampling:")
        for i, (dataset_name, indices) in enumerate(self.dataset_indices.items()):
            ratio = self.dataset_ratios[i]
            n_samples = int(self.epoch_size * ratio)
            sampling_mode = "over-sampling (with replacement)" if n_samples > len(indices) else "under-sampling"
            repetition = n_samples / len(indices) if len(indices) > 0 else 0
            print(f"  Dataset '{dataset_name}':")
            print(f"    - Original size: {len(indices)}")
            print(f"    - Ratio: {ratio:.2%}")
            print(f"    - Samples per epoch: {n_samples}")
            print(f"    - Mode: {sampling_mode} ({repetition:.2f}x)")
        print("=" * 80 + "\n")

    def _build_dataset_indices(self) -> dict[str, list[int]]:
        """Build a mapping from dataset name to list of indices.

        Returns:
            Dictionary mapping dataset name to list of sample indices.
        """
        dataset_indices = defaultdict(list)

        # Check if dataset has a dataframe attribute (for RLHFDataset)
        if hasattr(self.data_source, 'dataframe'):
            # Efficiently extract data_source column
            dataframe = self.data_source.dataframe
            if 'data_source' not in dataframe.column_names:
                raise ValueError(
                    "Dataset must have a 'data_source' column to use WeightedDatasetSampler. "
                    f"Available columns: {dataframe.column_names}"
                )

            data_sources = dataframe['data_source']
            for idx, source in enumerate(data_sources):
                dataset_indices[source].append(idx)
        else:
            # Fallback: iterate through dataset (slower)
            warnings.warn(
                "Dataset does not have 'dataframe' attribute. "
                "Falling back to iterating through dataset, which may be slow."
            )
            for idx in range(len(self.data_source)):
                item = self.data_source[idx]
                if 'data_source' not in item:
                    raise ValueError(
                        f"Sample at index {idx} does not have 'data_source' field. "
                        "All samples must have 'data_source' to use WeightedDatasetSampler."
                    )
                source = item['data_source']
                dataset_indices[source].append(idx)

        return dict(dataset_indices)

    def __iter__(self) -> Iterator[int]:
        """Generate indices for one epoch according to dataset ratios.

        Yields:
            Shuffled indices sampled according to dataset_ratios.
        """
        all_indices = []

        # Sample from each dataset according to its ratio
        for i, (dataset_name, indices_pool) in enumerate(self.dataset_indices.items()):
            ratio = self.dataset_ratios[i]
            n_samples = int(self.epoch_size * ratio)

            # Determine if we need replacement
            if n_samples > len(indices_pool):
                # Over-sampling: sample with replacement
                sampled_indices = self.generator.choice(
                    indices_pool,
                    size=n_samples,
                    replace=True
                )
            else:
                # Under-sampling: sample without replacement
                sampled_indices = self.generator.choice(
                    indices_pool,
                    size=n_samples,
                    replace=False
                )

            all_indices.extend(sampled_indices.tolist())

        # Shuffle all indices together
        self.generator.shuffle(all_indices)

        # Yield indices one by one
        yield from all_indices

    def __len__(self) -> int:
        """Return the number of samples per epoch."""
        return self.epoch_size
