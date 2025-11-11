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
Test schema normalization for RLHFDataset.
"""

import tempfile
from pathlib import Path

import datasets
import pandas as pd
import pytest
from omegaconf import OmegaConf

from verl.utils.dataset.rl_dataset import RLHFDataset


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self):
        self.bos_token = "<s>"
        self.eos_token = "</s>"
        self.pad_token = "<pad>"
        self.chat_template = None

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        if tokenize:
            return [1, 2, 3]
        return "".join([m['content'] for m in messages])

    def encode(self, text, add_special_tokens=True):
        return [1] * len(text.split())

    def __call__(self, text, **kwargs):
        return {'input_ids': self.encode(text)}


def create_parquet_file(data_dicts, file_path):
    """Helper to create a parquet file from list of dicts."""
    df = pd.DataFrame(data_dicts)
    df.to_parquet(file_path, index=False)


def test_schema_normalization_disabled_by_default(tmp_path):
    """Test that schema normalization is disabled by default."""
    # Create two parquet files with matching schemas
    file1 = tmp_path / "dataset1.parquet"
    file2 = tmp_path / "dataset2.parquet"

    data1 = [
        {'prompt': 'Question 1', 'data_source': 'dataset1', 'ability': 'math'},
        {'prompt': 'Question 2', 'data_source': 'dataset1', 'ability': 'math'},
    ]
    data2 = [
        {'prompt': 'Question 3', 'data_source': 'dataset2', 'ability': 'coding'},
        {'prompt': 'Question 4', 'data_source': 'dataset2', 'ability': 'coding'},
    ]

    create_parquet_file(data1, file1)
    create_parquet_file(data2, file2)

    config = OmegaConf.create({
        'max_prompt_length': 512,
        'max_response_length': 512,
        'prompt_key': 'prompt',
        # normalize_schema not set (default: false)
    })

    tokenizer = MockTokenizer()

    # Should work fine with compatible schemas
    dataset = RLHFDataset(
        data_files=[str(file1), str(file2)],
        tokenizer=tokenizer,
        config=config,
    )

    assert len(dataset.dataframe) == 4


def test_schema_normalization_missing_field(tmp_path):
    """Test that missing fields are added with default values."""
    file1 = tmp_path / "dataset1.parquet"
    file2 = tmp_path / "dataset2.parquet"

    # First dataset has extra_field, second doesn't
    data1 = [
        {'prompt': 'Q1', 'data_source': 'd1', 'extra_field': 'value1'},
        {'prompt': 'Q2', 'data_source': 'd1', 'extra_field': 'value2'},
    ]
    data2 = [
        {'prompt': 'Q3', 'data_source': 'd2'},  # Missing extra_field
        {'prompt': 'Q4', 'data_source': 'd2'},
    ]

    create_parquet_file(data1, file1)
    create_parquet_file(data2, file2)

    config = OmegaConf.create({
        'max_prompt_length': 512,
        'max_response_length': 512,
        'prompt_key': 'prompt',
        'normalize_schema': True,
        'schema_default_values': {
            'str': '',
            'dict': {},
            'list': [],
        }
    })

    tokenizer = MockTokenizer()

    dataset = RLHFDataset(
        data_files=[str(file1), str(file2)],
        tokenizer=tokenizer,
        config=config,
    )

    # Should have 4 rows
    assert len(dataset.dataframe) == 4

    # Check that extra_field exists in all rows
    assert 'extra_field' in dataset.dataframe.column_names

    # Rows from dataset2 should have default value (empty string)
    assert dataset.dataframe[2]['extra_field'] == ''
    assert dataset.dataframe[3]['extra_field'] == ''


def test_schema_normalization_extra_fields_preserved(tmp_path):
    """Test that extra fields in subsequent datasets are preserved."""
    file1 = tmp_path / "dataset1.parquet"
    file2 = tmp_path / "dataset2.parquet"

    # Second dataset has an extra field not in first
    data1 = [
        {'prompt': 'Q1', 'data_source': 'd1'},
    ]
    data2 = [
        {'prompt': 'Q2', 'data_source': 'd2', 'bonus_field': 'bonus'},
    ]

    create_parquet_file(data1, file1)
    create_parquet_file(data2, file2)

    config = OmegaConf.create({
        'max_prompt_length': 512,
        'max_response_length': 512,
        'prompt_key': 'prompt',
        'normalize_schema': True,
    })

    tokenizer = MockTokenizer()

    dataset = RLHFDataset(
        data_files=[str(file1), str(file2)],
        tokenizer=tokenizer,
        config=config,
    )

    # bonus_field should be in the combined dataset
    assert 'bonus_field' in dataset.dataframe.column_names


def test_schema_normalization_type_mismatch_error(tmp_path):
    """Test that incompatible type casting raises an error."""
    file1 = tmp_path / "dataset1.parquet"
    file2 = tmp_path / "dataset2.parquet"

    # Same field name, different types (string vs int)
    data1 = [
        {'prompt': 'Q1', 'data_source': 'd1', 'value': 'text'},
    ]
    data2 = [
        {'prompt': 'Q2', 'data_source': 'd2', 'value': 123},  # Different type!
    ]

    create_parquet_file(data1, file1)
    create_parquet_file(data2, file2)

    config = OmegaConf.create({
        'max_prompt_length': 512,
        'max_response_length': 512,
        'prompt_key': 'prompt',
        'normalize_schema': True,
    })

    tokenizer = MockTokenizer()

    # Should raise ValueError during schema normalization
    with pytest.raises(ValueError, match="Schema normalization failed"):
        RLHFDataset(
            data_files=[str(file1), str(file2)],
            tokenizer=tokenizer,
            config=config,
        )


def test_schema_normalization_with_dict_default(tmp_path):
    """Test that dict fields get {} as default value."""
    file1 = tmp_path / "dataset1.parquet"
    file2 = tmp_path / "dataset2.parquet"

    # First has extra_info dict, second doesn't
    data1 = [
        {'prompt': 'Q1', 'data_source': 'd1', 'extra_info': {'index': 0}},
    ]
    data2 = [
        {'prompt': 'Q2', 'data_source': 'd2'},
    ]

    create_parquet_file(data1, file1)
    create_parquet_file(data2, file2)

    config = OmegaConf.create({
        'max_prompt_length': 512,
        'max_response_length': 512,
        'prompt_key': 'prompt',
        'normalize_schema': True,
        'schema_default_values': {
            'dict': {},
        }
    })

    tokenizer = MockTokenizer()

    dataset = RLHFDataset(
        data_files=[str(file1), str(file2)],
        tokenizer=tokenizer,
        config=config,
    )

    # Second row should have empty dict
    assert dataset.dataframe[1]['extra_info'] == {}


def test_schema_normalization_with_list_default(tmp_path):
    """Test that list fields get [] as default value."""
    file1 = tmp_path / "dataset1.parquet"
    file2 = tmp_path / "dataset2.parquet"

    data1 = [
        {'prompt': 'Q1', 'data_source': 'd1', 'tags': ['tag1', 'tag2']},
    ]
    data2 = [
        {'prompt': 'Q2', 'data_source': 'd2'},
    ]

    create_parquet_file(data1, file1)
    create_parquet_file(data2, file2)

    config = OmegaConf.create({
        'max_prompt_length': 512,
        'max_response_length': 512,
        'prompt_key': 'prompt',
        'normalize_schema': True,
        'schema_default_values': {
            'list': [],
        }
    })

    tokenizer = MockTokenizer()

    dataset = RLHFDataset(
        data_files=[str(file1), str(file2)],
        tokenizer=tokenizer,
        config=config,
    )

    # Second row should have empty list
    assert dataset.dataframe[1]['tags'] == []


def test_schema_normalization_single_file(tmp_path):
    """Test that normalization with single file doesn't cause issues."""
    file1 = tmp_path / "dataset1.parquet"

    data1 = [
        {'prompt': 'Q1', 'data_source': 'd1'},
    ]

    create_parquet_file(data1, file1)

    config = OmegaConf.create({
        'max_prompt_length': 512,
        'max_response_length': 512,
        'prompt_key': 'prompt',
        'normalize_schema': True,  # Should not cause problems with single file
    })

    tokenizer = MockTokenizer()

    # Should work fine
    dataset = RLHFDataset(
        data_files=[str(file1)],
        tokenizer=tokenizer,
        config=config,
    )

    assert len(dataset.dataframe) == 1


def test_schema_normalization_three_datasets(tmp_path):
    """Test normalization with three datasets."""
    file1 = tmp_path / "dataset1.parquet"
    file2 = tmp_path / "dataset2.parquet"
    file3 = tmp_path / "dataset3.parquet"

    data1 = [
        {'prompt': 'Q1', 'data_source': 'd1', 'field_a': 'a1', 'field_b': 'b1'},
    ]
    data2 = [
        {'prompt': 'Q2', 'data_source': 'd2', 'field_b': 'b2'},  # Missing field_a
    ]
    data3 = [
        {'prompt': 'Q3', 'data_source': 'd3', 'field_a': 'a3'},  # Missing field_b
    ]

    create_parquet_file(data1, file1)
    create_parquet_file(data2, file2)
    create_parquet_file(data3, file3)

    config = OmegaConf.create({
        'max_prompt_length': 512,
        'max_response_length': 512,
        'prompt_key': 'prompt',
        'normalize_schema': True,
        'schema_default_values': {
            'str': 'default',
        }
    })

    tokenizer = MockTokenizer()

    dataset = RLHFDataset(
        data_files=[str(file1), str(file2), str(file3)],
        tokenizer=tokenizer,
        config=config,
    )

    assert len(dataset.dataframe) == 3

    # Check that missing fields are filled with defaults
    assert dataset.dataframe[1]['field_a'] == 'default'  # dataset2 missing field_a
    assert dataset.dataframe[2]['field_b'] == 'default'  # dataset3 missing field_b


def test_get_default_value_method(tmp_path):
    """Test _get_default_value method directly."""
    file1 = tmp_path / "dataset1.parquet"
    data1 = [{'prompt': 'Q1', 'data_source': 'd1'}]
    create_parquet_file(data1, file1)

    config = OmegaConf.create({
        'max_prompt_length': 512,
        'max_response_length': 512,
        'prompt_key': 'prompt',
        'schema_default_values': {
            'dict': {'custom': 'dict'},
            'list': ['custom', 'list'],
            'str': 'custom_str',
            'int': 999,
            'float': 3.14,
            'bool': True,
        }
    })

    tokenizer = MockTokenizer()
    dataset = RLHFDataset(
        data_files=[str(file1)],
        tokenizer=tokenizer,
        config=config,
    )

    # Test different types
    from datasets import Value, Sequence

    assert dataset._get_default_value(Value('string')) == 'custom_str'
    assert dataset._get_default_value(Value('int64')) == 999
    assert dataset._get_default_value(Value('float')) == 3.14
    assert dataset._get_default_value(Value('bool')) == True
