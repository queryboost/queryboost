import pytest
import pyarrow as pa
from datasets import Dataset, IterableDataset

from queryboost.exceptions import (
    QueryboostDataTypeError,
    QueryboostBatchSizeError,
    QueryboostDataColumnError,
)
from queryboost.utils.data import DataBatcher, invert_list_of_dicts


class TestInvertListOfDicts:
    """Test suite for invert_list_of_dicts function."""

    def test_invert_simple_list(self):
        """Test inverting a simple list of dicts."""
        data = [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25},
            {"name": "Charlie", "age": 35},
        ]

        result = invert_list_of_dicts(data)

        assert result == {
            "name": ["Alice", "Bob", "Charlie"],
            "age": [30, 25, 35],
        }

    def test_invert_empty_list(self):
        """Test inverting an empty list returns empty dict."""
        result = invert_list_of_dicts([])
        assert result == {}

    def test_invert_single_element(self):
        """Test inverting a list with single element."""
        data = [{"key": "value", "number": 42}]

        result = invert_list_of_dicts(data)

        assert result == {
            "key": ["value"],
            "number": [42],
        }

    def test_invert_with_various_types(self):
        """Test inverting list with various data types."""
        data = [
            {"str": "text", "int": 1, "float": 1.5, "bool": True, "none": None},
            {"str": "more", "int": 2, "float": 2.5, "bool": False, "none": None},
        ]

        result = invert_list_of_dicts(data)

        assert result == {
            "str": ["text", "more"],
            "int": [1, 2],
            "float": [1.5, 2.5],
            "bool": [True, False],
            "none": [None, None],
        }


class TestDataBatcher:
    """Test suite for DataBatcher class."""

    def test_init_with_list(self):
        """Test DataBatcher initialization with list."""
        data = [
            {"text": "hello", "id": 1},
            {"text": "world", "id": 2},
        ]
        batcher = DataBatcher(data, batch_size=2)

        assert batcher._batch_size == 2
        assert isinstance(batcher.schema, pa.Schema)
        assert batcher.num_rows == 2

    def test_init_with_dataset(self):
        """Test DataBatcher initialization with Hugging Face Dataset."""
        dataset = Dataset.from_dict(
            {
                "text": ["a", "b", "c"],
                "label": [0, 1, 0],
            }
        )
        batcher = DataBatcher(dataset, batch_size=2)

        assert batcher._batch_size == 2
        assert isinstance(batcher.schema, pa.Schema)
        assert batcher.num_rows == 3

    def test_init_with_iterable_dataset(self):
        """Test DataBatcher initialization with Hugging Face IterableDataset."""
        from datasets import Value, Features

        def gen():
            for i in range(5):
                yield {"value": i, "name": f"item_{i}"}

        features = Features({"value": Value("int64"), "name": Value("string")})
        iterable_dataset = IterableDataset.from_generator(gen, features=features)
        batcher = DataBatcher(iterable_dataset, batch_size=2)

        assert batcher._batch_size == 2
        assert isinstance(batcher.schema, pa.Schema)
        # num_rows is None for IterableDataset without metadata
        assert batcher.num_rows is None

    def test_init_with_iterator(self):
        """Test DataBatcher initialization with iterator."""

        def data_generator():
            yield {"value": 1}
            yield {"value": 2}
            yield {"value": 3}

        batcher = DataBatcher(data_generator(), batch_size=2)

        assert batcher._batch_size == 2
        assert isinstance(batcher.schema, pa.Schema)
        assert batcher.num_rows is None  # Unknown for iterators

    def test_zero_batch_size_raises_error(self):
        """Test that batch size of 0 raises error."""
        data = [{"x": 1}]

        with pytest.raises(QueryboostBatchSizeError) as exc_info:
            DataBatcher(data, batch_size=0)

        assert "greater than 0" in str(exc_info.value)

    def test_negative_batch_size_raises_error(self):
        """Test that negative batch size raises error."""
        data = [{"x": 1}]

        with pytest.raises(QueryboostBatchSizeError) as exc_info:
            DataBatcher(data, batch_size=-5)

        assert "greater than 0" in str(exc_info.value)

    def test_unsupported_data_type_raises_error(self):
        """Test that unsupported data type raises error."""
        invalid_data = "not a valid type"

        with pytest.raises(QueryboostDataTypeError) as exc_info:
            DataBatcher(invalid_data, batch_size=2)  # type: ignore

        assert "Unsupported data type" in str(exc_info.value)

    def test_batch_list_data(self):
        """Test batching list data."""
        data = [{"text": f"item_{i}", "id": i} for i in range(10)]
        batcher = DataBatcher(data, batch_size=3)

        batches = list(batcher)

        assert len(batches) == 4  # 3 + 3 + 3 + 1
        assert batches[0]["text"] == ["item_0", "item_1", "item_2"]
        assert batches[0]["id"] == [0, 1, 2]
        assert batches[-1]["text"] == ["item_9"]
        assert batches[-1]["id"] == [9]

    def test_batch_dataset(self):
        """Test batching Hugging Face Dataset."""
        dataset = Dataset.from_dict(
            {
                "value": list(range(7)),
                "name": [f"name_{i}" for i in range(7)],
            }
        )
        batcher = DataBatcher(dataset, batch_size=3)

        batches = list(batcher)

        assert len(batches) == 3  # 3 + 3 + 1
        assert len(batches[0]["value"]) == 3
        assert len(batches[1]["value"]) == 3
        assert len(batches[2]["value"]) == 1

    def test_batch_iterable_dataset(self):
        """Test batching Hugging Face IterableDataset."""
        from datasets import Value, Features

        def gen():
            for i in range(8):
                yield {"num": i, "text": f"text_{i}"}

        features = Features({"num": Value("int64"), "text": Value("string")})
        iterable_dataset = IterableDataset.from_generator(gen, features=features)
        batcher = DataBatcher(iterable_dataset, batch_size=3)

        batches = list(batcher)

        assert len(batches) == 3  # 3 + 3 + 2
        assert len(batches[0]["num"]) == 3
        assert len(batches[1]["num"]) == 3
        assert len(batches[2]["num"]) == 2

    def test_batch_iterator(self):
        """Test batching iterator data."""

        def data_gen():
            for i in range(5):
                yield {"num": i, "doubled": i * 2}

        batcher = DataBatcher(data_gen(), batch_size=2)

        batches = list(batcher)

        assert len(batches) == 3  # 2 + 2 + 1
        assert batches[0] == {"num": [0, 1], "doubled": [0, 2]}
        assert batches[1] == {"num": [2, 3], "doubled": [4, 6]}
        assert batches[2] == {"num": [4], "doubled": [8]}

    def test_schema_inference_from_list(self):
        """Test PyArrow schema inference from list."""
        data = [
            {"text": "hello", "count": 5, "score": 0.9},
            {"text": "world", "count": 3, "score": 0.8},
        ]
        batcher = DataBatcher(data, batch_size=2)

        assert "text" in batcher.schema.names
        assert "count" in batcher.schema.names
        assert "score" in batcher.schema.names

    def test_batch_size_larger_than_data(self):
        """Test batch size larger than available data."""
        data = [{"x": 1}, {"x": 2}]
        batcher = DataBatcher(data, batch_size=10)

        batches = list(batcher)

        assert len(batches) == 1
        assert batches[0] == {"x": [1, 2]}

    def test_default_batch_size(self):
        """Test default batch size of 16."""
        data = [{"val": i} for i in range(100)]
        batcher = DataBatcher(data)

        assert batcher._batch_size == 16

    def test_iterator_consumed_once(self):
        """Test that iterator-based batcher can only be iterated once."""

        def data_gen():
            for i in range(3):
                yield {"x": i}

        batcher = DataBatcher(data_gen(), batch_size=2)

        # First iteration
        batches1 = list(batcher)
        assert len(batches1) == 2

        # Second iteration should be empty (generator consumed)
        batches2 = list(batcher)
        assert len(batches2) == 0

    def test_single_row_data(self):
        """Test DataBatcher with single row of data."""
        data = [{"x": 42}]
        batcher = DataBatcher(data, batch_size=10)

        batches = list(batcher)

        assert len(batches) == 1
        assert batches[0] == {"x": [42]}

    def test_consistent_schema_across_batches(self):
        """Test that schema is consistent across all batches."""
        data = [{"a": i, "b": f"val_{i}"} for i in range(10)]
        batcher = DataBatcher(data, batch_size=3)

        expected_names = batcher.schema.names
        for batch in batcher:
            assert list(batch.keys()) == expected_names

    def test_next_method(self):
        """Test __next__ method directly."""
        data = [{"x": 1}, {"x": 2}]
        batcher = DataBatcher(data, batch_size=1)

        # Use next() to get first batch
        first_batch = next(batcher)
        assert first_batch == {"x": [1]}

        # Get second batch
        second_batch = next(batcher)
        assert second_batch == {"x": [2]}

        # Should raise StopIteration when exhausted
        with pytest.raises(StopIteration):
            next(batcher)

    def test_reserved_column_name_inference_raises_error(self):
        """Test that using reserved column name '_inference' raises error."""
        data = [{"text": "hello", "_inference": "not allowed"}]

        with pytest.raises(QueryboostDataColumnError) as exc_info:
            DataBatcher(data, batch_size=1)

        assert "Reserved column names are not allowed" in str(exc_info.value)
        assert "_inference" in str(exc_info.value) or "_error" in str(exc_info.value)

    def test_reserved_column_name_error_raises_error(self):
        """Test that using reserved column name '_error' raises error."""
        data = [{"text": "hello", "_error": "not allowed"}]

        with pytest.raises(QueryboostDataColumnError) as exc_info:
            DataBatcher(data, batch_size=1)

        assert "Reserved column names are not allowed" in str(exc_info.value)
        assert "_inference" in str(exc_info.value) or "_error" in str(exc_info.value)

    def test_multiple_reserved_column_names_raises_error(self):
        """Test that using both reserved column names raises error."""
        data = [{"_inference": "bad", "_error": "also bad", "text": "ok"}]

        with pytest.raises(QueryboostDataColumnError) as exc_info:
            DataBatcher(data, batch_size=1)

        assert "Reserved column names are not allowed" in str(exc_info.value)
