"""Tests for the Tensor class."""

import pytest
from pyexp import Tensor, sweep


class TestTensor:
    """Tests for Tensor shape tracking and indexing."""

    def test_create_from_list(self):
        tensor = Tensor([{"a": 1}, {"a": 2}, {"a": 3}])
        assert tensor.shape == (3,)
        assert len(tensor) == 3

    def test_create_with_explicit_shape(self):
        data = [{"a": i} for i in range(6)]
        tensor = Tensor(data, shape=(2, 3))
        assert tensor.shape == (2, 3)
        assert len(tensor) == 6

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="does not match data length"):
            Tensor([{"a": 1}, {"a": 2}], shape=(3,))

    def test_iteration(self):
        data = [{"a": 1}, {"a": 2}]
        tensor = Tensor(data)
        assert list(tensor) == data

    def test_flat_integer_indexing(self):
        tensor = Tensor([{"a": 1}, {"a": 2}, {"a": 3}])
        assert tensor[0] == {"a": 1}
        assert tensor[1] == {"a": 2}
        assert tensor[2] == {"a": 3}

    def test_negative_flat_indexing(self):
        tensor = Tensor([{"a": 1}, {"a": 2}, {"a": 3}])
        assert tensor[-1] == {"a": 3}
        assert tensor[-2] == {"a": 2}

    def test_multidim_single_element(self):
        data = [{"a": i, "b": j} for i in range(2) for j in range(3)]
        tensor = Tensor(data, shape=(2, 3))
        assert tensor[0, 0] == {"a": 0, "b": 0}
        assert tensor[0, 2] == {"a": 0, "b": 2}
        assert tensor[1, 1] == {"a": 1, "b": 1}

    def test_slice_first_dimension(self):
        data = [{"a": i, "b": j} for i in range(2) for j in range(3)]
        tensor = Tensor(data, shape=(2, 3))
        result = tensor[0, :]
        assert result.shape == (3,)
        assert len(result) == 3
        assert result[0] == {"a": 0, "b": 0}

    def test_slice_second_dimension(self):
        data = [{"a": i, "b": j} for i in range(2) for j in range(3)]
        tensor = Tensor(data, shape=(2, 3))
        result = tensor[:, 1]
        assert result.shape == (2,)
        assert result[0] == {"a": 0, "b": 1}
        assert result[1] == {"a": 1, "b": 1}

    def test_slice_both_dimensions(self):
        data = [{"a": i, "b": j} for i in range(3) for j in range(4)]
        tensor = Tensor(data, shape=(3, 4))
        result = tensor[1:3, 0:2]
        assert result.shape == (2, 2)
        assert len(result) == 4

    def test_partial_indexing_pads_with_slices(self):
        data = [{"v": i} for i in range(6)]
        tensor = Tensor(data, shape=(2, 3))
        result = tensor[0]  # Should be flat index, returns single dict
        assert result == {"v": 0}

    def test_tuple_partial_indexing(self):
        data = [{"v": i} for i in range(6)]
        tensor = Tensor(data, shape=(2, 3))
        result = tensor[1,]  # Tuple with one element, pads to (1, :)
        assert result.shape == (3,)

    def test_index_out_of_range(self):
        tensor = Tensor([{"a": 1}], shape=(1,))
        with pytest.raises(IndexError):
            _ = tensor[(5,)]

    def test_too_many_indices(self):
        tensor = Tensor([{"a": 1}], shape=(1,))
        with pytest.raises(IndexError, match="Too many indices"):
            _ = tensor[0, 0]

    def test_tolist(self):
        data = [{"a": 1}, {"a": 2}]
        tensor = Tensor(data)
        assert tensor.tolist() == data

    def test_repr(self):
        tensor = Tensor([{"a": 1}, {"a": 2}], shape=(2,))
        assert "shape=(2,)" in repr(tensor)
        assert "len=2" in repr(tensor)

    def test_3d_indexing(self):
        data = [{"i": i, "j": j, "k": k} for i in range(2) for j in range(3) for k in range(4)]
        tensor = Tensor(data, shape=(2, 3, 4))
        assert len(tensor) == 24

        # Single element
        assert tensor[1, 2, 3] == {"i": 1, "j": 2, "k": 3}

        # Slice last dim
        result = tensor[0, 0, :]
        assert result.shape == (4,)

        # Slice middle dim
        result = tensor[0, :, 0]
        assert result.shape == (3,)

        # Slice first dim
        result = tensor[:, 0, 0]
        assert result.shape == (2,)

    def test_works_with_non_dict_data(self):
        """Tensor should work with any data type, not just dicts."""
        tensor = Tensor([1, 2, 3, 4], shape=(2, 2))
        assert tensor[0, 0] == 1
        assert tensor[1, 1] == 4
        assert tensor[:, 0].tolist() == [1, 3]
