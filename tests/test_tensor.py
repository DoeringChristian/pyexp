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


class TestTensorPatternMatching:
    """Tests for Tensor pattern-based name matching."""

    def test_exact_match_single_result(self):
        tensor = Tensor([{"name": "a"}, {"name": "b"}])
        result = tensor["a"]
        assert result == {"name": "a"}

    def test_exact_match_returns_single_element(self):
        tensor = Tensor(
            [{"name": "a_x"}, {"name": "a_y"}, {"name": "b_x"}, {"name": "b_y"}],
            shape=(2, 2),
        )
        result = tensor["a_x"]
        assert result == {"name": "a_x"}

    def test_wildcard_match(self):
        tensor = Tensor(
            [{"name": "a_x"}, {"name": "a_y"}, {"name": "b_x"}, {"name": "b_y"}],
            shape=(2, 2),
        )
        result = tensor["a_*"]
        assert result.shape == (1, 2)
        assert [c["name"] for c in result] == ["a_x", "a_y"]

    def test_wildcard_second_dimension(self):
        tensor = Tensor(
            [{"name": "a_x"}, {"name": "a_y"}, {"name": "b_x"}, {"name": "b_y"}],
            shape=(2, 2),
        )
        result = tensor["*_x"]
        assert result.shape == (2, 1)
        assert [c["name"] for c in result] == ["a_x", "b_x"]

    def test_pattern_preserves_structure(self):
        """Pattern matching should preserve tensor structure."""
        tensor = Tensor(
            [
                {"name": "e_a_x"}, {"name": "e_a_y"},
                {"name": "e_b_x"}, {"name": "e_b_y"},
            ],
            shape=(1, 2, 2),
        )
        result = tensor["e_a_*"]
        assert result.shape == (1, 1, 2)

    def test_pattern_no_match_raises(self):
        tensor = Tensor([{"name": "a"}, {"name": "b"}])
        with pytest.raises(IndexError, match="No configs match pattern"):
            _ = tensor["xyz"]

    def test_pattern_with_question_mark(self):
        tensor = Tensor([{"name": "a1"}, {"name": "a2"}, {"name": "b1"}])
        result = tensor["a?"]
        assert result.shape == (2,)
        assert [c["name"] for c in result] == ["a1", "a2"]

    def test_pattern_all_match(self):
        tensor = Tensor(
            [{"name": "x_a"}, {"name": "x_b"}, {"name": "x_c"}, {"name": "x_d"}],
            shape=(2, 2),
        )
        result = tensor["x_*"]
        assert result.shape == (2, 2)

    def test_pattern_on_non_dict_no_match(self):
        """Pattern matching on non-dict items won't match specific patterns."""
        tensor = Tensor([1, 2, 3])
        with pytest.raises(IndexError, match="No configs match pattern"):
            _ = tensor["some_name"]


class TestTensorDictMatching:
    """Tests for Tensor dict-based matching."""

    def test_simple_dict_match(self):
        tensor = Tensor([{"x": 1}, {"x": 2}, {"x": 3}])
        result = tensor[{"x": 2}]
        assert result == {"x": 2}

    def test_dict_match_multiple_results(self):
        tensor = Tensor(
            [{"a": 1, "b": 10}, {"a": 1, "b": 20}, {"a": 2, "b": 10}, {"a": 2, "b": 20}],
            shape=(2, 2),
        )
        result = tensor[{"a": 1}]
        assert result.shape == (1, 2)
        assert [c["b"] for c in result] == [10, 20]

    def test_dict_match_second_dimension(self):
        tensor = Tensor(
            [{"a": 1, "b": 10}, {"a": 1, "b": 20}, {"a": 2, "b": 10}, {"a": 2, "b": 20}],
            shape=(2, 2),
        )
        result = tensor[{"b": 10}]
        assert result.shape == (2, 1)
        assert [c["a"] for c in result] == [1, 2]

    def test_dict_match_multiple_keys(self):
        tensor = Tensor(
            [{"a": 1, "b": 10}, {"a": 1, "b": 20}, {"a": 2, "b": 10}, {"a": 2, "b": 20}],
            shape=(2, 2),
        )
        result = tensor[{"a": 1, "b": 20}]
        assert result == {"a": 1, "b": 20}

    def test_dict_match_dot_notation(self):
        tensor = Tensor([
            {"mlp": {"width": 32}},
            {"mlp": {"width": 64}},
        ])
        result = tensor[{"mlp.width": 64}]
        assert result == {"mlp": {"width": 64}}

    def test_dict_match_nested_dict(self):
        tensor = Tensor([
            {"mlp": {"width": 32, "depth": 2}},
            {"mlp": {"width": 64, "depth": 2}},
        ])
        result = tensor[{"mlp": {"width": 32}}]
        assert result["mlp"]["width"] == 32

    def test_dict_match_no_results_raises(self):
        tensor = Tensor([{"x": 1}, {"x": 2}])
        with pytest.raises(IndexError, match="No configs match query"):
            _ = tensor[{"x": 99}]

    def test_dict_match_preserves_structure(self):
        tensor = Tensor(
            [
                {"a": 0, "b": 0, "c": 0}, {"a": 0, "b": 0, "c": 1},
                {"a": 0, "b": 1, "c": 0}, {"a": 0, "b": 1, "c": 1},
                {"a": 1, "b": 0, "c": 0}, {"a": 1, "b": 0, "c": 1},
                {"a": 1, "b": 1, "c": 0}, {"a": 1, "b": 1, "c": 1},
            ],
            shape=(2, 2, 2),
        )
        result = tensor[{"b": 0}]
        assert result.shape == (2, 1, 2)

    def test_dict_match_missing_key_no_match(self):
        tensor = Tensor([{"x": 1}, {"y": 2}])
        with pytest.raises(IndexError, match="No configs match query"):
            _ = tensor[{"z": 1}]

    def test_dict_match_deep_dot_notation(self):
        tensor = Tensor([
            {"a": {"b": {"c": 1}}},
            {"a": {"b": {"c": 2}}},
        ])
        result = tensor[{"a.b.c": 1}]
        assert result["a"]["b"]["c"] == 1
