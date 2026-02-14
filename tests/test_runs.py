"""Tests for the Runs class."""

import pytest
from pyexp import Runs, sweep


class TestRuns:
    """Tests for Runs collection and indexing."""

    def test_create_from_list(self):
        runs = Runs([{"a": 1}, {"a": 2}, {"a": 3}])
        assert len(runs) == 3

    def test_iteration(self):
        data = [{"a": 1}, {"a": 2}]
        runs = Runs(data)
        assert list(runs) == data

    def test_flat_integer_indexing(self):
        runs = Runs([{"a": 1}, {"a": 2}, {"a": 3}])
        assert runs[0] == {"a": 1}
        assert runs[1] == {"a": 2}
        assert runs[2] == {"a": 3}

    def test_negative_flat_indexing(self):
        runs = Runs([{"a": 1}, {"a": 2}, {"a": 3}])
        assert runs[-1] == {"a": 3}
        assert runs[-2] == {"a": 2}

    def test_tolist(self):
        data = [{"a": 1}, {"a": 2}]
        runs = Runs(data)
        assert runs.tolist() == data

    def test_repr(self):
        runs = Runs([{"a": 1}, {"a": 2}])
        assert "len=2" in repr(runs)
        assert "Runs" in repr(runs)

    def test_works_with_non_dict_data(self):
        """Runs should work with any data type, not just dicts."""
        runs = Runs([1, 2, 3, 4])
        assert runs[0] == 1
        assert runs[3] == 4
        assert runs.tolist() == [1, 2, 3, 4]


class TestRunsPatternMatching:
    """Tests for Runs pattern-based name matching."""

    def test_exact_match_single_result(self):
        runs = Runs([{"name": "a"}, {"name": "b"}])
        result = runs["a"]
        assert result == {"name": "a"}

    def test_exact_match_returns_single_element(self):
        runs = Runs(
            [{"name": "a_x"}, {"name": "a_y"}, {"name": "b_x"}, {"name": "b_y"}],
        )
        result = runs["a_x"]
        assert result == {"name": "a_x"}

    def test_wildcard_match(self):
        runs = Runs(
            [{"name": "a_x"}, {"name": "a_y"}, {"name": "b_x"}, {"name": "b_y"}],
        )
        result = runs["a_*"]
        assert len(result) == 2
        assert [c["name"] for c in result] == ["a_x", "a_y"]

    def test_wildcard_second_dimension(self):
        runs = Runs(
            [{"name": "a_x"}, {"name": "a_y"}, {"name": "b_x"}, {"name": "b_y"}],
        )
        result = runs["*_x"]
        assert len(result) == 2
        assert [c["name"] for c in result] == ["a_x", "b_x"]

    def test_pattern_no_match_raises(self):
        runs = Runs([{"name": "a"}, {"name": "b"}])
        with pytest.raises(IndexError, match="No configs match pattern"):
            _ = runs["xyz"]

    def test_pattern_with_question_mark(self):
        runs = Runs([{"name": "a1"}, {"name": "a2"}, {"name": "b1"}])
        result = runs["a?"]
        assert len(result) == 2
        assert [c["name"] for c in result] == ["a1", "a2"]

    def test_pattern_all_match(self):
        runs = Runs(
            [{"name": "x_a"}, {"name": "x_b"}, {"name": "x_c"}, {"name": "x_d"}],
        )
        result = runs["x_*"]
        assert len(result) == 4

    def test_pattern_on_non_dict_no_match(self):
        """Pattern matching on non-dict items won't match specific patterns."""
        runs = Runs([1, 2, 3])
        with pytest.raises(IndexError, match="No configs match pattern"):
            _ = runs["some_name"]


class TestRunsDictMatching:
    """Tests for Runs dict-based matching."""

    def test_simple_dict_match(self):
        runs = Runs([{"x": 1}, {"x": 2}, {"x": 3}])
        result = runs[{"x": 2}]
        assert result == {"x": 2}

    def test_dict_match_multiple_results(self):
        runs = Runs(
            [{"a": 1, "b": 10}, {"a": 1, "b": 20}, {"a": 2, "b": 10}, {"a": 2, "b": 20}],
        )
        result = runs[{"a": 1}]
        assert len(result) == 2
        assert [c["b"] for c in result] == [10, 20]

    def test_dict_match_second_key(self):
        runs = Runs(
            [{"a": 1, "b": 10}, {"a": 1, "b": 20}, {"a": 2, "b": 10}, {"a": 2, "b": 20}],
        )
        result = runs[{"b": 10}]
        assert len(result) == 2
        assert [c["a"] for c in result] == [1, 2]

    def test_dict_match_multiple_keys(self):
        runs = Runs(
            [{"a": 1, "b": 10}, {"a": 1, "b": 20}, {"a": 2, "b": 10}, {"a": 2, "b": 20}],
        )
        result = runs[{"a": 1, "b": 20}]
        assert result == {"a": 1, "b": 20}

    def test_dict_match_dot_notation(self):
        runs = Runs([
            {"mlp": {"width": 32}},
            {"mlp": {"width": 64}},
        ])
        result = runs[{"mlp.width": 64}]
        assert result == {"mlp": {"width": 64}}

    def test_dict_match_nested_dict(self):
        runs = Runs([
            {"mlp": {"width": 32, "depth": 2}},
            {"mlp": {"width": 64, "depth": 2}},
        ])
        result = runs[{"mlp": {"width": 32}}]
        assert result["mlp"]["width"] == 32

    def test_dict_match_no_results_raises(self):
        runs = Runs([{"x": 1}, {"x": 2}])
        with pytest.raises(IndexError, match="No configs match query"):
            _ = runs[{"x": 99}]

    def test_dict_match_missing_key_no_match(self):
        runs = Runs([{"x": 1}, {"y": 2}])
        with pytest.raises(IndexError, match="No configs match query"):
            _ = runs[{"z": 1}]

    def test_dict_match_deep_dot_notation(self):
        runs = Runs([
            {"a": {"b": {"c": 1}}},
            {"a": {"b": {"c": 2}}},
        ])
        result = runs[{"a.b.c": 1}]
        assert result["a"]["b"]["c"] == 1
