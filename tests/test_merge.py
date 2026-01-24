"""Tests for the merge function."""

import pytest
from pyexp import merge


class TestMerge:
    """Tests for merge function with dot-notation support."""

    def test_simple_merge(self):
        """Basic merge without dot notation."""
        base = {"a": 1, "b": 2}
        update = {"b": 3, "c": 4}
        result = merge(base, update)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_does_not_mutate_original(self):
        """Merge should not modify the original dict."""
        base = {"a": 1}
        update = {"b": 2}
        result = merge(base, update)
        assert base == {"a": 1}
        assert result == {"a": 1, "b": 2}

    def test_dot_notation_set_nested(self):
        """Dot notation should set nested values."""
        base = {"mlp": {"width": 32, "depth": 2}}
        update = {"mlp.width": 64}
        result = merge(base, update)
        assert result == {"mlp": {"width": 64, "depth": 2}}

    def test_dot_notation_deep_nested(self):
        """Dot notation should work with deeply nested paths."""
        base = {"a": {"b": {"c": {"d": 1}}}}
        update = {"a.b.c.d": 42}
        result = merge(base, update)
        assert result == {"a": {"b": {"c": {"d": 42}}}}

    def test_dot_notation_creates_intermediate_dicts(self):
        """Dot notation should create intermediate dicts if they don't exist."""
        base = {"a": 1}
        update = {"b.c.d": 2}
        result = merge(base, update)
        assert result == {"a": 1, "b": {"c": {"d": 2}}}

    def test_dot_notation_set_none(self):
        """Setting a value to None should work."""
        base = {"mlp": {"encoding": {"type": "Sin", "octaves": 4}}}
        update = {"mlp.encoding": None}
        result = merge(base, update)
        assert result == {"mlp": {"encoding": None}}

    def test_dict_replacement_not_deep_merge(self):
        """Setting a dict value should replace, not deep merge."""
        base = {"mlp": {"encoding": {"type": "Sin", "octaves": 4}}}
        update = {"mlp.encoding": {"type": "Tri", "n_funcs": 4}}
        result = merge(base, update)
        # Should NOT have "octaves" - full replacement
        assert result == {"mlp": {"encoding": {"type": "Tri", "n_funcs": 4}}}

    def test_top_level_dict_replacement(self):
        """Top-level dict replacement should also not deep merge."""
        base = {"mlp": {"width": 32, "depth": 2}}
        update = {"mlp": {"width": 64}}
        result = merge(base, update)
        # Should NOT have "depth" - full replacement
        assert result == {"mlp": {"width": 64}}

    def test_mixed_dot_notation_and_regular(self):
        """Can mix dot notation and regular keys."""
        base = {"a": 1, "b": {"c": 2}}
        update = {"a": 10, "b.c": 20, "d": 30}
        result = merge(base, update)
        assert result == {"a": 10, "b": {"c": 20}, "d": 30}

    def test_dot_notation_with_new_nested_key(self):
        """Dot notation can add new keys to existing nested dicts."""
        base = {"mlp": {"width": 32}}
        update = {"mlp.depth": 4}
        result = merge(base, update)
        assert result == {"mlp": {"width": 32, "depth": 4}}

    def test_empty_base(self):
        """Merge with empty base dict."""
        base = {}
        update = {"a.b": 1}
        result = merge(base, update)
        assert result == {"a": {"b": 1}}

    def test_empty_update(self):
        """Merge with empty update dict."""
        base = {"a": 1}
        update = {}
        result = merge(base, update)
        assert result == {"a": 1}

    def test_overwrite_non_dict_with_nested(self):
        """Can overwrite a non-dict value when setting nested path."""
        base = {"a": 1}
        update = {"a.b": 2}
        result = merge(base, update)
        assert result == {"a": {"b": 2}}

    def test_multiple_dot_notation_same_parent(self):
        """Multiple dot notation updates to same parent."""
        base = {"mlp": {"width": 32, "depth": 2}}
        update = {"mlp.width": 64, "mlp.depth": 4}
        result = merge(base, update)
        assert result == {"mlp": {"width": 64, "depth": 4}}

    def test_preserves_unrelated_nested_values(self):
        """Updating one nested path shouldn't affect siblings."""
        base = {
            "mlp": {
                "width": 32,
                "encoding": {"type": "Sin", "octaves": 4}
            }
        }
        update = {"mlp.width": 64}
        result = merge(base, update)
        assert result["mlp"]["width"] == 64
        assert result["mlp"]["encoding"] == {"type": "Sin", "octaves": 4}
