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


class TestStarStarMerge:
    """Tests for ** deep-merge prefix."""

    def test_deep_merge_basic(self):
        """** should merge dict value into existing dict."""
        base = {"bsdf": {"type": "Diffuse", "color": 5, "roughness": 0.1}}
        result = merge(base, {"**bsdf": {"type": "Test", "color": 10}})
        assert result == {"bsdf": {"type": "Test", "color": 10, "roughness": 0.1}}

    def test_deep_merge_nested_path(self):
        """** should work with dot-notation paths."""
        base = {"a": {"b": {"x": 1, "y": 2, "z": 3}}}
        result = merge(base, {"**a.b": {"x": 10, "y": 20}})
        assert result == {"a": {"b": {"x": 10, "y": 20, "z": 3}}}

    def test_deep_merge_equivalent_to_dot_notation(self):
        """** merge should produce the same result as individual dot-notation keys."""
        base = {"bsdf": {"bsdf": {"type": "Diffuse", "color": 5, "roughness": 0.1}}}
        r1 = merge(base, {"**bsdf.bsdf": {"type": "Test", "color": 10}})
        r2 = merge(base, {"bsdf.bsdf.type": "Test", "bsdf.bsdf.color": 10})
        assert r1 == r2

    def test_deep_merge_adds_new_keys(self):
        """** should add new keys to the target dict."""
        base = {"mlp": {"width": 32}}
        result = merge(base, {"**mlp": {"depth": 4, "bias": True}})
        assert result == {"mlp": {"width": 32, "depth": 4, "bias": True}}

    def test_deep_merge_does_not_mutate_base(self):
        """** merge should not mutate the base dict."""
        base = {"mlp": {"width": 32, "depth": 2}}
        result = merge(base, {"**mlp": {"width": 64}})
        assert base == {"mlp": {"width": 32, "depth": 2}}
        assert result == {"mlp": {"width": 64, "depth": 2}}

    def test_deep_merge_creates_intermediate_dicts(self):
        """** should create intermediate dicts if the path doesn't exist."""
        base = {"a": 1}
        result = merge(base, {"**b.c": {"x": 1, "y": 2}})
        assert result == {"a": 1, "b": {"c": {"x": 1, "y": 2}}}

    def test_deep_merge_top_level(self):
        """** with a simple key (no dots) should merge into top-level dict."""
        base = {"mlp": {"width": 32, "depth": 2}}
        result = merge(base, {"**mlp": {"width": 64}})
        assert result == {"mlp": {"width": 64, "depth": 2}}

    def test_deep_merge_requires_dict_value(self):
        """** should raise ValueError if the value is not a dict."""
        base = {"a": {"x": 1}}
        with pytest.raises(ValueError, match="requires a dict value"):
            merge(base, {"**a": 42})

    def test_deep_merge_mixed_with_regular_keys(self):
        """** can be mixed with regular and dot-notation keys."""
        base = {"mlp": {"width": 32, "depth": 2}, "lr": 0.01}
        result = merge(base, {
            "**mlp": {"width": 64, "bias": True},
            "lr": 0.1,
        })
        assert result == {"mlp": {"width": 64, "depth": 2, "bias": True}, "lr": 0.1}

    def test_deep_merge_is_shallow(self):
        """** only merges one level deep — nested dicts are replaced, not merged."""
        base = {
            "bsdf": {
                "type": "Diffuse",
                "color": 5,
                "mlp": {"type": "OldMLP", "width": 16, "depth": 4},
            }
        }
        result = merge(base, {
            "**bsdf": {"type": "Test", "mlp": {"type": "MLP", "width": 32}},
        })
        # color is preserved (** merges at the bsdf level)
        assert result["bsdf"]["color"] == 5
        # mlp is fully replaced (** does NOT recurse into sub-dicts)
        assert result["bsdf"]["mlp"] == {"type": "MLP", "width": 32}

    def test_without_star_star_replaces(self):
        """Contrast: without ** the dict is replaced entirely."""
        base = {"bsdf": {"type": "Diffuse", "color": 5, "roughness": 0.1}}
        result = merge(base, {"bsdf": {"type": "Test", "color": 10}})
        # roughness is gone
        assert result == {"bsdf": {"type": "Test", "color": 10}}
