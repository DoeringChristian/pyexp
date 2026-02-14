"""Tests for the sweep function."""

import pytest
from pyexp import sweep, Runs


class TestSweep:
    """Tests for sweep function that generates config combinations."""

    def test_sweep_from_list(self):
        configs = [{"name": "exp"}]
        result = sweep(configs, [{"lr": 0.1}, {"lr": 0.01}])
        assert isinstance(result, Runs)
        assert len(result) == 2

    def test_sweep_creates_cartesian_product(self):
        configs = [{"name": "exp"}]
        result = sweep(configs, [{"lr": 0.1}, {"lr": 0.01}])
        assert result[0] == {"name": "exp", "lr": 0.1}
        assert result[1] == {"name": "exp", "lr": 0.01}

    def test_sweep_overrides_values(self):
        configs = [{"lr": 0.001}]
        result = sweep(configs, [{"lr": 0.1}])
        assert result[0]["lr"] == 0.1

    def test_sweep_multiple_base_configs(self):
        configs = [{"name": "a"}, {"name": "b"}]
        result = sweep(configs, [{"x": 1}, {"x": 2}])
        assert len(result) == 4
        assert result[0] == {"name": "a", "x": 1}
        assert result[1] == {"name": "a", "x": 2}
        assert result[2] == {"name": "b", "x": 1}
        assert result[3] == {"name": "b", "x": 2}

    def test_sweep_chained(self):
        configs = [{"name": "exp"}]
        configs = sweep(configs, [{"lr": 0.1}, {"lr": 0.01}])
        configs = sweep(configs, [{"epochs": 10}, {"epochs": 20}])
        assert len(configs) == 4

    def test_sweep_chained_values(self):
        configs = [{"name": "exp"}]
        configs = sweep(configs, [{"lr": 0.1}, {"lr": 0.01}])
        configs = sweep(configs, [{"epochs": 10}, {"epochs": 20}])

        # Check all combinations exist
        results = list(configs)
        assert {"name": "exp", "lr": 0.1, "epochs": 10} in results
        assert {"name": "exp", "lr": 0.1, "epochs": 20} in results
        assert {"name": "exp", "lr": 0.01, "epochs": 10} in results
        assert {"name": "exp", "lr": 0.01, "epochs": 20} in results

    def test_sweep_from_runs(self):
        configs = Runs([{"a": 1}, {"a": 2}])
        result = sweep(configs, [{"b": 10}, {"b": 20}])
        assert len(result) == 4

    def test_sweep_multiple_params_per_variation(self):
        configs = [{"name": "exp"}]
        result = sweep(
            configs,
            [
                {"lr": 0.1, "batch_size": 32},
                {"lr": 0.01, "batch_size": 64},
            ],
        )
        assert result[0] == {"name": "exp", "lr": 0.1, "batch_size": 32}
        assert result[1] == {"name": "exp", "lr": 0.01, "batch_size": 64}

    def test_sweep_single_variation(self):
        configs = [{"a": 1}]
        result = sweep(configs, [{"b": 2}])
        assert len(result) == 1
        assert result[0] == {"a": 1, "b": 2}

    def test_sweep_empty_variations(self):
        configs = [{"a": 1}]
        result = sweep(configs, [])
        assert len(result) == 0

    def test_sweep_with_dot_notation(self):
        """Sweep should support dot-notation for nested updates."""
        configs = [{"mlp": {"width": 32, "depth": 2}}]
        result = sweep(configs, [{"mlp.width": 64}, {"mlp.width": 128}])
        assert result[0]["mlp"]["width"] == 64
        assert result[0]["mlp"]["depth"] == 2  # Preserved
        assert result[1]["mlp"]["width"] == 128
        assert result[1]["mlp"]["depth"] == 2  # Preserved

    def test_sweep_dot_notation_replaces_dict(self):
        """Dot notation dict replacement should not deep merge."""
        configs = [{"mlp": {"encoding": {"type": "Sin", "octaves": 4}}}]
        result = sweep(
            configs,
            [
                {"mlp.encoding": None},
                {"mlp.encoding": {"type": "Tri", "n_funcs": 4}},
            ],
        )
        assert result[0]["mlp"]["encoding"] is None
        assert result[1]["mlp"]["encoding"] == {"type": "Tri", "n_funcs": 4}
        assert "octaves" not in result[1]["mlp"]["encoding"]  # Not merged!


class TestSweepNames:
    """Tests for sweep name combination and pattern matching."""

    def test_sweep_combines_names(self):
        configs = [{"name": "exp"}]
        result = sweep(configs, [{"name": "lr0.1"}, {"name": "lr0.2"}])
        assert result[0]["name"] == "exp_lr0.1"
        assert result[1]["name"] == "exp_lr0.2"

    def test_sweep_chained_combines_all_names(self):
        configs = [{"name": "exp"}]
        configs = sweep(configs, [{"name": "lr0.1"}, {"name": "lr0.2"}])
        configs = sweep(configs, [{"name": "e10"}, {"name": "e20"}])

        # All 4 configs have combined names
        assert configs[0]["name"] == "exp_lr0.1_e10"
        assert configs[1]["name"] == "exp_lr0.1_e20"
        assert configs[2]["name"] == "exp_lr0.2_e10"
        assert configs[3]["name"] == "exp_lr0.2_e20"

    def test_sweep_pattern_matching(self):
        configs = [{"name": "exp"}]
        configs = sweep(
            configs, [{"name": "a", "x": 1}, {"name": "b", "x": 2}]
        )
        configs = sweep(
            configs, [{"name": "0", "y": 10}, {"name": "1", "y": 20}]
        )

        # Pattern match
        result = configs["exp_a_*"]
        assert len(result) == 2
        assert [c["name"] for c in result] == ["exp_a_0", "exp_a_1"]

    def test_sweep_exact_name_match(self):
        configs = [{"name": "exp"}]
        configs = sweep(
            configs, [{"name": "a", "x": 1}, {"name": "b", "x": 2}]
        )
        cfg = configs["exp_a"]
        assert cfg["x"] == 1

    def test_sweep_variation_without_name_keeps_base(self):
        """Variation without name keeps base name unchanged."""
        configs = [{"name": "exp"}]
        result = sweep(configs, [{"lr": 0.1}, {"lr": 0.2}])
        assert result[0]["name"] == "exp"
        assert result[1]["name"] == "exp"

    def test_sweep_base_without_name_uses_variation(self):
        """Base without name uses variation name directly."""
        configs = [{"x": 1}]
        result = sweep(configs, [{"name": "a"}, {"name": "b"}])
        assert result[0]["name"] == "a"
        assert result[1]["name"] == "b"

    def test_sweep_pattern_second_dimension(self):
        configs = [{"name": "exp"}]
        configs = sweep(configs, [{"name": "a"}, {"name": "b"}])
        configs = sweep(configs, [{"name": "x"}, {"name": "y"}])

        result = configs["*_x"]
        assert len(result) == 2
        assert [c["name"] for c in result] == ["exp_a_x", "exp_b_x"]
