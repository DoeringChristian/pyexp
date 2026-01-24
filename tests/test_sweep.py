"""Tests for the sweep function."""

import pytest
from pyexp import sweep, Tensor


class TestSweep:
    """Tests for sweep function that generates config combinations."""

    def test_sweep_from_list(self):
        configs = [{"name": "exp"}]
        result = sweep(configs, [{"lr": 0.1}, {"lr": 0.01}])
        assert isinstance(result, Tensor)
        assert result.shape == (1, 2)
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
        assert result.shape == (2, 2)
        assert len(result) == 4
        assert result[0] == {"name": "a", "x": 1}
        assert result[1] == {"name": "a", "x": 2}
        assert result[2] == {"name": "b", "x": 1}
        assert result[3] == {"name": "b", "x": 2}

    def test_sweep_chained(self):
        configs = [{"name": "exp"}]
        configs = sweep(configs, [{"lr": 0.1}, {"lr": 0.01}])
        configs = sweep(configs, [{"epochs": 10}, {"epochs": 20}])
        assert configs.shape == (1, 2, 2)
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

    def test_sweep_from_tensor(self):
        configs = Tensor([{"a": 1}, {"a": 2}])
        result = sweep(configs, [{"b": 10}, {"b": 20}])
        assert result.shape == (2, 2)
        assert len(result) == 4

    def test_sweep_preserves_shape_from_tensor(self):
        configs = Tensor([{"a": i} for i in range(6)], shape=(2, 3))
        result = sweep(configs, [{"b": 1}, {"b": 2}])
        assert result.shape == (2, 3, 2)
        assert len(result) == 12

    def test_sweep_multiple_params_per_variation(self):
        configs = [{"name": "exp"}]
        result = sweep(configs, [
            {"lr": 0.1, "batch_size": 32},
            {"lr": 0.01, "batch_size": 64},
        ])
        assert result[0] == {"name": "exp", "lr": 0.1, "batch_size": 32}
        assert result[1] == {"name": "exp", "lr": 0.01, "batch_size": 64}

    def test_sweep_single_variation(self):
        configs = [{"a": 1}]
        result = sweep(configs, [{"b": 2}])
        assert result.shape == (1, 1)
        assert result[0] == {"a": 1, "b": 2}

    def test_sweep_empty_variations(self):
        configs = [{"a": 1}]
        result = sweep(configs, [])
        assert result.shape == (1, 0)
        assert len(result) == 0

    def test_sweep_indexing_after_chain(self):
        configs = [{"name": "exp"}]
        configs = sweep(configs, [{"lr": 0.1}, {"lr": 0.01}])
        configs = sweep(configs, [{"epochs": 10}, {"epochs": 20}])

        # Get all configs with first lr value
        first_lr = configs[:, 0, :]
        assert first_lr.shape == (1, 2)
        for c in first_lr:
            assert c["lr"] == 0.1

        # Get all configs with second epochs value
        second_epochs = configs[:, :, 1]
        assert second_epochs.shape == (1, 2)
        for c in second_epochs:
            assert c["epochs"] == 20

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
        result = sweep(configs, [
            {"mlp.encoding": None},
            {"mlp.encoding": {"type": "Tri", "n_funcs": 4}},
        ])
        assert result[0]["mlp"]["encoding"] is None
        assert result[1]["mlp"]["encoding"] == {"type": "Tri", "n_funcs": 4}
        assert "octaves" not in result[1]["mlp"]["encoding"]  # Not merged!
