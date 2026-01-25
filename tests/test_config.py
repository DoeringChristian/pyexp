"""Tests for the Config class and load_config."""

import pytest
from pyexp import Config, load_config


class TestConfig:
    """Tests for Config dict with dot notation access."""

    def test_create_from_dict(self):
        config = Config({"a": 1, "b": 2})
        assert config["a"] == 1
        assert config["b"] == 2

    def test_dot_notation_access(self):
        config = Config({"learning_rate": 0.01, "epochs": 10})
        assert config.learning_rate == 0.01
        assert config.epochs == 10

    def test_nested_dict_conversion(self):
        config = Config({"optimizer": {"lr": 0.01, "momentum": 0.9}})
        assert isinstance(config["optimizer"], Config)
        assert config.optimizer.lr == 0.01
        assert config.optimizer.momentum == 0.9

    def test_deeply_nested(self):
        config = Config({"a": {"b": {"c": {"d": 42}}}})
        assert config.a.b.c.d == 42

    def test_dot_notation_set(self):
        config = Config({})
        config.new_key = "value"
        assert config["new_key"] == "value"
        assert config.new_key == "value"

    def test_set_nested_dict(self):
        config = Config({})
        config.optimizer = {"lr": 0.01}
        assert isinstance(config.optimizer, Config)
        assert config.optimizer.lr == 0.01

    def test_delete_attribute(self):
        config = Config({"a": 1, "b": 2})
        del config.a
        assert "a" not in config
        assert config.b == 2

    def test_attribute_error_on_missing(self):
        config = Config({"a": 1})
        with pytest.raises(AttributeError, match="Config has no attribute 'missing'"):
            _ = config.missing

    def test_delete_missing_raises(self):
        config = Config({"a": 1})
        with pytest.raises(AttributeError, match="Config has no attribute 'missing'"):
            del config.missing

    def test_dict_methods_still_work(self):
        config = Config({"a": 1, "b": 2})
        assert list(config.keys()) == ["a", "b"]
        assert list(config.values()) == [1, 2]
        assert len(config) == 2
        assert "a" in config

    def test_update_preserves_config_type(self):
        config = Config({"a": 1})
        config["nested"] = {"x": 10}
        assert isinstance(config["nested"], Config)


class TestLoadConfig:
    """Tests for YAML config loading with imports."""

    def test_load_single_file(self, tmp_path):
        """Load a single YAML config file."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("learning_rate: 0.01\nepochs: 10\n")

        config = load_config(config_file)
        assert config.learning_rate == 0.01
        assert config.epochs == 10
        assert isinstance(config, Config)

    def test_load_single_file_string_path(self, tmp_path):
        """Load config using string path."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("value: 42\n")

        config = load_config(str(config_file))
        assert config.value == 42

    def test_load_multiple_files(self, tmp_path):
        """Load and merge multiple config files."""
        base = tmp_path / "base.yaml"
        base.write_text("a: 1\nb: 2\n")

        override = tmp_path / "override.yaml"
        override.write_text("b: 20\nc: 3\n")

        config = load_config([base, override])
        assert config.a == 1
        assert config.b == 20  # Overridden
        assert config.c == 3

    def test_load_with_imports(self, tmp_path):
        """Load config with imports field."""
        base = tmp_path / "base.yaml"
        base.write_text("model:\n  hidden_size: 256\n  num_layers: 4\n")

        experiment = tmp_path / "experiment.yaml"
        experiment.write_text("imports:\n  - base.yaml\nbatch_size: 32\n")

        config = load_config(experiment)
        assert config.model.hidden_size == 256
        assert config.model.num_layers == 4
        assert config.batch_size == 32

    def test_load_with_imports_override(self, tmp_path):
        """Imports can be overridden by the importing file."""
        base = tmp_path / "base.yaml"
        base.write_text("learning_rate: 0.001\nepochs: 100\n")

        experiment = tmp_path / "experiment.yaml"
        experiment.write_text("imports:\n  - base.yaml\nlearning_rate: 0.01\n")

        config = load_config(experiment)
        assert config.learning_rate == 0.01  # Overridden
        assert config.epochs == 100  # From base

    def test_load_with_dot_notation_override(self, tmp_path):
        """Dot notation updates nested values without replacing siblings."""
        base = tmp_path / "base.yaml"
        base.write_text("model:\n  hidden_size: 256\n  num_layers: 4\n")

        experiment = tmp_path / "experiment.yaml"
        experiment.write_text("imports:\n  - base.yaml\nmodel.hidden_size: 512\n")

        config = load_config(experiment)
        assert config.model.hidden_size == 512  # Updated
        assert config.model.num_layers == 4  # Preserved

    def test_load_nested_imports(self, tmp_path):
        """Imports can themselves have imports (recursive)."""
        level1 = tmp_path / "level1.yaml"
        level1.write_text("a: 1\n")

        level2 = tmp_path / "level2.yaml"
        level2.write_text("imports:\n  - level1.yaml\nb: 2\n")

        level3 = tmp_path / "level3.yaml"
        level3.write_text("imports:\n  - level2.yaml\nc: 3\n")

        config = load_config(level3)
        assert config.a == 1
        assert config.b == 2
        assert config.c == 3

    def test_load_multiple_imports(self, tmp_path):
        """Config can import multiple files."""
        base1 = tmp_path / "base1.yaml"
        base1.write_text("a: 1\n")

        base2 = tmp_path / "base2.yaml"
        base2.write_text("b: 2\n")

        experiment = tmp_path / "experiment.yaml"
        experiment.write_text("imports:\n  - base1.yaml\n  - base2.yaml\nc: 3\n")

        config = load_config(experiment)
        assert config.a == 1
        assert config.b == 2
        assert config.c == 3

    def test_load_imports_relative_path(self, tmp_path):
        """Imports are resolved relative to the config file's directory."""
        subdir = tmp_path / "configs"
        subdir.mkdir()

        base = subdir / "base.yaml"
        base.write_text("value: 42\n")

        experiment = subdir / "experiment.yaml"
        experiment.write_text("imports:\n  - base.yaml\nname: test\n")

        config = load_config(experiment)
        assert config.value == 42
        assert config.name == "test"

    def test_load_empty_list(self):
        """Loading empty list returns empty Config."""
        config = load_config([])
        assert config == {}
        assert isinstance(config, Config)

    def test_load_empty_file(self, tmp_path):
        """Loading empty YAML file returns empty Config."""
        empty = tmp_path / "empty.yaml"
        empty.write_text("")

        config = load_config(empty)
        assert config == {}

    def test_load_single_import_as_string(self, tmp_path):
        """Single import can be a string instead of list."""
        base = tmp_path / "base.yaml"
        base.write_text("value: 1\n")

        experiment = tmp_path / "experiment.yaml"
        experiment.write_text("imports: base.yaml\nname: test\n")

        config = load_config(experiment)
        assert config.value == 1
        assert config.name == "test"
