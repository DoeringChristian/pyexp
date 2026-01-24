"""Tests for the Config class."""

import pytest
from pyexp import Config


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
