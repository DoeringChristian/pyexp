"""Tests for the database persistence layer."""

import pytest

from pyexp.database import (
    Entry,
    FileDatabase,
    get_default_database,
    set_default_database,
)


@pytest.fixture(autouse=True)
def _reset_db():
    yield
    set_default_database(None)


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------


def test_entry_defaults():
    e = Entry(key="abc")
    assert e.key == "abc"
    assert e.result is None
    assert e.timestamp == ""
    assert e.log == ""
    assert e.metadata == {}


def test_entry_with_values():
    e = Entry(key="k", result=42, timestamp="2025-01-01_00-00-00", log="ok", metadata={"a": 1})
    assert e.result == 42
    assert e.metadata == {"a": 1}


# ---------------------------------------------------------------------------
# FileDatabase
# ---------------------------------------------------------------------------


def test_save_and_load_roundtrip(tmp_path):
    db = FileDatabase(tmp_path / "db")
    db.save("mykey", {"value": 123})
    entries = db.load("mykey")
    assert len(entries) == 1
    assert entries[0].result == {"value": 123}
    assert entries[0].key == "mykey"
    assert entries[0].timestamp != ""


def test_multiple_saves_same_key(tmp_path):
    db = FileDatabase(tmp_path / "db")
    db.save("k", 1)
    import time; time.sleep(1.1)  # ensure different timestamp
    db.save("k", 2)
    entries = db.load("k")
    assert len(entries) == 2
    assert entries[0].result == 1  # oldest first
    assert entries[1].result == 2


def test_load_nonexistent_key(tmp_path):
    db = FileDatabase(tmp_path / "db")
    assert db.load("missing") == []


def test_log_persistence(tmp_path):
    db = FileDatabase(tmp_path / "db")
    db.save("k", "val", log="some output")
    entries = db.load("k")
    assert entries[0].log == "some output"


def test_empty_log_no_file(tmp_path):
    db = FileDatabase(tmp_path / "db")
    db.save("k", "val")
    # No log.out file should be created
    key_dir = tmp_path / "db" / "k"
    ts_dirs = list(key_dir.iterdir())
    assert not (ts_dirs[0] / "log.out").exists()


def test_metadata_persistence(tmp_path):
    db = FileDatabase(tmp_path / "db")
    db.save("k", "val", metadata={"fn": "mod.func", "extra": 42})
    entries = db.load("k")
    assert entries[0].metadata == {"fn": "mod.func", "extra": 42}


def test_empty_metadata_no_file(tmp_path):
    db = FileDatabase(tmp_path / "db")
    db.save("k", "val")
    key_dir = tmp_path / "db" / "k"
    ts_dirs = list(key_dir.iterdir())
    assert not (ts_dirs[0] / "metadata.json").exists()


def test_directory_structure(tmp_path):
    db = FileDatabase(tmp_path / "db")
    entry = db.save("mykey", 99)
    ts_dir = tmp_path / "db" / "mykey" / entry.timestamp
    assert ts_dir.is_dir()
    assert (ts_dir / "result.pkl").exists()


def test_save_returns_entry(tmp_path):
    db = FileDatabase(tmp_path / "db")
    entry = db.save("k", [1, 2, 3], log="hi", metadata={"x": 1})
    assert isinstance(entry, Entry)
    assert entry.key == "k"
    assert entry.result == [1, 2, 3]
    assert entry.log == "hi"
    assert entry.metadata == {"x": 1}


# ---------------------------------------------------------------------------
# Global default
# ---------------------------------------------------------------------------


def test_get_default_creates_file_database():
    db = get_default_database()
    assert isinstance(db, FileDatabase)


def test_set_default_overrides(tmp_path):
    custom = FileDatabase(tmp_path / "custom")
    set_default_database(custom)
    assert get_default_database() is custom


def test_set_default_none_resets():
    set_default_database(FileDatabase("x"))
    set_default_database(None)
    # Next call creates a fresh FileDatabase
    db = get_default_database()
    assert isinstance(db, FileDatabase)
