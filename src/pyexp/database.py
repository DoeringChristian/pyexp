"""Persistent storage for task results.

Provides a :class:`Database` protocol and a :class:`FileDatabase`
implementation that writes results to disk as pickled files.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@dataclass
class Entry:
    """A single persisted result entry."""

    key: str
    result: Any = None
    timestamp: str = ""
    log: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Database(Protocol):
    """Protocol for result persistence backends."""

    def save(
        self, key: str, value: Any, *, log: str = "", metadata: dict | None = None
    ) -> Entry: ...

    def load(self, key: str) -> list[Entry]: ...


class FileDatabase:
    """File-based database that stores results under ``<base_dir>/<key>/<timestamp>/``."""

    def __init__(self, base_dir: str | Path = "out") -> None:
        self.base_dir = Path(base_dir)

    def save(
        self, key: str, value: Any, *, log: str = "", metadata: dict | None = None
    ) -> Entry:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = self.base_dir / key / ts
        run_dir.mkdir(parents=True, exist_ok=True)

        with open(run_dir / "result.pkl", "wb") as f:
            pickle.dump(value, f)

        if log:
            (run_dir / "log.out").write_text(log)

        meta = metadata or {}
        if meta:
            (run_dir / "metadata.json").write_text(json.dumps(meta))

        return Entry(key=key, result=value, timestamp=ts, log=log, metadata=meta)

    def load(self, key: str) -> list[Entry]:
        key_dir = self.base_dir / key
        if not key_dir.is_dir():
            return []

        entries: list[Entry] = []
        for ts_dir in sorted(key_dir.iterdir()):
            if not ts_dir.is_dir():
                continue
            result_file = ts_dir / "result.pkl"
            if not result_file.exists():
                continue

            with open(result_file, "rb") as f:
                result = pickle.load(f)

            log = ""
            log_file = ts_dir / "log.out"
            if log_file.exists():
                log = log_file.read_text()

            meta: dict[str, Any] = {}
            meta_file = ts_dir / "metadata.json"
            if meta_file.exists():
                meta = json.loads(meta_file.read_text())

            entries.append(
                Entry(
                    key=key,
                    result=result,
                    timestamp=ts_dir.name,
                    log=log,
                    metadata=meta,
                )
            )

        return entries


# ---------------------------------------------------------------------------
# Global default
# ---------------------------------------------------------------------------

_default_database: Database | None = None


def set_default_database(db: Database | None) -> None:
    """Set the global default database."""
    global _default_database
    _default_database = db


def get_default_database() -> Database:
    """Return the global default database, creating a :class:`FileDatabase` if unset."""
    global _default_database
    if _default_database is None:
        _default_database = FileDatabase()
    return _default_database
