"""Subprocess worker for running isolated experiments.

This module is invoked as a subprocess to run a single experiment in isolation.
It receives a serialized payload (function + config) via a temp file, executes
the experiment, and writes the result to the designated output path.

Usage:
    python -m pyexp.worker <payload_path>

The payload file contains cloudpickle-serialized data:
    {
        "fn": <experiment function>,
        "config": <config dict>,
        "result_path": <path to write result>,
    }
"""

import pickle
import sys
import traceback
from pathlib import Path

import cloudpickle


def run_worker(payload_path: str) -> int:
    """Run a single experiment from a serialized payload.

    Args:
        payload_path: Path to the cloudpickle-serialized payload file.

    Returns:
        Exit code: 0 for success, 1 for failure.

    The worker saves a structured result dict:
        {"result": <return value or None>, "error": <error string or None>}
    """
    from pyexp.log import Logger

    logger = None
    try:
        # Load the payload
        with open(payload_path, "rb") as f:
            payload = cloudpickle.load(f)

        fn = payload["fn"]
        config = payload["config"]
        result_path = Path(payload["result_path"])

        # Create logger for this experiment's output directory
        logger = Logger(config["out"])

        # Log config as YAML at iteration 0
        import yaml
        config_to_log = {k: v for k, v in config.items() if not k.startswith("_") and k not in ("out", "logger")}
        logger.add_text("config", yaml.dump(config_to_log, default_flow_style=False))

        # Log git commit hash if stash enabled
        stash_enabled = config.get("_stash", True)
        if stash_enabled:
            try:
                from pyexp.utils import stash as git_stash
                commit_hash = git_stash()
                logger.add_text("git_commit", commit_hash)
            except Exception:
                pass  # Silently ignore if not in a git repo

        config = config.__class__({**config, "logger": logger})

        # Run the experiment
        result = fn(config)

        # Flush logger before writing result
        logger.flush()

        # Write structured result
        structured = {"result": result, "error": None}
        result_path.parent.mkdir(parents=True, exist_ok=True)
        with open(result_path, "wb") as f:
            pickle.dump(structured, f)

        return 0

    except Exception as e:
        # Flush logger if it exists
        if logger:
            try:
                logger.flush()
            except Exception:
                pass

        # Write error information to result path if possible
        try:
            error_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            structured = {"result": None, "error": error_msg}
            result_path = Path(payload["result_path"])
            result_path.parent.mkdir(parents=True, exist_ok=True)
            with open(result_path, "wb") as f:
                pickle.dump(structured, f)
        except Exception:
            # If we can't even write the error, just print it
            traceback.print_exc()

        return 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python -m pyexp.worker <payload_path>", file=sys.stderr)
        sys.exit(2)

    sys.exit(run_worker(sys.argv[1]))
