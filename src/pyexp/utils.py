"""Utility functions for pyexp."""

import os
import subprocess
import tempfile
from pathlib import Path


def stash() -> str:
    """Capture the current git repository state in a detached commit for reproducibility.

    Creates a temporary commit containing all tracked and untracked files without
    affecting the working tree or current HEAD. This allows exact reproducibility
    of training runs.

    Returns:
        Git commit hash of the snapshot.
    """
    path = Path.cwd()
    path = path.resolve()

    # Save current HEAD (to restore later)
    head = (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path).decode().strip()
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create temporary index and tree
        env = os.environ.copy()
        env["GIT_INDEX_FILE"] = str(Path(tmpdir) / "index")

        # Add all files (including untracked)
        subprocess.check_call(
            ["git", "add", "-A", "--intent-to-add"], cwd=path, env=env
        )
        subprocess.check_call(["git", "add", "-A"], cwd=path, env=env)

        # Write tree object
        tree_hash = (
            subprocess.check_output(["git", "write-tree"], cwd=path, env=env)
            .decode()
            .strip()
        )

        # Create a detached commit object (not checked out)
        commit_hash = (
            subprocess.check_output(
                [
                    "git",
                    "commit-tree",
                    tree_hash,
                    "-p",
                    head,
                    "-m",
                    "Temporary reproducible snapshot",
                ],
                cwd=path,
                env=env,
            )
            .decode()
            .strip()
        )

    return commit_hash
