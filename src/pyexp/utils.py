"""Utility functions for pyexp."""

import os
import subprocess
import tempfile
from pathlib import Path


def _find_git_root() -> Path:
    """Find the root directory of the current git repository.

    Returns:
        Path to the git repository root.

    Raises:
        subprocess.CalledProcessError: If not inside a git repository.
    """
    root = (
        subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], cwd=Path.cwd()
        )
        .decode()
        .strip()
    )
    return Path(root)


def stash() -> str:
    """Capture the current git repository state in a detached commit for reproducibility.

    Creates a temporary commit containing all tracked and untracked files without
    affecting the working tree or current HEAD. This allows exact reproducibility
    of training runs.

    Returns:
        Git commit hash of the snapshot.
    """
    path = _find_git_root()

    # Try to get current HEAD (may not exist in a fresh repo with no commits)
    try:
        head = (
            subprocess.check_output(
                ["git", "rev-parse", "--verify", "HEAD"],
                cwd=path,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except subprocess.CalledProcessError:
        head = None

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
        # If HEAD exists, parent the snapshot to it; otherwise create a root commit
        cmd = ["git", "commit-tree", tree_hash, "-m", "Temporary reproducible snapshot"]
        if head is not None:
            cmd[3:3] = ["-p", head]

        commit_hash = (
            subprocess.check_output(cmd, cwd=path, env=env).decode().strip()
        )

    return commit_hash


def create_worktree(commit_hash: str, worktree_dir: Path) -> Path:
    """Create a git worktree at the given directory for the specified commit.

    Args:
        commit_hash: Git commit hash to check out in the worktree.
        worktree_dir: Path where the worktree should be created.

    Returns:
        Path to the created worktree directory.

    Raises:
        subprocess.CalledProcessError: If git worktree creation fails.
    """
    git_root = _find_git_root()
    worktree_dir = worktree_dir.resolve()
    subprocess.check_call(
        ["git", "worktree", "add", "--detach", str(worktree_dir), commit_hash],
        cwd=git_root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return worktree_dir


def remove_worktree(worktree_dir: Path) -> None:
    """Remove a git worktree.

    Args:
        worktree_dir: Path to the worktree to remove.

    Raises:
        subprocess.CalledProcessError: If git worktree removal fails.
    """
    git_root = _find_git_root()
    subprocess.check_call(
        ["git", "worktree", "remove", "--force", str(worktree_dir.resolve())],
        cwd=git_root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def stash_and_worktree(worktree_dir: Path) -> tuple[str, Path]:
    """Convenience: stash current state and create a worktree from that snapshot.

    Args:
        worktree_dir: Path where the worktree should be created.

    Returns:
        Tuple of (commit_hash, worktree_path).
    """
    commit_hash = stash()
    worktree_path = create_worktree(commit_hash, worktree_dir)
    return commit_hash, worktree_path
