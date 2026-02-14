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

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create temporary index and tree
        env = os.environ.copy()
        env["GIT_INDEX_FILE"] = str(Path(tmpdir) / "index")
        # Ensure author/committer identity for commit-tree (may not be
        # configured in all environments).
        env.setdefault("GIT_AUTHOR_NAME", "pyexp")
        env.setdefault("GIT_AUTHOR_EMAIL", "pyexp@snapshot")
        env.setdefault("GIT_COMMITTER_NAME", "pyexp")
        env.setdefault("GIT_COMMITTER_EMAIL", "pyexp@snapshot")

        # Add all files (including untracked).
        # Suppress stderr to silence warnings about embedded git repos.
        subprocess.check_call(
            ["git", "add", "-A", "--intent-to-add"],
            cwd=path,
            env=env,
            stderr=subprocess.DEVNULL,
        )
        subprocess.check_call(
            ["git", "add", "-A"],
            cwd=path,
            env=env,
            stderr=subprocess.DEVNULL,
        )

        # Write tree object
        tree_hash = (
            subprocess.check_output(["git", "write-tree"], cwd=path, env=env)
            .decode()
            .strip()
        )

        # Create a detached root commit (no parent) so it stays truly
        # dangling and never appears in git log --all or similar.
        commit_hash = (
            subprocess.check_output(
                ["git", "commit-tree", tree_hash, "-m", "pyexp snapshot"],
                cwd=path,
                env=env,
            )
            .decode()
            .strip()
        )

    return commit_hash


def _find_submodule_paths(git_root: Path) -> list[str]:
    """Return relative paths of git submodules in the repository."""
    try:
        output = subprocess.check_output(
            ["git", "submodule", "status"],
            cwd=git_root,
            stderr=subprocess.DEVNULL,
        ).decode()
    except subprocess.CalledProcessError:
        return []
    paths = []
    for line in output.strip().splitlines():
        # Format: " <hash> <path> (<describe>)" or "-<hash> <path>"
        parts = line.strip().split()
        if len(parts) >= 2:
            paths.append(parts[1])
    return paths


def _symlink_gitignored(git_root: Path, dest: Path) -> None:
    """Symlink gitignored files/directories into the snapshot.

    Recursively walks the repo tree and symlinks any gitignored entries
    that are missing from the snapshot. This makes data files (datasets,
    large binaries, etc.) accessible to experiments running from the
    snapshot without copying them.

    When a gitignored entry is found and symlinked, its subtree is not
    explored further. Directories that exist in both the repo and the
    snapshot are recursed into to discover nested gitignored entries
    (e.g. from a .gitignore in a subdirectory).
    """
    _symlink_gitignored_walk(git_root, dest, git_root, dest)


def _symlink_gitignored_walk(
    repo_dir: Path, snapshot_dir: Path, git_root: Path, dest_root: Path
) -> None:
    """Walk repo_dir and symlink gitignored entries missing from snapshot_dir."""
    for entry in repo_dir.iterdir():
        name = entry.name
        # Skip git metadata and the snapshot destination itself
        if name == ".git" or dest_root.is_relative_to(entry.resolve()):
            continue
        target_in_snapshot = snapshot_dir / name
        if target_in_snapshot.exists() or target_in_snapshot.is_symlink():
            # Entry exists in snapshot — recurse into directories to find
            # deeper gitignored entries
            if entry.is_dir() and not entry.is_symlink():
                _symlink_gitignored_walk(entry, target_in_snapshot, git_root, dest_root)
            continue
        # Entry missing from snapshot — check if it's gitignored
        rel_path = str(entry.relative_to(git_root))
        # Append trailing slash for directories so git check-ignore
        # matches directory-only patterns (e.g. "dir/")
        if entry.is_dir():
            rel_path += "/"
        result = subprocess.run(
            ["git", "check-ignore", "-q", rel_path],
            cwd=git_root,
            capture_output=True,
        )
        if result.returncode == 0:
            target_in_snapshot.symlink_to(entry.resolve())


def checkout_snapshot(commit_hash: str, dest: Path) -> Path:
    """Check out a snapshot commit into a plain directory (no git metadata).

    Uses `git archive` to extract the committed tree into dest, avoiding
    worktrees and any impact on git history or refs. Submodule directories
    are replaced with symlinks to the original submodule locations.

    Args:
        commit_hash: Git commit hash to extract.
        dest: Directory to populate with the snapshot files.

    Returns:
        Path to the created directory.
    """
    git_root = _find_git_root()
    dest = dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    archive = subprocess.Popen(
        ["git", "archive", "--format=tar", commit_hash],
        cwd=git_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    subprocess.check_call(
        ["tar", "xf", "-"],
        cwd=dest,
        stdin=archive.stdout,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    archive.wait()
    if archive.returncode != 0:
        raise subprocess.CalledProcessError(archive.returncode, "git archive")

    # Replace submodule placeholders with symlinks to the real directories
    for sub_path in _find_submodule_paths(git_root):
        sub_in_snapshot = dest / sub_path
        sub_in_repo = git_root / sub_path
        if sub_in_repo.is_dir():
            # Remove the empty placeholder (dir or file) and symlink
            if sub_in_snapshot.is_dir():
                sub_in_snapshot.rmdir()
            elif sub_in_snapshot.exists():
                sub_in_snapshot.unlink()
            sub_in_snapshot.symlink_to(sub_in_repo.resolve())

    # Symlink gitignored files/directories so experiments can access data
    # files that aren't committed (e.g. datasets, large binaries).
    _symlink_gitignored(git_root, dest)

    return dest


def stash_and_snapshot(dest: Path) -> tuple[str, Path]:
    """Stash current repo state and extract a file snapshot into dest.

    Args:
        dest: Directory to populate with the snapshot files.

    Returns:
        Tuple of (commit_hash, snapshot_path).
    """
    commit_hash = stash()
    snapshot_path = checkout_snapshot(commit_hash, dest)
    return commit_hash, snapshot_path
