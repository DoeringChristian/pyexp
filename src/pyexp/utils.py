"""Utility functions for pyexp."""

import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

# Default file suffixes to include in code packages (like Metaflow)
DEFAULT_PACKAGE_SUFFIXES = {".py"}

# Directories to always exclude from packaging
EXCLUDE_DIRS = {
    "__pycache__",
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    ".env",
    "env",
    ".tox",
    ".nox",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".hypothesis",
    "node_modules",
    ".eggs",
    "*.egg-info",
    "dist",
    "build",
    ".ipynb_checkpoints",
    ".coverage",
    "htmlcov",
    "__pypackages__",
}


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

    return dest


def _should_exclude_dir(name: str) -> bool:
    """Check if a directory name should be excluded from packaging."""
    if name.startswith("."):
        return True
    if name in EXCLUDE_DIRS:
        return True
    if name.endswith(".egg-info"):
        return True
    return False


def _collect_files_by_suffix(
    root: Path,
    suffixes: set[str] | None = None,
) -> list[tuple[Path, Path]]:
    """Collect files matching the given suffixes from a directory tree.

    Args:
        root: Root directory to walk.
        suffixes: Set of file suffixes to include (e.g., {".py"}).
                  If None, uses DEFAULT_PACKAGE_SUFFIXES.

    Returns:
        List of (source_path, relative_path) tuples.
    """
    if suffixes is None:
        suffixes = DEFAULT_PACKAGE_SUFFIXES

    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Filter out excluded directories in-place to prevent descending
        dirnames[:] = [d for d in dirnames if not _should_exclude_dir(d)]

        rel_dir = Path(dirpath).relative_to(root)

        for filename in filenames:
            # Skip hidden files
            if filename.startswith("."):
                continue

            # Check suffix
            if any(filename.endswith(suffix) for suffix in suffixes):
                src = Path(dirpath) / filename
                rel = rel_dir / filename
                files.append((src, rel))

    return files


def content_hash(
    root: Path | None = None,
    suffixes: set[str] | None = None,
) -> str:
    """Compute a deterministic SHA-256 hash of all collected source files.

    Hashes each file's relative path and contents so that any change in the
    file tree produces a different digest.  Files are sorted by relative path
    for determinism.

    Args:
        root: Root directory to hash from. Defaults to cwd.
        suffixes: Set of file suffixes to include (e.g., {".py"}).
                  If None, uses DEFAULT_PACKAGE_SUFFIXES.

    Returns:
        Hex digest string.
    """
    if root is None:
        root = Path.cwd()

    files = _collect_files_by_suffix(root, suffixes)
    files.sort(key=lambda pair: pair[1])

    hasher = hashlib.sha256()
    for src, rel in files:
        hasher.update(str(rel).encode())
        hasher.update(src.read_bytes())

    return hasher.hexdigest()


def package_files(
    dest: Path,
    root: Path | None = None,
    suffixes: set[str] | None = None,
) -> Path:
    """Package files with specific suffixes into a destination directory.

    This follows the Metaflow approach of only including relevant source files
    rather than copying the entire repository.

    Args:
        dest: Destination directory to copy files into.
        root: Root directory to package from. Defaults to git root.
        suffixes: Set of file suffixes to include (e.g., {".py"}).
                  If None, uses DEFAULT_PACKAGE_SUFFIXES.

    Returns:
        Path to the destination directory.
    """
    if root is None:
        root = _find_git_root()

    dest = dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)

    files = _collect_files_by_suffix(root, suffixes)

    for src, rel in files:
        dst = dest / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    return dest


def stash_and_snapshot(
    dest: Path,
    suffixes: set[str] | None = None,
) -> tuple[str, Path]:
    """Stash current repo state and create a packaged snapshot.

    Creates a git commit hash for reference/comparison, then packages only
    the relevant source files (by suffix) into the destination directory.

    Args:
        dest: Directory to populate with the snapshot files.
        suffixes: Set of file suffixes to include. Defaults to {".py"}.

    Returns:
        Tuple of (commit_hash, snapshot_path).
    """
    # Create git commit for reference (not exported, just for tracking)
    commit_hash = stash()

    # Package only relevant files by suffix
    snapshot_path = package_files(dest, suffixes=suffixes)

    return commit_hash, snapshot_path
