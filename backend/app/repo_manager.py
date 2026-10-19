# app/repo_manager.py
"""
Repository Manager for Codebase Intelligence.

Handles:
- Detecting whether a repository input is a public Git URL or local directory path
- Shallow cloning and updating public Git repositories (GitHub, GitLab, Bitbucket, etc.)
- Sanitizing and extracting deterministic repo identifiers
- Tracking repository metadata in MongoDB
- Cross-platform file cleanup and caching
"""
from __future__ import annotations

import datetime
import os
import re
import shutil
import stat
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import structlog
from pymongo.database import Database

log = structlog.get_logger()

# Default directory for cloned repositories
DEFAULT_STORAGE_DIR = os.environ.get(
    "REPOS_STORAGE_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "storage", "repos")),
)

GIT_URL_PATTERNS = [
    re.compile(r"^https?://", re.IGNORECASE),
    re.compile(r"^git@", re.IGNORECASE),
    re.compile(r"^ssh://", re.IGNORECASE),
    re.compile(r"^git://", re.IGNORECASE),
]


def is_git_url(source: str) -> bool:
    """Check if the given string represents a remote Git repository URL."""
    if not source or not isinstance(source, str):
        return False
    source = source.strip()
    if any(p.match(source) for p in GIT_URL_PATTERNS):
        return True
    if source.endswith(".git"):
        return True
    if "github.com/" in source or "gitlab.com/" in source or "bitbucket.org/" in source:
        return True
    return False


def sanitize_repo_id(path_or_url: str) -> str:
    """
    Derive a clean, deterministic repository ID from either a local path or a Git URL.
    Examples:
      - https://github.com/psf/requests.git -> psf_requests
      - https://github.com/tiangolo/fastapi -> tiangolo_fastapi
      - git@github.com:octocat/Hello-World.git -> octocat_hello_world
      - /home/user/projects/my-api -> my_api
    """
    s = path_or_url.strip().rstrip("/")
    if s.endswith(".git"):
        s = s[:-4]

    if is_git_url(s):
        # Handle SSH style: git@github.com:owner/repo
        if "@" in s and ":" in s and not s.startswith("http"):
            after_colon = s.split(":")[-1]
            parts = [p for p in after_colon.split("/") if p]
        else:
            parsed = urlparse(s)
            parts = [p for p in parsed.path.split("/") if p]

        if len(parts) >= 2:
            candidate = f"{parts[-2]}_{parts[-1]}"
        elif len(parts) == 1:
            candidate = parts[0]
        else:
            candidate = "remote_repo"
    else:
        candidate = Path(s).resolve().name

    clean = re.sub(r"[^a-zA-Z0-9_-]", "_", candidate).strip("_").lower()
    return clean or "default_repo"


def remove_dir_force(path: Path | str) -> None:
    """Recursively delete a directory, handling read-only git pack files on Windows/Linux."""
    p = Path(path)
    if not p.exists():
        return

    def _handle_readonly(func, file_path, exc_info):
        try:
            os.chmod(file_path, stat.S_IWRITE)
            func(file_path)
        except Exception:
            pass

    shutil.rmtree(p, onerror=_handle_readonly)


def clone_or_update_repo(
    repo_url: str,
    repo_id: str,
    branch: Optional[str] = None,
    storage_dir: Optional[str] = None,
    timeout_seconds: int = 120,
) -> Dict[str, Any]:
    """
    Clones a public Git repository using shallow cloning (--depth 1).
    If the repo already exists, pulls latest changes.

    Returns dict with:
      - repo_path: local path on disk
      - branch: active branch name
      - commit_hash: HEAD commit hash
      - is_new_clone: bool
    """
    target_root = Path(storage_dir or DEFAULT_STORAGE_DIR).resolve()
    target_root.mkdir(parents=True, exist_ok=True)
    target_dir = target_root / repo_id

    # Non-interactive git environment to prevent hanging on auth/passwords
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_ASKPASS"] = ""

    is_new = False

    # Check if existing valid clone
    if (target_dir / ".git").is_dir():
        log.info("repo_manager.updating_existing", repo_id=repo_id, path=str(target_dir))
        try:
            # Check if remote URL matches
            remote_cmd = subprocess.run(
                ["git", "remote", "get-url", "origin"],
                cwd=str(target_dir),
                capture_output=True,
                text=True,
                timeout=15,
                env=env,
            )
            current_remote = remote_cmd.stdout.strip() if remote_cmd.returncode == 0 else ""

            if current_remote and current_remote.rstrip("/").removesuffix(".git") == repo_url.rstrip("/").removesuffix(".git"):
                # Fetch and pull latest shallow commit
                fetch_cmd = ["git", "fetch", "--depth", "1"]
                if branch:
                    fetch_cmd.extend(["origin", branch])
                subprocess.run(
                    fetch_cmd,
                    cwd=str(target_dir),
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                    env=env,
                    check=True,
                )
                target_ref = f"origin/{branch}" if branch else "FETCH_HEAD"
                subprocess.run(
                    ["git", "reset", "--hard", target_ref],
                    cwd=str(target_dir),
                    capture_output=True,
                    text=True,
                    timeout=30,
                    env=env,
                    check=True,
                )
            else:
                # Different repo URL, re-clone
                log.info("repo_manager.remote_url_mismatch_recloning", repo_id=repo_id)
                remove_dir_force(target_dir)
                is_new = True
        except Exception as exc:
            log.warning("repo_manager.update_failed_recloning", error=str(exc), repo_id=repo_id)
            remove_dir_force(target_dir)
            is_new = True
    else:
        if target_dir.exists():
            remove_dir_force(target_dir)
        is_new = True

    if is_new or not (target_dir / ".git").is_dir():
        log.info("repo_manager.cloning_repo", repo_url=repo_url, repo_id=repo_id, target=str(target_dir))
        clone_cmd = ["git", "clone", "--depth", "1"]
        if branch:
            clone_cmd.extend(["--branch", branch])
        clone_cmd.extend([repo_url, str(target_dir)])

        try:
            result = subprocess.run(
                clone_cmd,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                env=env,
            )
            if result.returncode != 0:
                err_msg = (result.stderr or result.stdout or "").strip()
                log.error("repo_manager.clone_failed", error=err_msg, repo_url=repo_url)

                # Detect common git clone failure modes
                lower_err = err_msg.lower()
                if "could not read username" in lower_err or "authentication failed" in lower_err or "terminal prompts disabled" in lower_err:
                    raise ValueError(
                        f"Authentication required or repository not found for '{repo_url}'. "
                        "Currently, this service supports public repositories. "
                        "Please verify the repository URL is public and accessible without credentials."
                    )
                elif "not found" in lower_err or "repository" in lower_err and "does not exist" in lower_err:
                    raise ValueError(f"Repository not found at '{repo_url}'. Please check the URL and try again.")
                else:
                    raise RuntimeError(f"Git clone failed: {err_msg}")
        except subprocess.TimeoutExpired:
            remove_dir_force(target_dir)
            raise TimeoutError(f"Git clone timed out after {timeout_seconds}s for '{repo_url}'.")

    # Extract commit hash and active branch
    commit_hash = ""
    active_branch = branch or ""
    try:
        rev_cmd = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(target_dir),
            capture_output=True,
            text=True,
            timeout=10,
            env=env,
        )
        if rev_cmd.returncode == 0:
            commit_hash = rev_cmd.stdout.strip()

        if not active_branch:
            branch_cmd = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=str(target_dir),
                capture_output=True,
                text=True,
                timeout=10,
                env=env,
            )
            if branch_cmd.returncode == 0:
                active_branch = branch_cmd.stdout.strip()
    except Exception as exc:
        log.warning("repo_manager.inspect_git_meta_failed", error=str(exc))

    return {
        "repo_path": str(target_dir),
        "branch": active_branch or "default",
        "commit_hash": commit_hash,
        "is_new_clone": is_new,
    }


def resolve_repository_source(
    repo_url: Optional[str] = None,
    repo_path: Optional[str] = None,
    repo_id: Optional[str] = None,
    branch: Optional[str] = None,
) -> Tuple[str, str, bool, Optional[str], Dict[str, Any]]:
    """
    Unified resolver:
    Takes either a public Git URL or local filesystem path (or detects if repo_path is actually a URL),
    prepares the local directory, and returns:
      (resolved_local_path, final_repo_id, is_remote, normalized_url, clone_details)
    """
    raw_source = (repo_url or repo_path or "").strip()
    if not raw_source:
        raise ValueError("Either 'repo_url' or 'repo_path' must be provided.")

    if is_git_url(raw_source):
        final_repo_id = repo_id or sanitize_repo_id(raw_source)
        clone_details = clone_or_update_repo(
            repo_url=raw_source,
            repo_id=final_repo_id,
            branch=branch,
        )
        return (
            clone_details["repo_path"],
            final_repo_id,
            True,
            raw_source,
            clone_details,
        )
    else:
        local_p = Path(raw_source).resolve()
        if not local_p.exists() or not local_p.is_dir():
            raise ValueError(f"Local repository path '{raw_source}' does not exist or is not a directory.")

        final_repo_id = repo_id or sanitize_repo_id(str(local_p))
        return (
            str(local_p),
            final_repo_id,
            False,
            None,
            {"repo_path": str(local_p), "branch": "local", "commit_hash": ""},
        )


def record_repo_metadata(
    db: Database,
    repo_id: str,
    repo_path: str,
    repo_url: Optional[str] = None,
    is_remote: bool = False,
    symbol_count: int = 0,
    branch: Optional[str] = None,
    commit_hash: Optional[str] = None,
) -> None:
    """Store or update repository metadata in the 'repo_metadata' collection."""
    col = db["repo_metadata"]
    col.update_one(
        {"_id": repo_id},
        {
            "$set": {
                "repo_id": repo_id,
                "repo_path": repo_path,
                "repo_url": repo_url,
                "is_remote": is_remote,
                "symbol_count": symbol_count,
                "branch": branch,
                "commit_hash": commit_hash,
                "indexed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
        },
        upsert=True,
    )


def list_repositories_with_metadata(db: Database) -> List[Dict[str, Any]]:
    """
    List all repositories available in MongoDB, combining symbol collection counts
    with any metadata from repo_metadata.
    """
    collections = db.list_collection_names()
    meta_col = db["repo_metadata"]
    all_meta = {doc["_id"]: doc for doc in meta_col.find({})}

    repos = []
    for col in collections:
        if col.endswith("_symbols"):
            repo_id = col.removesuffix("_symbols")
            count = db[col].estimated_document_count()
            meta = all_meta.get(repo_id, {})
            repos.append({
                "repo_id": repo_id,
                "symbol_count": count,
                "repo_url": meta.get("repo_url"),
                "is_remote": meta.get("is_remote", False if not meta.get("repo_url") else True),
                "branch": meta.get("branch"),
                "commit_hash": meta.get("commit_hash"),
                "indexed_at": meta.get("indexed_at"),
            })

    # Sort repositories alphabetically by repo_id
    repos.sort(key=lambda r: r["repo_id"])
    return repos

