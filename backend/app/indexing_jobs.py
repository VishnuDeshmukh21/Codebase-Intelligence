"""Shared parsing and resumable embedding job lifecycle."""
from __future__ import annotations

import datetime
import logging
import uuid
from typing import Any, Dict, Optional

from pymongo import MongoClient
from pymongo.database import Database

from app.indexers.cpg_indexer import RepoGraphIndexer
from app.repo_manager import record_repo_metadata, resolve_repository_source


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def create_index_job(
    db: Database,
    request_data: Dict[str, Any],
    owner_id: str,
) -> Dict[str, Any]:
    job_id = str(uuid.uuid4())
    job = {
        "_id": job_id,
        "owner_id": owner_id,
        "status": "queued",
        "phase": "queued",
        "parse_status": "queued",
        "embedding_status": "pending",
        "stop_requested": False,
        "total_files": 0,
        "files_completed": 0,
        "parsed_files": 0,
        "current_file": None,
        "percent": 0,
        "symbols_indexed": 0,
        "embeddings_generated": 0,
        "embedding_symbols_completed": 0,
        "embedding_symbols_total": 0,
        "embedding_files_completed": 0,
        "embedding_files_total": 0,
        "request": request_data,
        "created_at": _now(),
        "updated_at": _now(),
    }
    db["indexing_jobs"].insert_one(job)
    return public_job(job)


def public_job(job: Dict[str, Any]) -> Dict[str, Any]:
    """Return job state without internal ownership or request fields."""
    return {
        key: value
        for key, value in job.items()
        if key not in {"_id", "owner_id", "request", "stop_requested"}
    } | {"job_id": str(job.get("_id", job.get("job_id", "")))}


def _update_job(jobs, job_id: str, values: Dict[str, Any]) -> None:
    jobs.update_one(
        {"_id": job_id},
        {"$set": {**values, "updated_at": _now()}},
    )


def _run_embedding_stage(
    job_id: str,
    repo_id: str,
    jobs,
    indexer: RepoGraphIndexer,
) -> None:
    def update_job(values: Dict[str, Any]) -> None:
        _update_job(jobs, job_id, values)

    update_job(
        {
            "status": "running",
            "phase": "embedding",
            "embedding_status": "running",
            "current_file": None,
        }
    )
    try:
        result = indexer.generate_missing_embeddings(
            should_stop=lambda: bool(
                (jobs.find_one({"_id": job_id}, {"stop_requested": 1}) or {}).get(
                    "stop_requested"
                )
            )
        )
        if result["stopped"]:
            update_job(
                {
                    "status": "paused",
                    "phase": "embedding_paused",
                    "embedding_status": "paused",
                    "stop_requested": False,
                    "percent": 20
                    + int(
                        75
                        * result["embedding_symbols_completed"]
                        / max(result["embedding_symbols_total"], 1)
                    ),
                    "embeddings_generated": result["embedding_symbols_completed"],
                    "embedding_symbols_completed": result["embedding_symbols_completed"],
                    "embedding_symbols_total": result["embedding_symbols_total"],
                    "embedding_files_completed": result["embedding_files_completed"],
                    "embedding_files_total": result["embedding_files_total"],
                    "current_file": None,
                }
            )
            return

        existing_result = (
            (jobs.find_one({"_id": job_id}, {"result": 1}) or {}).get("result") or {}
        )
        _update_job(
            jobs,
            job_id,
            {
                "status": "completed",
                "phase": "completed",
                "embedding_status": "completed",
                "percent": 100,
                "embeddings_generated": result["embedding_symbols_completed"],
                "embedding_symbols_completed": result["embedding_symbols_completed"],
                "embedding_symbols_total": result["embedding_symbols_total"],
                "embedding_files_completed": result["embedding_files_completed"],
                "embedding_files_total": result["embedding_files_total"],
                "current_file": None,
                "result": {
                    **existing_result,
                    "repo_id": repo_id,
                    "symbols_indexed": jobs.database[f"{repo_id}_symbols"].estimated_document_count(),
                    "embeddings_generated": result["embedding_symbols_completed"],
                },
            },
        )
    except Exception as exc:
        logging.exception("Embedding stage for indexing job %s failed", job_id)
        update_job(
            {
                "status": "failed",
                "phase": "failed",
                "embedding_status": "failed",
                "current_file": None,
                "error": str(exc),
            }
        )


def run_index_job(job_id: str, request_data: Dict[str, Any], mongo_uri: str) -> None:
    """Parse a repository, persist symbols, then run embeddings as a separate stage."""
    client = MongoClient(mongo_uri)
    db = client["prode_engine"]
    jobs = db["indexing_jobs"]
    indexer: Optional[RepoGraphIndexer] = None

    def update_job(values: Dict[str, Any]) -> None:
        _update_job(jobs, job_id, values)

    try:
        update_job(
            {
                "status": "running",
                "phase": "preparing",
                "parse_status": "running",
            }
        )
        (
            resolved_local_path,
            actual_repo_id,
            is_remote,
            normalized_url,
            clone_details,
        ) = resolve_repository_source(
            repo_url=request_data.get("repo_url"),
            repo_path=request_data.get("repo_path"),
            repo_id=request_data.get("repo_id"),
            branch=request_data.get("branch"),
        )

        indexer = RepoGraphIndexer(
            repo_path=resolved_local_path,
            repo_id=actual_repo_id,
            mongo_uri=mongo_uri,
            progress_callback=update_job,
        )
        update_job({"repo_id": actual_repo_id})
        indexer.index_repository()

        symbol_count = db[f"{actual_repo_id}_symbols"].estimated_document_count()
        record_repo_metadata(
            db=db,
            repo_id=actual_repo_id,
            repo_path=resolved_local_path,
            repo_url=normalized_url,
            is_remote=is_remote,
            symbol_count=symbol_count,
            branch=clone_details.get("branch"),
            commit_hash=clone_details.get("commit_hash"),
        )
        update_job(
            {
                "parse_status": "completed",
                "embedding_status": "queued",
                "phase": "embedding",
                "symbols_indexed": symbol_count,
                "percent": 20,
                "result": {
                    "repo_id": actual_repo_id,
                    "source_type": "git_url" if is_remote else "local_path",
                    "repo_url": normalized_url,
                    "branch": clone_details.get("branch"),
                    "commit_hash": clone_details.get("commit_hash"),
                    "symbols_indexed": symbol_count,
                    "embeddings_generated": 0,
                },
            }
        )
        _run_embedding_stage(job_id, actual_repo_id, jobs, indexer)
        final_job = jobs.find_one({"_id": job_id}) or {}
        if final_job.get("status") == "completed":
            result = final_job.get("result", {})
            _update_job(
                jobs,
                job_id,
                {
                    "result": {
                        **result,
                        "source_type": "git_url" if is_remote else "local_path",
                        "repo_url": normalized_url,
                        "branch": clone_details.get("branch"),
                        "commit_hash": clone_details.get("commit_hash"),
                    }
                },
            )
    except Exception as exc:
        logging.exception("Repository parsing job %s failed", job_id)
        update_job(
            {
                "status": "failed",
                "phase": "failed",
                "parse_status": "failed",
                "current_file": None,
                "error": str(exc),
            }
        )
    finally:
        if indexer is not None:
            indexer.client.close()
        client.close()


def run_embedding_job(job_id: str, mongo_uri: str) -> None:
    """Resume a paused/failed embedding stage; persisted vectors are skipped."""
    client = MongoClient(mongo_uri)
    db = client["prode_engine"]
    jobs = db["indexing_jobs"]
    indexer: Optional[RepoGraphIndexer] = None
    try:
        job = jobs.find_one({"_id": job_id}) or {}
        repo_id = job.get("repo_id")
        if not repo_id:
            raise ValueError(f"Indexing job {job_id} has no parsed repository.")
        indexer = RepoGraphIndexer(
            repo_path=".",
            repo_id=repo_id,
            mongo_uri=mongo_uri,
            progress_callback=lambda progress: _update_job(jobs, job_id, progress),
        )
        _run_embedding_stage(job_id, repo_id, jobs, indexer)
    except Exception as exc:
        logging.exception("Could not resume embedding job %s", job_id)
        _update_job(
            jobs,
            job_id,
            {
                "status": "failed",
                "phase": "failed",
                "embedding_status": "failed",
                "current_file": None,
                "error": str(exc),
            },
        )
    finally:
        if indexer is not None:
            indexer.client.close()
        client.close()
