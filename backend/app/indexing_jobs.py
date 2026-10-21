"""Shared indexing job lifecycle for local and remote indexing APIs."""
from __future__ import annotations

import datetime
import logging
import uuid
from typing import Any, Dict

from pymongo import MongoClient
from pymongo.database import Database

from app.indexers.cpg_indexer import RepoGraphIndexer
from app.repo_manager import record_repo_metadata, resolve_repository_source


def create_index_job(
    db: Database,
    request_data: Dict[str, Any],
    owner_id: str,
) -> Dict[str, Any]:
    job_id = str(uuid.uuid4())
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    job = {
        "_id": job_id,
        "owner_id": owner_id,
        "status": "queued",
        "phase": "queued",
        "total_files": 0,
        "files_completed": 0,
        "parsed_files": 0,
        "current_file": None,
        "percent": 0,
        "symbols_indexed": 0,
        "embeddings_generated": 0,
        "request": request_data,
        "created_at": now,
        "updated_at": now,
    }
    db["indexing_jobs"].insert_one(job)
    return public_job(job)


def public_job(job: Dict[str, Any]) -> Dict[str, Any]:
    """Return job state without internal ownership or request fields."""
    return {
        key: value
        for key, value in job.items()
        if key not in {"_id", "owner_id", "request"}
    } | {"job_id": str(job.get("_id", job.get("job_id", "")))}


def run_index_job(job_id: str, request_data: Dict[str, Any], mongo_uri: str) -> None:
    client = MongoClient(mongo_uri)
    db = client["prode_engine"]
    jobs = db["indexing_jobs"]
    indexer = None

    def update_job(values: Dict[str, Any]) -> None:
        values["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        jobs.update_one({"_id": job_id}, {"$set": values})

    try:
        update_job({"status": "running", "phase": "preparing"})
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
            progress_callback=lambda progress: update_job(progress),
        )
        indexer.index_repository()

        collection_name = f"{actual_repo_id}_symbols"
        symbol_count = db[collection_name].estimated_document_count()
        job_progress = jobs.find_one({"_id": job_id}, {"embeddings_generated": 1}) or {}
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
                "status": "completed",
                "phase": "completed",
                "percent": 100,
                "current_file": None,
                "symbols_indexed": symbol_count,
                "result": {
                    "repo_id": actual_repo_id,
                    "source_type": "git_url" if is_remote else "local_path",
                    "repo_url": normalized_url,
                    "branch": clone_details.get("branch"),
                    "commit_hash": clone_details.get("commit_hash"),
                    "symbols_indexed": symbol_count,
                    "embeddings_generated": job_progress.get("embeddings_generated", 0),
                },
            }
        )
    except Exception as exc:
        logging.exception("Repository indexing job %s failed", job_id)
        update_job(
            {
                "status": "failed",
                "phase": "failed",
                "current_file": None,
                "error": str(exc),
            }
        )
    finally:
        if indexer is not None:
            indexer.client.close()
        client.close()
