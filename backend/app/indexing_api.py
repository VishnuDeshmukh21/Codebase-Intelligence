"""Standalone HTTP service for repository indexing and embedding generation."""
from __future__ import annotations

import hmac
import os
from contextlib import asynccontextmanager
from typing import Optional

import structlog
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field
from pymongo import MongoClient

from app.indexing_jobs import (
    create_index_job,
    public_job,
    run_embedding_job,
    run_index_job,
)
from app.repo_manager import is_git_url

log = structlog.get_logger()


class IndexJobRequest(BaseModel):
    repo_url: str = Field(..., description="Public Git repository URL")
    repo_id: Optional[str] = None
    branch: Optional[str] = None
    owner_id: str = Field(..., description="Main application user ID")


@asynccontextmanager
async def lifespan(app: FastAPI):
    mongo_uri = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=10000)
    app.state.mongo_client = client
    app.state.db = client["prode_engine"]
    try:
        yield
    finally:
        client.close()


app = FastAPI(
    title="Codebase Intelligence Indexing Service",
    description="Asynchronous repository parsing and semantic embedding generation.",
    version="1.0.0",
    lifespan=lifespan,
)


def verify_service_key(
    x_indexing_service_key: Optional[str] = Header(default=None),
) -> None:
    expected_key = os.environ.get("INDEXING_SERVICE_API_KEY")
    if not expected_key:
        raise HTTPException(
            status_code=503,
            detail="INDEXING_SERVICE_API_KEY must be configured for the indexing service.",
        )
    if not x_indexing_service_key or not hmac.compare_digest(x_indexing_service_key, expected_key):
        raise HTTPException(status_code=401, detail="Invalid indexing service key.")


@app.get("/health")
def health(request: Request):
    try:
        request.app.state.mongo_client.admin.command("ping")
        return {"status": "healthy", "mongodb": "connected"}
    except Exception as exc:
        log.error("indexing_service.health_failed", error=str(exc))
        raise HTTPException(status_code=503, detail="Database unavailable.") from exc


@app.post("/api/index/jobs", status_code=202, dependencies=[Depends(verify_service_key)])
def start_index_job(
    body: IndexJobRequest,
    request: Request,
    background_tasks: BackgroundTasks,
):
    if not is_git_url(body.repo_url):
        raise HTTPException(status_code=400, detail="The indexing service accepts public Git URLs only.")

    request_data = body.dict(exclude={"owner_id"})
    job = create_index_job(request.app.state.db, request_data, body.owner_id)
    background_tasks.add_task(
        run_index_job,
        job["job_id"],
        request_data,
        os.environ.get("MONGODB_URI", "mongodb://localhost:27017"),
    )
    return job


@app.get(
    "/api/index/jobs/{job_id}",
    dependencies=[Depends(verify_service_key)],
)
def get_index_job(job_id: str, request: Request):
    job = request.app.state.db["indexing_jobs"].find_one({"_id": job_id})
    if not job:
        raise HTTPException(status_code=404, detail="Indexing job not found.")
    return public_job(job)


@app.post(
    "/api/index/jobs/{job_id}/stop",
    dependencies=[Depends(verify_service_key)],
)
def stop_index_job(job_id: str, request: Request):
    jobs = request.app.state.db["indexing_jobs"]
    result = jobs.update_one(
        {"_id": job_id, "status": "running", "phase": "embedding"},
        {"$set": {"stop_requested": True}},
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=409, detail="No running embedding stage to stop.")
    return public_job(jobs.find_one({"_id": job_id}))


@app.post(
    "/api/index/jobs/{job_id}/embeddings",
    status_code=202,
    dependencies=[Depends(verify_service_key)],
)
def resume_index_embeddings(
    job_id: str,
    request: Request,
    background_tasks: BackgroundTasks,
):
    jobs = request.app.state.db["indexing_jobs"]
    result = jobs.update_one(
        {
            "_id": job_id,
            "status": {"$in": ["paused", "failed"]},
            "parse_status": "completed",
        },
        {
            "$set": {
                "status": "queued",
                "phase": "embedding_queued",
                "embedding_status": "queued",
                "stop_requested": False,
            },
            "$unset": {"error": ""},
        },
    )
    if result.matched_count == 0:
        raise HTTPException(
            status_code=409,
            detail="Only a paused or failed embedding stage can be resumed.",
        )
    background_tasks.add_task(
        run_embedding_job,
        job_id,
        os.environ.get("MONGODB_URI", "mongodb://localhost:27017"),
    )
    return public_job(jobs.find_one({"_id": job_id}))
