# app/server.py
"""
FastAPI HTTP & Server-Sent Events (SSE) server for the Codebase Planning Agent.

Features:
- JWT Authentication (register / login)
- VIP user greetings (names stored in MongoDB, served via API)
- Rate limiting (4 msg/min for normal users, unlimited for VIPs)
- Real-time event streaming (SSE) for agent reasoning steps
- Human-in-the-Loop (HITL) pause/resume via thread checkpointing in MongoDB
- Session state inspection & history per user
- Repo listing and indexing endpoints
- Repo file-map for generalised architecture questions
- Per-session usage tracking (messages, tokens, cost)
- Production-ready for Docker and GCP Cloud Run
"""
from __future__ import annotations

import json
import os
import re
import sys
import urllib.parse
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Any, AsyncGenerator, Dict, Iterator, Optional

import structlog
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from pymongo import MongoClient
from starlette.types import ASGIApp, Receive, Scope, Send

# Ensure app/ is in sys.path
sys.path.insert(0, os.path.dirname(__file__))

from agent.graph import build_graph
from agent.state import AgentState, PlanStep
from agent.tools import AgentToolkit
from auth import (
    check_rate_limit,
    create_access_token,
    create_session,
    create_user,
    get_current_user,
    get_user_by_email,
    get_user_sessions,
    get_vip_users,
    is_vip_user,
    log_user_message,
    seed_vip_users,
    verify_password,
)
from graph_tools import CodeGraphTools
from indexers.cpg_indexer import RepoGraphIndexer
from llm_service.factory import LLMFactory
from memory.mongo_checkpointer import build_mongo_checkpointer
from repo_manager import (
    is_git_url,
    list_repositories_with_metadata,
    record_repo_metadata,
    resolve_repository_source,
    sanitize_repo_id,
)

from app.auth_mcp import MCPAuthManager, current_request_api_key
from main import mcp

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
log = structlog.get_logger()


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

class RegisterRequest(BaseModel):
    email: str = Field(..., description="User email address")
    password: str = Field(..., min_length=6, description="Password (min 6 chars)")
    name: str = Field(..., min_length=1, description="Display name")


class LoginRequest(BaseModel):
    email: str
    password: str


class ChatRequest(BaseModel):
    task: str = Field(..., description="User query or engineering task to plan")
    repo_id: str = Field(..., description="Target repository ID indexed in MongoDB")
    thread_id: Optional[str] = Field(None, description="Existing thread ID to resume session")
    interactive: bool = Field(True, description="Pause at human review gate before spec generation")


class FeedbackRequest(BaseModel):
    thread_id: str = Field(..., description="Conversation thread ID to resume")
    feedback: str = Field("approve", description="'approve', 'revise: <feedback>', or 'stop'")


class IndexRepoRequest(BaseModel):
    repo_url: Optional[str] = Field(None, description="Public Git repository URL")
    repo_path: Optional[str] = Field(None, description="Filesystem path of repository to index")
    repo_id: Optional[str] = Field(None, description="Identifier for repo (defaults to sanitized name)")
    branch: Optional[str] = Field(None, description="Optional branch or tag to clone")


# ---------------------------------------------------------------------------
# Lifespan / Dependency Setup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    mongo_uri = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
    log.info("server.startup", mongo_uri=mongo_uri[:35] + "...")

    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=10000)
    db = client["prode_engine"]
    graph_tools = CodeGraphTools(db)
    toolkit = AgentToolkit(graph_tools)
    llm_client = LLMFactory.get_client()
    checkpointer = build_mongo_checkpointer(mongo_uri, client=client)
    agent_app = build_graph(llm_client=llm_client, toolkit=toolkit, checkpointer=checkpointer)

    # Seed VIP users into DB (idempotent — won't overwrite if already present)
    vip_names_env = os.environ.get(
        "VIP_USERS",
        "Krishna Deshmukh,Adarsh Gaur,Prateek Atluri"
    )
    seed_vip_users(db, [n.strip() for n in vip_names_env.split(",") if n.strip()])

    app.state.mongo_client = client
    app.state.db = db
    app.state.graph_tools = graph_tools
    app.state.toolkit = toolkit
    app.state.llm_client = llm_client
    app.state.agent_app = agent_app
    app.state.mongo_uri = mongo_uri

    try:
        # Run MCP session manager for Streamable HTTP transport if available
        if hasattr(mcp, "session_manager") and mcp.session_manager is not None:
            async with mcp.session_manager.run():
                yield
        else:
            yield
    finally:
        log.info("server.shutdown")
        client.close()


app = FastAPI(
    title="Codebase Intelligence Agent Server",
    description="Agentic code planning with deterministic AST/CPG graphs and real-time SSE streaming.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# MCP Authentication Middleware (Preserves Bearer token for MCP Tools)
# ---------------------------------------------------------------------------
_sse_session_keys: Dict[str, str] = {}


class MCPApiKeyMiddleware:
    """
    Extracts Bearer token or API key from HTTP headers, query parameters,
    or SSE session mappings, and binds it to current_request_api_key ContextVar.
    """
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] == "http":
            token = None

            # 1. Check HTTP Headers (Authorization: Bearer <key> or X-API-Key)
            for raw_k, raw_v in scope.get("headers", []):
                k = raw_k.decode("latin1", errors="ignore").lower()
                if k == "authorization":
                    v = raw_v.decode("latin1", errors="ignore").strip()
                    if v.lower().startswith("bearer "):
                        token = v[7:].strip()
                    elif v:
                        token = v
                elif k == "x-api-key" and not token:
                    token = raw_v.decode("latin1", errors="ignore").strip()

            # 2. Check query params fallback (?api_key=... or ?token=...)
            query_string = scope.get("query_string", b"").decode("latin1", errors="ignore")
            session_id = None
            if query_string:
                try:
                    params = urllib.parse.parse_qs(query_string)
                    if not token:
                        for qk in ("api_key", "token", "key", "code_graph_key"):
                            if qk in params and params[qk]:
                                token = params[qk][0].strip()
                                break
                    session_id = params.get("session_id", [None])[0]
                except Exception:
                    pass

            # 3. Associate with or retrieve from SSE session
            if session_id:
                if token:
                    _sse_session_keys[session_id] = token
                elif session_id in _sse_session_keys:
                    token = _sse_session_keys[session_id]

            if token:
                token_reset = current_request_api_key.set(token)
                try:
                    await self.app(scope, receive, send)
                finally:
                    current_request_api_key.reset(token_reset)
                return

        await self.app(scope, receive, send)


app.add_middleware(MCPApiKeyMiddleware)

# ---------------------------------------------------------------------------
# MCP Server Transport Setup (Streamable HTTP + SSE)
# ---------------------------------------------------------------------------
try:
    from starlette.routing import Route
    from mcp.server.transport_security import TransportSecuritySettings

    # Allow remote MCP connections across IP / reverse-proxy without DNS rebinding rejection
    mcp_security = TransportSecuritySettings(enable_dns_rebinding_protection=False)
except Exception:
    mcp_security = None

mcp_streamable_app = None
mcp_sse_app = None

try:
    if hasattr(mcp, "streamable_http_app"):
        mcp_streamable_app = mcp.streamable_http_app(
            transport_security=mcp_security,
            streamable_http_path="/mcp",
        )
    if hasattr(mcp, "sse_app"):
        mcp_sse_app = mcp.sse_app(
            transport_security=mcp_security,
            message_path="/mcp/messages/",
        )
except Exception as exc:
    log.warning("server.mcp_init_failed", error=str(exc))

# 1. Mount Streamable HTTP (modern MCP standard used by Antigravity, Cursor, etc. via POST)
if mcp_streamable_app is not None and getattr(mcp_streamable_app, "routes", None):
    endpoint = mcp_streamable_app.routes[0].endpoint
    app.router.routes.append(Route("/mcp", endpoint=endpoint, methods=["GET", "POST", "DELETE"]))
    app.router.routes.append(Route("/mcp/", endpoint=endpoint, methods=["GET", "POST", "DELETE"]))
    # Also handle POST /mcp/sse so clients configured with /mcp/sse work seamlessly
    app.router.routes.append(Route("/mcp/sse", endpoint=endpoint, methods=["POST"]))

# 2. Mount SSE Transport (for legacy MCP clients expecting GET /sse or GET /mcp/sse)
if mcp_sse_app is not None:
    try:
        app.mount("/sse", mcp_sse_app)
        if len(mcp_sse_app.routes) > 0:
            sse_get_endpoint = mcp_sse_app.routes[0].endpoint
            app.router.routes.append(Route("/mcp/sse", endpoint=sse_get_endpoint, methods=["GET"]))
        if len(mcp_sse_app.routes) > 1 and hasattr(mcp_sse_app.routes[1], "app"):
            app.mount("/mcp/messages", mcp_sse_app.routes[1].app)
    except Exception as exc:
        log.warning("server.mcp_sse_mount_failed", error=str(exc))


# ---------------------------------------------------------------------------
# SSE Serialization Helpers
# ---------------------------------------------------------------------------

def sse_event(event: str, data: Any) -> str:
    """Format a Server-Sent Event."""
    payload = json.dumps(data, default=str)
    return f"event: {event}\ndata: {payload}\n\n"


def _format_plan(plan: list) -> list[dict]:
    res = []
    for s in plan:
        if hasattr(s, "__dataclass_fields__"):
            res.append(asdict(s))
        elif isinstance(s, dict):
            res.append(s)
        else:
            res.append({"description": str(s)})
    return res


def _serialize_chunk(node_name: str, state: dict) -> dict:
    """Extract clean, client-friendly details from a LangGraph node output."""
    details: dict[str, Any] = {"node": node_name}

    if node_name == "planner":
        raw_plan = state.get("plan", [])
        details["plan"] = _format_plan(raw_plan)
        details["iteration"] = state.get("iteration", 1)
        messages = state.get("messages", [])
        for msg in reversed(messages):
            txt = getattr(msg, "content", str(msg))
            if isinstance(txt, str) and "reasoning" in txt:
                try:
                    clean = re.sub(r"^```(?:json)?\s*", "", txt.strip(), flags=re.MULTILINE)
                    clean = re.sub(r"\s*```$", "", clean.strip(), flags=re.MULTILINE)
                    p = json.loads(clean)
                    if isinstance(p, dict) and "reasoning" in p:
                        details["reasoning"] = p["reasoning"]
                        break
                except Exception:
                    pass

    elif node_name == "tool_caller":
        obs = state.get("observations", [])
        if obs:
            last = obs[-1]
            details["tool"] = last.get("tool")
            details["args"] = last.get("args")
            details["error"] = last.get("error")
            res = last.get("result")
            if isinstance(res, (dict, list)):
                try:
                    res_str = json.dumps(res, indent=2, default=str)
                except Exception:
                    res_str = str(res)
            else:
                res_str = str(res or "")
            details["result_preview"] = res_str[:600] + ("..." if len(res_str) > 600 else "")
        details["current_step"] = state.get("current_step", 0)

    elif node_name == "reflector":
        details["reflection"] = state.get("reflection", "")

    elif node_name == "human_review":
        details["awaiting_human"] = state.get("awaiting_human", True)

    elif node_name == "spec_writer":
        details["final_spec"] = state.get("final_spec", "")

    return details


def _build_awaiting_human_payload(thread_id: str, state_values: dict) -> dict:
    """Build a clean, structured payload for human-in-the-loop review."""
    raw_plan = state_values.get("plan", [])
    plan = _format_plan(raw_plan)
    task = state_values.get("task", "")
    repo_id = state_values.get("repo_id", "")
    reflection = state_values.get("reflection", "")
    observations = state_values.get("observations", [])
    iteration = state_values.get("iteration", 1)

    reasoning = ""
    for msg in reversed(state_values.get("messages", [])):
        txt = getattr(msg, "content", str(msg))
        if isinstance(txt, str) and "reasoning" in txt:
            try:
                clean = re.sub(r"^```(?:json)?\s*", "", txt.strip(), flags=re.MULTILINE)
                clean = re.sub(r"\s*```$", "", clean.strip(), flags=re.MULTILINE)
                p = json.loads(clean)
                if isinstance(p, dict) and "reasoning" in p:
                    reasoning = p["reasoning"]
                    break
            except Exception:
                pass

    completed_steps = sum(1 for s in plan if s.get("status") == "done")
    total_steps = len(plan)

    clean_prompt = (
        f"The planning agent has finished inspecting the codebase AST graph ({completed_steps}/{total_steps} steps completed). "
        "Review the execution findings, self-critique, and architecture plan below before generating the final specification."
    )

    tools_used = []
    for obs in observations:
        t = obs.get("tool")
        if t and t not in tools_used:
            tools_used.append(t)

    return {
        "thread_id": thread_id,
        "prompt": clean_prompt,
        "task": task,
        "repo_id": repo_id,
        "reasoning": reasoning,
        "plan": plan,
        "reflection": reflection,
        "iteration": iteration,
        "stats": {
            "total_steps": total_steps,
            "completed_steps": completed_steps,
            "observations_count": len(observations),
            "tools_used": tools_used,
        },
        "instructions": "Call POST /api/agent/feedback with feedback='approve', 'revise: <feedback>', or 'stop'",
    }


# ---------------------------------------------------------------------------
# Auth Routes
# ---------------------------------------------------------------------------

@app.post("/api/auth/register")
def register(body: RegisterRequest, request: Request):
    """Register a new user account."""
    db = request.app.state.db
    user = create_user(db, email=body.email, password=body.password, name=body.name)
    vip = is_vip_user(db, body.name)
    token = create_access_token({
        "sub": body.email.lower(),
        "name": body.name.strip(),
        "is_vip": vip,
    })
    return {
        "token": token,
        "user": {
            "email": user["email"],
            "name": user["name"],
            "is_vip": vip,
        },
    }


@app.post("/api/auth/login")
def login(body: LoginRequest, request: Request):
    """Authenticate a user and return a JWT."""
    db = request.app.state.db
    user = get_user_by_email(db, body.email)
    if not user or not verify_password(body.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    vip = is_vip_user(db, user["name"])
    token = create_access_token({
        "sub": user["email"],
        "name": user["name"],
        "is_vip": vip,
    })
    return {
        "token": token,
        "user": {
            "email": user["email"],
            "name": user["name"],
            "is_vip": vip,
        },
    }


@app.get("/api/auth/vip-users")
def list_vip_users(request: Request):
    """Return VIP user names (for frontend greeting, names only — no PII)."""
    db = request.app.state.db
    names = get_vip_users(db)
    return {"vip_names": names}


@app.get("/api/auth/me")
def get_me(current_user: dict = Depends(get_current_user)):
    """Return current authenticated user's profile."""
    return {
        "email": current_user.get("sub"),
        "name": current_user.get("name"),
        "is_vip": current_user.get("is_vip", False),
    }


# ---------------------------------------------------------------------------
# Session Routes
# ---------------------------------------------------------------------------

@app.get("/api/sessions")
def list_sessions(request: Request, current_user: dict = Depends(get_current_user)):
    """List all sessions for the authenticated user."""
    db = request.app.state.db
    sessions = get_user_sessions(db, user_id=current_user["sub"])
    return {"sessions": sessions}


@app.delete("/api/sessions/{thread_id}")
def delete_session(thread_id: str, request: Request, current_user: dict = Depends(get_current_user)):
    """Delete (archive) a session for the current user."""
    db = request.app.state.db
    result = db["sessions"].update_one(
        {"thread_id": thread_id, "user_id": current_user["sub"]},
        {"$set": {"status": "deleted"}},
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"status": "deleted"}


# ---------------------------------------------------------------------------
# Plug-and-Play MCP API Key Management (Self-Service for Users)
# ---------------------------------------------------------------------------

class CreateKeyRequest(BaseModel):
    name: str = Field("My Agent", description="Client or agent name (e.g. Cursor, Slack Bot, Claude Desktop)")


@app.post("/api/keys/generate")
def generate_mcp_key(body: CreateKeyRequest, request: Request, current_user: dict = Depends(get_current_user)):
    """Generate a new plug-and-play MCP API key for the authenticated user."""
    db = request.app.state.db
    auth_mgr = MCPAuthManager(db)
    raw_key, info = auth_mgr.generate_api_key(
        name=body.name,
        tenant_id=current_user.get("sub", "default"),
    )
    base_url = str(request.base_url).rstrip("/")
    mcp_endpoint = f"{base_url}/mcp"
    mcp_sse_endpoint = f"{base_url}/mcp/sse"
    return {
        "api_key": raw_key,
        "key_info": info,
        "mcp_url": mcp_endpoint,
        "mcp_sse_url": mcp_sse_endpoint,
        "configs": {
            "antigravity": {
                "mcpServers": {
                    "CodeGraph": {
                        "url": mcp_endpoint,
                        "headers": {
                            "Authorization": f"Bearer {raw_key}"
                        }
                    }
                }
            },
            "cursor": {
                "mcpServers": {
                    "CodeGraph": {
                        "url": mcp_endpoint,
                        "headers": {
                            "Authorization": f"Bearer {raw_key}"
                        }
                    }
                }
            },
            "claude_desktop": {
                "mcpServers": {
                    "CodeGraph": {
                        "url": mcp_endpoint,
                        "headers": {
                            "Authorization": f"Bearer {raw_key}"
                        }
                    }
                }
            },
            "slack_bot_env": {
                "CODEGRAPH_MCP_URL": mcp_endpoint,
                "CODEGRAPH_API_KEY": raw_key
            }
        }
    }


@app.get("/api/keys")
def list_mcp_keys(request: Request, current_user: dict = Depends(get_current_user)):
    """List all MCP API keys for the current user."""
    db = request.app.state.db
    auth_mgr = MCPAuthManager(db)
    keys = auth_mgr.list_api_keys(tenant_id=current_user.get("sub", "default"))
    return {"keys": keys}


@app.delete("/api/keys/{key_id}")
def revoke_mcp_key(key_id: str, request: Request, current_user: dict = Depends(get_current_user)):
    """Revoke an MCP API key."""
    db = request.app.state.db
    auth_mgr = MCPAuthManager(db)
    success = auth_mgr.revoke_api_key(key_id)
    if not success:
        raise HTTPException(status_code=404, detail="Key not found or already revoked.")
    return {"status": "revoked", "key_id": key_id}



# ---------------------------------------------------------------------------
# Static Routes & Health
# ---------------------------------------------------------------------------

@app.get("/login")
@app.get("/signup")
@app.get("/")
def serve_ui():
    """Serves the Web UI Cockpit."""
    index_file = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return JSONResponse({"message": "Codebase Intelligence Agent Server is running. Visit /docs for Swagger."})


@app.get("/health")
@app.get("/api/health")
def health(request: Request):
    """Health check endpoint for GCP Cloud Run and load balancers."""
    try:
        request.app.state.mongo_client.admin.command("ping")
        return {"status": "healthy", "mongodb": "connected"}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database unreachable: {exc}")


# ---------------------------------------------------------------------------
# Repository Routes
# ---------------------------------------------------------------------------

@app.get("/api/repos")
def list_repositories(request: Request, current_user: dict = Depends(get_current_user)):
    """List all indexed repositories currently available in MongoDB with metadata."""
    db = request.app.state.db
    repos = list_repositories_with_metadata(db)
    return {"repositories": repos}


@app.post("/api/repos/index")
def index_repository(body: IndexRepoRequest, request: Request, current_user: dict = Depends(get_current_user)):
    """
    Index a codebase into the AST graph database from a public Git repository URL
    or local directory path.
    """
    mongo_uri = request.app.state.mongo_uri
    db = request.app.state.db

    try:
        (
            resolved_local_path,
            actual_repo_id,
            is_remote,
            normalized_url,
            clone_details,
        ) = resolve_repository_source(
            repo_url=body.repo_url,
            repo_path=body.repo_path,
            repo_id=body.repo_id,
            branch=body.branch,
        )

        indexer = RepoGraphIndexer(
            repo_path=resolved_local_path,
            repo_id=actual_repo_id,
            mongo_uri=mongo_uri,
        )
        indexer.index_repository()

        count = db[f"{actual_repo_id}_symbols"].estimated_document_count()

        record_repo_metadata(
            db=db,
            repo_id=actual_repo_id,
            repo_path=resolved_local_path,
            repo_url=normalized_url,
            is_remote=is_remote,
            symbol_count=count,
            branch=clone_details.get("branch"),
            commit_hash=clone_details.get("commit_hash"),
        )

        source_desc = f"public repository '{normalized_url}'" if is_remote else f"local path '{resolved_local_path}'"

        return {
            "status": "success",
            "repo_id": actual_repo_id,
            "source_type": "git_url" if is_remote else "local_path",
            "repo_url": normalized_url,
            "branch": clone_details.get("branch"),
            "commit_hash": clone_details.get("commit_hash"),
            "symbols_indexed": count,
            "message": f"Successfully indexed {source_desc} under '{actual_repo_id}'.",
        }
    except ValueError as val_err:
        raise HTTPException(status_code=400, detail=str(val_err))
    except Exception as exc:
        log.error("server.index_failed", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/repos/{repo_id}/filemap")
def get_repo_filemap(repo_id: str, request: Request, current_user: dict = Depends(get_current_user)):
    """
    Returns a structured map of all files and their top-level symbols for a repo.
    Used to answer generalised architecture / structure questions without assuming
    the presence of README files or docs.
    """
    db = request.app.state.db
    col_name = f"{repo_id}_symbols"
    if col_name not in db.list_collection_names():
        raise HTTPException(status_code=404, detail=f"Repository '{repo_id}' not indexed.")

    col = db[col_name]
    # Aggregate files → symbols (lightweight, no source_code)
    cursor = col.find(
        {},
        {"_id": 1, "name": 1, "type": 1, "file_path": 1, "signature": 1, "start_line": 1, "end_line": 1},
    )

    file_map: dict[str, list] = {}
    for doc in cursor:
        fp = doc.get("file_path", "unknown")
        if fp not in file_map:
            file_map[fp] = []
        file_map[fp].append({
            "id": doc["_id"],
            "name": doc.get("name"),
            "type": doc.get("type"),
            "signature": doc.get("signature"),
            "lines": f"{doc.get('start_line')}-{doc.get('end_line')}",
        })

    # Sort each file's symbols by start_line
    for fp in file_map:
        file_map[fp].sort(key=lambda s: (s.get("lines") or "0").split("-")[0].zfill(6))

    return {
        "repo_id": repo_id,
        "total_files": len(file_map),
        "total_symbols": sum(len(v) for v in file_map.values()),
        "file_map": file_map,
    }


# ---------------------------------------------------------------------------
# Agent Routes
# ---------------------------------------------------------------------------

@app.get("/api/agent/threads/{thread_id}")
def get_thread_state(thread_id: str, request: Request, current_user: dict = Depends(get_current_user)):
    """Inspect current state and history for a conversation thread."""
    agent_app = request.app.state.agent_app
    thread_config = {"configurable": {"thread_id": thread_id}}
    state = agent_app.get_state(thread_config)

    if not state or not state.values:
        raise HTTPException(status_code=404, detail=f"Thread '{thread_id}' not found.")

    vals = state.values
    review_info = None
    if vals.get("awaiting_human", False) or (state.next and "human_review" in state.next):
        review_info = _build_awaiting_human_payload(thread_id, vals)

    return {
        "thread_id": thread_id,
        "task": vals.get("task"),
        "repo_id": vals.get("repo_id"),
        "next_nodes": list(state.next) if state.next else [],
        "awaiting_human": vals.get("awaiting_human", False) or bool(state.next and "human_review" in state.next),
        "plan": _format_plan(vals.get("plan", [])),
        "current_step": vals.get("current_step", 0),
        "reflection": vals.get("reflection", ""),
        "final_spec": vals.get("final_spec", ""),
        "error": vals.get("error", ""),
        "review_info": review_info,
    }


@app.post("/api/agent/stream")
def stream_agent_task(body: ChatRequest, request: Request, current_user: dict = Depends(get_current_user)):
    """
    Start or continue an agent task, streaming reasoning events via SSE.
    Enforces per-user rate limiting. VIP users are exempt.
    """
    db = request.app.state.db
    agent_app = request.app.state.agent_app

    user_id = current_user["sub"]
    user_name = current_user.get("name", "")
    vip = current_user.get("is_vip", False) or is_vip_user(db, user_name)

    # Rate limit check
    rl = check_rate_limit(user_id, is_vip=vip)
    if not rl["allowed"]:
        raise HTTPException(
            status_code=429,
            detail={
                "message": f"Rate limit reached. You can send {RATE_LIMIT_MAX_MESSAGES} messages per minute. Please wait {rl['reset_in']}s.",
                "reset_in": rl["reset_in"],
                "limit": rl["limit"],
            },
        )

    thread_id = body.thread_id or str(uuid.uuid4())
    thread_config = {"configurable": {"thread_id": thread_id}}
    initial_state = AgentState.initial(task=body.task, repo_id=body.repo_id)

    # Create/update session record
    create_session(db, user_id=user_id, user_name=user_name, repo_id=body.repo_id, thread_id=thread_id)
    log_user_message(db, thread_id=thread_id, user_id=user_id, task=body.task)

    def event_generator() -> Iterator[str]:
        yield sse_event("session_start", {
            "thread_id": thread_id,
            "task": body.task,
            "repo_id": body.repo_id,
            "interactive": body.interactive,
            "rate_limit": rl,
        })

        try:
            for chunk in agent_app.stream(initial_state, thread_config, stream_mode="updates"):
                node_name = list(chunk.keys())[0]
                node_state = chunk[node_name]
                details = _serialize_chunk(node_name, node_state)
                yield sse_event("node_update", details)

            current = agent_app.get_state(thread_config)
            if current.next and "human_review" in current.next:
                if body.interactive:
                    payload = _build_awaiting_human_payload(thread_id, current.values)
                    yield sse_event("awaiting_human", payload)
                    return
                else:
                    agent_app.update_state(
                        thread_config,
                        {"human_feedback": "approve", "awaiting_human": False},
                        as_node="human_review",
                    )
                    for chunk in agent_app.stream(None, thread_config, stream_mode="updates"):
                        node_name = list(chunk.keys())[0]
                        details = _serialize_chunk(node_name, chunk[node_name])
                        yield sse_event("node_update", details)

            final = agent_app.get_state(thread_config)
            spec = final.values.get("final_spec", "")
            yield sse_event("completed", {
                "thread_id": thread_id,
                "final_spec": spec,
                "status": "completed",
            })

        except Exception as exc:
            log.error("agent_stream.failed", thread_id=thread_id, error=str(exc))
            yield sse_event("error", {"error": str(exc), "thread_id": thread_id})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/agent/feedback")
def submit_feedback_and_resume(body: FeedbackRequest, request: Request, current_user: dict = Depends(get_current_user)):
    """
    Resume an interrupted execution with human feedback ('approve', 'revise: ...', 'stop').
    """
    agent_app = request.app.state.agent_app
    thread_config = {"configurable": {"thread_id": body.thread_id}}

    current = agent_app.get_state(thread_config)
    if not current or not current.values:
        raise HTTPException(status_code=404, detail=f"Thread '{body.thread_id}' not found.")

    def event_generator() -> Iterator[str]:
        yield sse_event("feedback_received", {
            "thread_id": body.thread_id,
            "feedback": body.feedback,
        })

        feedback_norm = body.feedback.strip()
        if feedback_norm.lower() == "stop":
            agent_app.update_state(
                thread_config,
                {"human_feedback": "stop", "awaiting_human": False},
                as_node="human_review",
            )
            yield sse_event("completed", {
                "thread_id": body.thread_id,
                "final_spec": "",
                "status": "stopped",
            })
            return

        try:
            agent_app.update_state(
                thread_config,
                {"human_feedback": feedback_norm, "awaiting_human": False},
                as_node="human_review",
            )

            for chunk in agent_app.stream(None, thread_config, stream_mode="updates"):
                node_name = list(chunk.keys())[0]
                node_state = chunk[node_name]
                details = _serialize_chunk(node_name, node_state)
                yield sse_event("node_update", details)

            state_now = agent_app.get_state(thread_config)
            if state_now.next and "human_review" in state_now.next:
                payload = _build_awaiting_human_payload(body.thread_id, state_now.values)
                yield sse_event("awaiting_human", payload)
                return

            spec = state_now.values.get("final_spec", "")
            yield sse_event("completed", {
                "thread_id": body.thread_id,
                "final_spec": spec,
                "status": "completed",
            })

        except Exception as exc:
            log.error("agent_feedback_resume.failed", thread_id=body.thread_id, error=str(exc))
            yield sse_event("error", {"error": str(exc), "thread_id": body.thread_id})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Import RATE_LIMIT_MAX_MESSAGES for use inside stream handler
# ---------------------------------------------------------------------------
from auth import RATE_LIMIT_MAX_MESSAGES  # noqa: E402 (re-import for closure access)


# ---------------------------------------------------------------------------
# Direct Execution Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.server:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        proxy_headers=True,
        forwarded_allow_ips="*"
    )