# app/main.py
"""
Codebase Intelligence: Plug-and-Play Model Context Protocol (MCP) Server.

Architecture & Philosophy:
--------------------------
1. Zero LLM Dependencies / Model-Agnostic:
   - Does NOT require GEMINI_API_KEY, OpenAI, or any server-side LLM keys.
   - All tools perform 100% deterministic AST parsing and MongoDB Graph ($graphLookup) traversals.
   - The HOST application/agent (Slack bot, Cursor, Claude Desktop, Antigravity, etc.)
     uses its own model to reason, plan, and orchestrate the tools.

2. Plug-and-Play Authentication:
   - Authenticates external agents via CODEGRAPH_API_KEY (env, CLI argument, or header).
   - Keys are managed with zero UI using `python -m app.auth_cli create-key`.

3. Transports:
   - stdio (default): for IDEs (Claude Desktop, Cursor, Cline) and local agent processes.
   - sse / streamable-http: for remote network integrations (e.g. cloud Slack bot).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

# Ensure app directory is discoverable
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv

load_dotenv()
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

# Import FastMCP or MCPServer according to mcp package version
try:
    from mcp.server.mcpserver import MCPServer
except ImportError:
    try:
        from mcp.server.fastmcp import FastMCP as MCPServer
    except Exception:
        class MCPServer:
            def __init__(self, *args, **kwargs): pass
            def tool(self): return lambda fn: fn
            def prompt(self): return lambda fn: fn
            def resource(self, *args, **kwargs): return lambda fn: fn
            def run(self, *args, **kwargs): print("MCP server library not available.")

from pymongo import MongoClient

from app.auth_mcp import MCPAuthManager
from app.graph_tools import CodeGraphTools
from app.indexers.cpg_indexer import RepoGraphIndexer
from app.repo_manager import (
    list_repositories_with_metadata,
    record_repo_metadata,
    resolve_repository_source,
    sanitize_repo_id,
)

# ---------------------------------------------------------------------------
# Server Initialization
# ---------------------------------------------------------------------------
mcp = MCPServer(
    name="CodeGraph-MCP",
    description="Deterministic AST & Code Property Graph server for AI coding agents",
)

# MongoDB Connection (Resilient: does not crash on startup if Mongo is temporarily offline)
db_uri = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
db = None
graph_tools = None
auth_manager = None

try:
    mongo_client = MongoClient(db_uri, serverSelectionTimeoutMS=3000)
    mongo_client.admin.command("ping")
    db = mongo_client["prode_engine"]
    graph_tools = CodeGraphTools(db)
    auth_manager = MCPAuthManager(db)
except Exception as exc:
    # Graceful fallback: tools will return clear error message on invocation rather than crashing stdio
    graph_tools = CodeGraphTools(None)
    auth_manager = MCPAuthManager(None)


def _check_auth() -> Optional[str]:
    """Validates the client's API key. Returns an error message if unauthorized, or None if valid."""
    is_valid, err_msg, _ = auth_manager.verify_api_key()
    if not is_valid:
        return (
            f"[Authentication Error] {err_msg} "
            "Please provide a valid CODEGRAPH_API_KEY in your MCP client environment or CLI args. "
            "(Generate one using: 'python -m app.auth_cli create-key --name <agent-name>')"
        )
    return None


def _check_db() -> Optional[str]:
    """Validates MongoDB connection for database operations."""
    if db is None:
        return (
            f"[Database Error] Could not connect to MongoDB at '{db_uri}'. "
            "Please ensure MongoDB Atlas IP allowlist allows your connection, or run a local MongoDB instance."
        )
    return None


# ---------------------------------------------------------------------------
# Deterministic MCP Tools (Zero LLM Keys Required)
# ---------------------------------------------------------------------------

@mcp.tool()
def index_repository(repo_root: str, repo_id: Optional[str] = None, branch: Optional[str] = None) -> str:
    """
    MANDATORY FIRST STEP: Clones (if remote Git URL) or reads a local folder,
    parses Python AST symbols, extracts programmatic call edges, and stores
    them in MongoDB as a deterministic Code Property Graph.
    
    Accepts:
    - Public Git URL (e.g., https://github.com/pallets/flask.git)
    - Local directory path (e.g., ./my-project)
    """
    auth_err = _check_auth()
    if auth_err:
        return auth_err

    db_err = _check_db()
    if db_err:
        return db_err

    try:
        resolved_local_path, actual_repo_id, is_remote, normalized_url, clone_details = resolve_repository_source(
            repo_path=repo_root,
            repo_id=repo_id,
            branch=branch,
        )
    except Exception as exc:
        return f"Failed to prepare repository source '{repo_root}': {exc}"

    collection_name = f"{actual_repo_id}_symbols"

    # Check if already indexed
    if collection_name in db.list_collection_names() and db[collection_name].estimated_document_count() > 0:
        count = db[collection_name].estimated_document_count()
        return (
            f"Repository '{repo_root}' is already indexed under repo_id: '{actual_repo_id}' ({count} symbols). "
            f"Proceed directly to search_symbols or get_repo_filemap."
        )

    try:
        indexer = RepoGraphIndexer(repo_path=resolved_local_path, repo_id=actual_repo_id, mongo_uri=db_uri)
        indexer.index_repository()
        count = db[collection_name].estimated_document_count()
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
        return (
            f"Successfully indexed '{repo_root}' ({count} symbols) under repo_id: '{actual_repo_id}'. "
            f"You can now query symbols, blast radius, and dependencies."
        )
    except Exception as exc:
        return f"Indexing failed for '{repo_root}': {exc}"


@mcp.tool()
def list_repositories() -> str:
    """
    Lists all indexed repositories currently available in MongoDB,
    including symbol counts, Git commit hashes, branches, and index timestamps.
    """
    auth_err = _check_auth()
    if auth_err:
        return auth_err

    db_err = _check_db()
    if db_err:
        return db_err

    repos = list_repositories_with_metadata(db)
    return json.dumps(repos, indent=2)


@mcp.tool()
def get_repo_filemap(repo_id: str) -> str:
    """
    Returns a complete, structured overview of all files and their top-level symbols for a repo.
    Use this to answer high-level architecture questions (e.g. 'how is this codebase structured?')
    without needing documentation or reading raw files from disk.
    """
    auth_err = _check_auth()
    if auth_err:
        return auth_err

    db_err = _check_db()
    if db_err:
        return db_err

    result = graph_tools.get_repo_filemap(repo_id=repo_id)
    return json.dumps(result, indent=2)


@mcp.tool()
def search_symbols(query: str, repo_id: str, limit: int = 10) -> str:
    """
    Finds entry-point functions, classes, and methods matching a name or query.
    Returns lightweight IDs and signatures. To view the implementation, pass the ID to get_symbol_details.
    """
    auth_err = _check_auth()
    if auth_err:
        return auth_err

    db_err = _check_db()
    if db_err:
        return db_err

    results = graph_tools.search_symbols(query=query, repo_id=repo_id, limit=limit)
    return json.dumps(results, indent=2)


@mcp.tool()
def get_symbol_details(symbol_id: str, repo_id: str) -> str:
    """
    Fetches the exact Python source code, line numbers, and direct downstream calls for a symbol.
    """
    auth_err = _check_auth()
    if auth_err:
        return auth_err

    db_err = _check_db()
    if db_err:
        return db_err

    doc = graph_tools.get_symbol_details(symbol_id=symbol_id, repo_id=repo_id)
    if "error" in doc:
        return doc["error"]

    code = doc.get("source_code", "")
    calls = doc.get("calls", [])
    deps_str = "\n".join([f"- {c}" for c in calls]) if calls else "None"
    return f"Symbol: {symbol_id}\nType: {doc.get('type')}\nLines: {doc.get('start_line')}-{doc.get('end_line')}\n\nCode:\n```python\n{code}\n```\n\nCalls (Downstream):\n{deps_str}"


@mcp.tool()
def get_upstream_blast_radius(symbol_id: str, repo_id: str, max_depth: int = 3) -> str:
    """
    IMPACT ANALYSIS (BLAST RADIUS): Performs recursive reverse graph traversal ($graphLookup)
    to find every function, method, or class that directly or indirectly depends on `symbol_id`.
    Returns affected symbols, hop distance, and the complete list of impacted files.
    """
    auth_err = _check_auth()
    if auth_err:
        return auth_err

    db_err = _check_db()
    if db_err:
        return db_err

    result = graph_tools.get_upstream_blast_radius(symbol_id=symbol_id, repo_id=repo_id, max_depth=max_depth)
    return json.dumps(result, indent=2)


@mcp.tool()
def get_downstream_dependencies(symbol_id: str, repo_id: str, max_depth: int = 2) -> str:
    """
    EXECUTION FLOW TRACING: Traverses outgoing call edges to retrieve all downstream
    functions and classes that `symbol_id` calls, including their source code.
    """
    auth_err = _check_auth()
    if auth_err:
        return auth_err

    db_err = _check_db()
    if db_err:
        return db_err

    result = graph_tools.get_downstream_dependencies(symbol_id=symbol_id, repo_id=repo_id, max_depth=max_depth)
    return json.dumps(result, indent=2)


@mcp.tool()
def get_dependency_subgraph_code(symbol_id: str, repo_id: str, max_depth: int = 2, direction: str = "both") -> str:
    """
    Extracts the full source code for `symbol_id` and all connected components across the codebase.
    `direction` can be: 'downstream' (callees), 'upstream' (callers), or 'both'.
    """
    auth_err = _check_auth()
    if auth_err:
        return auth_err

    db_err = _check_db()
    if db_err:
        return db_err

    result = graph_tools.get_dependency_subgraph_code(
        symbol_id=symbol_id, repo_id=repo_id, max_depth=max_depth, direction=direction
    )
    return json.dumps(result, indent=2)


@mcp.tool()
def read_file_content(file_path: str, repo_id: str) -> str:
    """
    Reads full file content exclusively from the AST database index without touching disk.
    """
    auth_err = _check_auth()
    if auth_err:
        return auth_err

    db_err = _check_db()
    if db_err:
        return db_err

    return graph_tools.read_file_content(file_path=file_path, repo_id=repo_id)


@mcp.tool()
def search_codebase_text(pattern: str, repo_id: str, limit: int = 25) -> str:
    """
    Fast regex search across all indexed source code in the database.
    Returns matched lines with line numbers and enclosing symbol names.
    """
    auth_err = _check_auth()
    if auth_err:
        return auth_err

    db_err = _check_db()
    if db_err:
        return db_err

    return graph_tools.search_codebase_text(pattern=pattern, repo_id=repo_id, limit=limit)


@mcp.tool()
def find_files_by_name(pattern: str, repo_id: str) -> str:
    """
    Finds file paths matching a glob or regex pattern exclusively from the database index.
    """
    auth_err = _check_auth()
    if auth_err:
        return auth_err

    db_err = _check_db()
    if db_err:
        return db_err

    files = graph_tools.find_files_by_name(pattern=pattern, repo_id=repo_id)
    return "\n".join(files) if files else f"No files matching '{pattern}' found in repo '{repo_id}'."


@mcp.tool()
def query_graph(repo_id: str, pipeline_json: str) -> str:
    """
    Executes a custom PyMongo aggregation pipeline against the {repo_id}_symbols collection.
    Allows advanced graph queries ($match, $graphLookup, $project).
    """
    auth_err = _check_auth()
    if auth_err:
        return auth_err

    db_err = _check_db()
    if db_err:
        return db_err

    try:
        pipeline = json.loads(pipeline_json)
        results = graph_tools.execute_graph_query(repo_id=repo_id, pipeline=pipeline)
        return json.dumps(results, indent=2, default=str)
    except Exception as exc:
        return f"Query failed: {exc}"


# ---------------------------------------------------------------------------
# MCP Prompts (Exposes Guided Workflows to the Host Model)
# ---------------------------------------------------------------------------

@mcp.prompt()
def plan_architectural_change(task: str, repo_id: str) -> str:
    """
    Prompt template that instructs the host LLM how to investigate a codebase
    and generate an architectural change plan / PRD using the deterministic graph tools.
    """
    return f"""You are an expert Software Architect planning an architectural change for repository '{repo_id}'.
Your assigned task: {task}

Execute the following plan using your available CodeGraph MCP tools:
1. Entry Points: Call `search_symbols` to identify relevant functions, classes, or endpoints.
2. Code Inspection: Call `get_symbol_details` to inspect the implementation of key symbols.
3. Impact Analysis: Call `get_upstream_blast_radius` to calculate which components and files will break if these symbols are modified.
4. Downstream Dependencies: Call `get_downstream_dependencies` to understand dependencies.
5. High-Level Context: Call `get_repo_filemap` if you need the broader module layout.

Synthesize your findings into a comprehensive Architectural Change PRD including:
- Summary of the problem
- Affected components & impacted files (from blast radius)
- Step-by-step implementation plan
- Migration & backward-compatibility considerations
- Verification & test recommendations
"""


@mcp.prompt()
def analyze_blast_radius(symbol_id: str, repo_id: str) -> str:
    """
    Prompt template that instructs the host LLM to analyze the blast radius of a symbol.
    """
    return f"""You are conducting an impact analysis for symbol '{symbol_id}' in repository '{repo_id}'.

1. Call `get_upstream_blast_radius(symbol_id='{symbol_id}', repo_id='{repo_id}')` to retrieve all dependent callers.
2. Call `get_symbol_details(symbol_id='{symbol_id}', repo_id='{repo_id}')` to review its signature and code.
3. Group the impacted callers by file and hop distance.
4. Explain what will break if '{symbol_id}' is modified or deleted, and provide recommended test cases.
"""


@mcp.prompt()
def codebase_overview(repo_id: str) -> str:
    """
    Prompt template that instructs the host LLM to produce a complete architecture overview.
    """
    return f"""You are tasked with providing a complete architectural overview of repository '{repo_id}'.

1. Call `get_repo_filemap(repo_id='{repo_id}')` to examine the repository structure and all top-level symbols.
2. Call `list_repositories()` to view repository metadata and commit history.
3. Identify core layers (e.g., API routes, services, data models, utilities).
4. Provide a clear, structured architecture report outlining:
   - System purpose and component layout
   - Primary entry points and workflows
   - Key abstractions and dependencies
"""


# ---------------------------------------------------------------------------
# MCP Resources
# ---------------------------------------------------------------------------

@mcp.resource("codegraph://repositories")
def resource_repositories() -> str:
    """Resource returning metadata for all indexed repositories in MongoDB."""
    if db is None:
        return json.dumps({"error": "MongoDB not connected."})
    return json.dumps(list_repositories_with_metadata(db), indent=2)


@mcp.resource("codegraph://repo/{repo_id}/summary")
def resource_repo_summary(repo_id: str) -> str:
    """Resource returning summary symbol counts and file statistics for a repo."""
    if db is None:
        return json.dumps({"error": "MongoDB not connected."})
    col_name = f"{repo_id}_symbols"
    if col_name not in db.list_collection_names():
        return json.dumps({"error": f"Repository '{repo_id}' not found."})
    col = db[col_name]
    count = col.estimated_document_count()
    distinct_files = len(col.distinct("file_path"))
    return json.dumps({
        "repo_id": repo_id,
        "symbol_count": count,
        "file_count": distinct_files,
    }, indent=2)


# ---------------------------------------------------------------------------
# Lazy Application Export (Backward compatibility for ASGI / Uvicorn if imported)
# ---------------------------------------------------------------------------
def __getattr__(name: str):
    if name == "app":
        from app.server import app as fastapi_app
        return fastapi_app
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# ---------------------------------------------------------------------------
# CLI Runner
# ---------------------------------------------------------------------------
def run_cli():
    parser = argparse.ArgumentParser(description="Codebase Intelligence MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="MCP transport protocol (default: stdio)",
    )
    parser.add_argument("--port", type=int, default=8000, help="Port for SSE transport (default: 8000)")
    parser.add_argument("--host", default="0.0.0.0", help="Host for SSE transport (default: 0.0.0.0)")
    parser.add_argument("--api-key", default=None, help="Set CODEGRAPH_API_KEY for authentication")

    args, _ = parser.parse_known_args()

    if args.api_key:
        os.environ["CODEGRAPH_API_KEY"] = args.api_key

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        # For SSE or streamable-http
        mcp.run(transport=args.transport, host=args.host, port=args.port)


if __name__ == "__main__":
    run_cli()
