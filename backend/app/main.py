# app/main.py
import json
import os
from typing import Optional
from mcp.server.fastmcp import FastMCP
from pymongo import MongoClient
from graph_tools import CodeGraphTools
from indexer import RepoGraphIndexer, sanitize_repo_id
from dotenv import load_dotenv

load_dotenv()

# Initialize the MCP server
mcp = FastMCP("ProdE_Codebase_Graph")

# 1. Safely pull the MongoDB URI from the environment Copilot passes in
db_uri = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")

# # Connect to MongoDB Atlas
# client = MongoClient(db_uri)
# db = client["prode_engine"]
# graph_tools = CodeGraphTools(db)

try:
    client = MongoClient(db_uri, serverSelectionTimeoutMS=10000)
    client.admin.command("ping")
    print("Successfully connected to MongoDB!")

    db = client["prode_engine"]
    graph_tools = CodeGraphTools(db)

except Exception as e:
    print(f"Failed to connect to MongoDB: {e}")
    raise

@mcp.tool()
def index_repository(repo_root: str, repo_id: Optional[str] = None) -> str:
    """Parses a local repo. If repo_id is omitted, it auto-detects from the folder name."""
    actual_repo_id = repo_id or sanitize_repo_id(repo_root)
    indexer = RepoGraphIndexer(repo_path=repo_root, repo_id=actual_repo_id, mongo_uri=db_uri)
    indexer.index_repository()
    return f"Indexed '{repo_root}' under repo_id: '{actual_repo_id}'."

@mcp.tool()
def search_symbols(query: str, repo_id: str, limit: int = 5) -> str:
    """
    Finds entry-point symbols by name, signature, or file path. 
    CRITICAL: This tool only returns lightweight IDs. Once you find the correct symbol_id, 
    you MUST use `get_symbol_details`, `get_downstream_dependencies`, or 
    `get_upstream_blast_radius` to understand its execution flow and implementation.
    """
    results = graph_tools.search_symbols(query=query, repo_id=repo_id, limit=limit)
    return json.dumps(results, separators=(',', ':'))

@mcp.tool()
def get_symbol_details(symbol_id: str, repo_id: str) -> str:
    """Fetches exact source code and implementation for a symbol in a specific repository."""
    result = graph_tools.get_symbol_details(symbol_id=symbol_id, repo_id=repo_id)
    return json.dumps(result, indent=2)

@mcp.tool()
def get_upstream_blast_radius(symbol_id: str, repo_id: str, max_depth: int = 3) -> str:
    """
    Reverse Graph Traversal: Finds every function/class in the repo that depends on a symbol.
    CRITICAL: You MUST use this tool whenever a user asks "what gets affected", 
    "what breaks", "what is the impact", or "what relies on" a specific codebase component.
    Do NOT use native file search for impact analysis.
    """
    result = graph_tools.get_upstream_blast_radius(symbol_id=symbol_id, repo_id=repo_id, max_depth=max_depth)
    return json.dumps(result, indent=2)

@mcp.tool()
def get_downstream_dependencies(symbol_id: str, repo_id: str, max_depth: int = 2) -> str:
    """Forward Graph Traversal: Finds everything a symbol calls downstream within the specified repo."""
    result = graph_tools.get_downstream_dependencies(symbol_id=symbol_id, repo_id=repo_id, max_depth=max_depth)
    return json.dumps(result, indent=2)

if __name__ == "__main__":
    mcp.run()