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
def search_symbols(repo_id: str, query: str, limit: int = 8) -> str:
    """Finds entry-point symbols by name or file path within a specific repository."""
    results = graph_tools.search_symbols(repo_id, query, limit)
    return json.dumps(results, indent=2)

@mcp.tool()
def get_symbol_details(repo_id: str, symbol_id: str) -> str:
    """Fetches exact source code and implementation for a symbol in a specific repository."""
    result = graph_tools.get_symbol_details(repo_id, symbol_id)
    return json.dumps(result, indent=2)

@mcp.tool()
def get_upstream_blast_radius(repo_id: str, symbol_id: str, max_depth: int = 3) -> str:
    """
    Reverse Graph Traversal: Finds every function/class in the repo that depends on a symbol.
    CRITICAL: You MUST use this tool whenever a user asks "what gets affected", 
    "what breaks", "what is the impact", or "what relies on" a specific codebase component.
    Do NOT use native file search for impact analysis.
    """
    result = graph_tools.get_upstream_blast_radius(repo_id, symbol_id, max_depth)
    return json.dumps(result, indent=2)

@mcp.tool()
def get_downstream_dependencies(repo_id: str, symbol_id: str, max_depth: int = 2) -> str:
    """Finds everything a symbol calls downstream within the specified repo."""
    result = graph_tools.get_downstream_dependencies(repo_id, symbol_id, max_depth)
    return json.dumps(result, indent=2)

if __name__ == "__main__":
    mcp.run()