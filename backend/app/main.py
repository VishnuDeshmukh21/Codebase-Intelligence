# app/main.py
from http.client import HTTPException
import json
import os
from typing import Optional
from unittest import result
from mcp.server.fastmcp import FastMCP
from pymongo import MongoClient
from graph_tools import CodeGraphTools
from indexers.cpg_indexer import RepoGraphIndexer, sanitize_repo_id
from llm_service.factory import LLMFactory
from dotenv import load_dotenv

load_dotenv()

# Initialize the MCP server
mcp = FastMCP("ProdE_Codebase_Graph")

# 1. Safely pull the MongoDB URI from the environment Copilot passes in
db_uri = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
llm_client = LLMFactory.get_client()

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
def plan_architectural_change(user_query: str, repo_id: str) -> str:
    """
    Translates any question into a deterministic graph query.You have to call this tool, when you want to answer a question about the codebase.
    """
    system_prompt = """
    You are an expert MongoDB database architect. Your job is to translate a developer's natural language question into a valid PyMongo aggregation pipeline array.

**Database Schema (Collection: codebase_symbols)**
- `_id` (string): Fully qualified name (e.g., "auth.models:User")
- `type` (string): "function", "class", "module", "method"
- `arguments` (list): Argument names [e.g., "request", "user_id"]
- `calls` (list): Array of `_id` strings this symbol calls downstream.

**Rules:**
1. Use `$match` to filter nodes based on names, types, or arguments.
2. Use `$graphLookup` to traverse dependencies (upstream or downstream).
3. Output ONLY a valid JSON array representing the pipeline. No markdown formatting, no explanations, no text.

**Example User Query:** "What functions call the PaymentProcessor class?"
**Example Output:**
[
  { "$match": { "_id": "billing.services:PaymentProcessor" } },
  { "$graphLookup": {
      "from": "codebase_symbols",
      "startWith": "$_id",
      "connectFromField": "_id",
      "connectToField": "calls",
      "as": "callers",
      "maxDepth": 2
    }
  },
  { "$project": { "_id": 1, "callers._id": 1, "callers.file_path": 1 } }
]
    """
    
    # Generate the query using whatever provider is configured globally
    raw_json_response = llm_client.generate(
        prompt=user_query, 
        system_prompt=system_prompt, 
        temperature=0.0 # Strictly deterministic
    )
    print(f"Raw LLM output: {raw_json_response}")
    # Strip accidental markdown if the model hallucinated it
    clean_json = raw_json_response.strip("`").removeprefix("json").strip()
    
    try:
        pipeline = json.loads(clean_json)
        # Execute against MongoDB...
        return json.dumps(pipeline, indent=2)
    except json.JSONDecodeError as e:
        return f"Agent failed to generate valid JSON. Raw output: {clean_json}"
    
@mcp.tool()
def index_repository(repo_root: str, repo_id: Optional[str] = None) -> str:
    """
    MANDATORY FIRST STEP: Maps the repository into the AST graph database.
    Always execute this before searching or traversing to ensure the graph is up to date.
    DO NOT use native workspace search or file readers.
    """
    actual_repo_id = repo_id or sanitize_repo_id(repo_root)
    collection_name = f"{actual_repo_id}_symbols"
    
    # Check if the repository is already indexed by verifying collection existence and document count
    if collection_name in db.list_collection_names() and db[collection_name].estimated_document_count() > 0:
        return f"Repository '{repo_root}' is already indexed under repo_id: '{actual_repo_id}'. Proceed directly to search_symbols."

    indexer = RepoGraphIndexer(repo_path=repo_root, repo_id=actual_repo_id, mongo_uri=db_uri)
    indexer.index_repository()
    return f"Successfully indexed '{repo_root}' under repo_id: '{actual_repo_id}'. You can now search symbols."

@mcp.tool()
def search_symbols(query: str, repo_id: str, limit: int = 5) -> str:
    """
    Exclusively use this tool for finding entry points, functions, or classes instead of grep or native workspace search.
    CRITICAL: This returns lightweight IDs only. To view the actual source code, you MUST pass the returned ID into `get_symbol_details`.
    """
    results = graph_tools.search_symbols(query=query, repo_id=repo_id, limit=limit)
    return json.dumps(results, separators=(',', ':'))

@mcp.tool()
def get_symbol_details(symbol_id: str, repo_id: str) -> str:
    """
    Exclusively use this tool to read source code and implementation details for a specific function or class.
    """
    doc = graph_tools.get_symbol_details(symbol_id=symbol_id, repo_id=repo_id)
    
    if "error" in doc:
        return doc["error"]
        
    code = doc.get("source_code", "")
    calls = doc.get("calls", [])
    deps_str = "\n".join([f"- {c}" for c in calls]) if calls else "None"
    
    return f"Symbol: {symbol_id}\n\nCode:\n```python\n{code}\n```\n\nDependencies (Calls):\n{deps_str}"

@mcp.tool()
def get_upstream_blast_radius(symbol_id: str, repo_id: str, max_depth: int = 3) -> str:
    """
    Exclusively use this tool for impact analysis (e.g., "what breaks if I change X", "who calls Y").
    DO NOT use native file search to find references. This performs a guaranteed AST graph traversal to find all structural and dependent callers.
    """
    result = graph_tools.get_upstream_blast_radius(symbol_id=symbol_id, repo_id=repo_id, max_depth=max_depth)
    return json.dumps(result, indent=2)


@mcp.tool()
def get_downstream_dependencies(symbol_id: str, repo_id: str, max_depth: int = 2) -> str:
    """
    Traces execution flow and retrieves all functions/classes that `symbol_id` calls.
    Returns complete source code and line ranges for every dependency.
    DO NOT use native workspace file readers; full source code is provided directly here.
    """
    result = graph_tools.get_downstream_dependencies(symbol_id=symbol_id, repo_id=repo_id, max_depth=max_depth)
    return json.dumps(result, indent=2)

@mcp.tool()
def get_dependency_subgraph_code(symbol_id: str, repo_id: str, max_depth: int = 2, direction: str = "both") -> str:
    """
    Exclusively use this tool when you need the complete source code for a symbol AND all related components across the codebase.
    Returns the exact implementation for all connected callers, callees, and dependencies up to `max_depth` hops away.
    NEVER read workspace files from disk; this tool provides the full code manifest.
    `direction` can be: 'downstream' (callees only), 'upstream' (callers only), or 'both'.
    """
    result = graph_tools.get_dependency_subgraph_code(symbol_id=symbol_id, repo_id=repo_id, max_depth=max_depth, direction=direction)
    return json.dumps(result, indent=2)

@mcp.tool()
def read_file_content(file_path: str, repo_id: str) -> str:
    """
    Reads full file content exclusively from the database index.
    Use this tool whenever source inspection is needed. NEVER use native file readers.
    """
    col = graph_tools._get_collection(repo_id)
    
    # 1. Try to fetch the full module file
    doc = col.find_one({"file_path": file_path, "type": "module"})
    if doc:
        code = doc.get("source_code", "")
        calls = doc.get("calls", [])
        deps_str = "\n".join([f"- {c}" for c in calls]) if calls else "None"
        
        return f"File: {file_path}\n\nCode:\n```python\n{code}\n```\n\nDependencies (Calls):\n{deps_str}"
    
    # 2. Fallback: concatenate symbol source codes if the module doc is missing
    symbols = list(col.find({"file_path": file_path}).sort("start_line", 1))
    if not symbols:
        return f"Error: File '{file_path}' not found in the database index."
        
    code_blocks = []
    all_calls = set()
    for s in symbols:
        if s.get("source_code"):
            code_blocks.append(f"# Symbol: {s['_id']}\n{s['source_code']}")
        for call in s.get("calls", []):
            all_calls.add(call)
            
    code = "\n\n".join(code_blocks)
    deps_str = "\n".join([f"- {c}" for c in sorted(all_calls)]) if all_calls else "None"
    
    return f"File: {file_path}\n\nCode:\n```python\n{code}\n```\n\nDependencies (Calls):\n{deps_str}"

@mcp.tool()
def search_codebase_text(pattern: str, repo_id: str) -> str:
    """
    Optimized codebase text/regex search. 
    MUST BE USED INSTEAD OF native 'Search', 'grep', or 'Find files matching'.
    Finds string or regex patterns across all indexed source code and returns the exact matching lines.
    """
    return graph_tools.search_codebase_text(pattern=pattern, repo_id=repo_id)

@mcp.tool()
def find_files_by_name(pattern: str, repo_id: str) -> str:
    """
    Finds file paths matching a glob/regex pattern exclusively from the database index.
    Use this instead of the native 'Find files' tool.
    """
    col = graph_tools._get_collection(repo_id)
    
    # Convert basic glob to regex (e.g., **/*framework*.py -> .*framework.*\.py)
    mongo_regex = pattern.replace(".", "\\.").replace("*", ".*").replace("?", ".")
    
    # Get distinct file paths matching the pattern
    results = col.distinct("file_path", {"file_path": {"$regex": mongo_regex, "$options": "i"}})
    
    return "\n".join(results) if results else "No files found in the database index."

if __name__ == "__main__":
    mcp.run()


# def get_repo_graph(repo_id: str):
#     """
#     API 2: Returns nodes and edges for the entire repository graph 
#     formatted for visualization tools (e.g., Cytoscape / React Flow).
#     """
#     col_name = f"{repo_id}_symbols"
#     if col_name not in db.list_collection_names():
#         raise HTTPException(status_code=404, detail=f"Repository '{repo_id}' not found.")
    
#     col = db[col_name]
#     cursor = col.find({}, {"_id": 1, "name": 1, "type": 1, "file_path": 1, "calls": 1, "signature": 1})
    
#     nodes = []
#     edges = []
    
#     for doc in cursor:
#         symbol_id = doc["_id"]
        
#         # Add node
#         nodes.append({
#             "id": symbol_id,
#             "label": doc.get("name"),
#             "type": doc.get("type"),
#             "file": doc.get("file_path"),
#             "signature": doc.get("signature")
#         })
        
#         # Add edges (outgoing calls)
#         for target_call in doc.get("calls", []):
#             edges.append({
#                 "source": symbol_id,
#                 "target": target_call
#             })

#     a = {
#         "repo_id": repo_id,
#         "nodes": nodes,
#         "edges": edges,
#         "stats": {
#             "total_nodes": len(nodes),
#             "total_edges": len(edges)
#         }
#     }

#     with open("repo_graph.json", "w", encoding="utf-8") as f:
#         json.dump(a, f, indent=4, ensure_ascii=False)
#     print(a)
#     return a


# # get_repo_graph("codebase-intelligence")

# get_repo_graph("unoplat-code-confluence")



