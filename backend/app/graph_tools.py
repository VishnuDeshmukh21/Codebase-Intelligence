from typing import Dict, List, Any
from pymongo.database import Database
import re
class CodeGraphTools:
    def __init__(self, db: Database):
        self.db = db

    def _get_collection(self, repo_id: str):
        clean_repo_id = repo_id.replace("_symbols", "")
        return self.db[f"{clean_repo_id}_symbols"]

    def search_symbols(self, query: str, repo_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Finds entry-point symbols by name, signature, or file path in a specific repo collection."""
        col = self._get_collection(repo_id)
        regex = {"$regex": query, "$options": "i"}
        
        projection = {
            "_id": 1,
            "name": 1,
            "type": 1,
            "file_path": 1,
            "signature": 1,
            "start_line": 1,
            "end_line": 1
        }
        
        cursor = col.find(
            {"$or": [
                {"_id": regex},
                {"name": regex},
                {"file_path": regex}
            ]},
            projection
        ).limit(limit)
        
        return list(cursor)

    def get_symbol_details(self, symbol_id: str, repo_id: str) -> Dict[str, Any]:
        """Fetches exact source code, line numbers, and direct outgoing calls for a symbol."""
        col = self._get_collection(repo_id)
        doc = col.find_one({"_id": symbol_id})
        if not doc:
            return {"error": f"Symbol '{symbol_id}' not found in repo '{repo_id}'."}
        return doc

    def get_upstream_blast_radius(self, symbol_id: str, repo_id: str, max_depth: int = 3) -> Dict[str, Any]:
        """
        Reverse Graph Traversal: Finds every function/class in {repo_id}_symbols that depends on `symbol_id`.
        up to `max_depth` hops away, plus the exact list of impacted files.
        """
        col = self._get_collection(repo_id)
        target_collection_name = f"{repo_id}_symbols"

        pipeline = [
            {"$match": {"_id": symbol_id}},
            {
                "$graphLookup": {
                    "from": target_collection_name,
                    "startWith": "$_id",
                    "connectFromField": "_id",
                    "connectToField": "calls",
                    "as": "upstream_dependents",
                    "maxDepth": max_depth - 1,
                    "depthField": "hop_distance"
                }
            }
        ]
        results = list(col.aggregate(pipeline))
        if not results:
            return {"error": f"Symbol '{symbol_id}' not found in repo '{repo_id}'."}

        root = results[0]
        dependents = sorted(root.get("upstream_dependents", []), key=lambda x: x["hop_distance"])
        impacted_files = sorted(list({d.get("file_path", "") for d in dependents} | {root.get("file_path", "")}))

        return {
            "repo_id": repo_id,
            "target_symbol": symbol_id,
            "total_affected_symbols": len(dependents),
            "impacted_files": [f for f in impacted_files if f],
            "dependency_chain": [
                {
                    "symbol": d["_id"],
                    "file": d.get("file_path", ""),
                    "lines": f"{d.get('start_line')}-{d.get('end_line')}",
                    "hops_from_target": d["hop_distance"] + 1,
                    "signature": d.get("signature", "")                }
                for d in dependents
            ]
        }

    def get_downstream_dependencies(self, symbol_id: str, repo_id: str, max_depth: int = 2) -> Dict[str, Any]:
        """
        Forward Graph Traversal: Finds everything `symbol_id` calls downstream,
        including complete source code for each callee.
        """
        col = self._get_collection(repo_id)
        target_collection_name = f"{repo_id}_symbols"

        pipeline = [
            {"$match": {"_id": symbol_id}},
            {
                "$graphLookup": {
                    "from": target_collection_name,
                    "startWith": "$calls",
                    "connectFromField": "calls",
                    "connectToField": "_id",
                    "as": "downstream_callees",
                    "maxDepth": max_depth - 1,
                    "depthField": "hop_distance"
                }
            }
        ]
        results = list(col.aggregate(pipeline))
        if not results:
            return {"error": f"Symbol '{symbol_id}' not found in repo '{repo_id}'."}

        root = results[0]
        callees = sorted(root.get("downstream_callees", []), key=lambda x: x["hop_distance"])
        
        return {
            "repo_id": repo_id,
            "target_symbol": symbol_id,
            "target_source_code": root.get("source_code", ""),
            "total_callees": len(callees),
            "downstream_callees": [
                {
                    "symbol": c["_id"],
                    "file": c.get("file_path", ""),
                    "lines": f"{c.get('start_line')}-{c.get('end_line')}",
                    "hops": c["hop_distance"] + 1,
                    "signature": c.get("signature", ""),
                    "source_code": c.get("source_code", "")
                }
                for c in callees
            ]
        }

    def search_codebase_text(self, pattern: str, repo_id: str, limit: int = 30) -> str:
        """
        Searches the MongoDB index for a regex pattern and extracts matching lines.
        """
        col = self._get_collection(repo_id)
        
        try:
            # Ensure pattern is a valid regex for Python extraction later
            compiled_regex = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            return f"Invalid regex pattern: {e}"

        # MongoDB regex query on the source_code field
        cursor = col.find(
            {"source_code": {"$regex": pattern, "$options": "i"}},
            {"_id": 1, "name": 1, "file_path": 1, "start_line": 1, "type": 1, "source_code": 1}
        ).limit(limit)
        
        results = list(cursor)
        if not results:
            return f"No matches found for pattern '{pattern}' in repo '{repo_id}'."
            
        formatted_results = []
        for res in results:
            file_path = res.get("file_path", "Unknown")
            name = res.get("name", "Unknown")
            sym_type = res.get("type", "Unknown")
            start_line = res.get("start_line", 1)
            source = res.get("source_code", "")
            
            # Extract only the specific lines that match the pattern
            match_lines = []
            for i, line in enumerate(source.splitlines()):
                if compiled_regex.search(line):
                    match_lines.append(f"  Line {start_line + i}: {line.strip()}")
                    if len(match_lines) >= 3:  # Cap at 3 lines per symbol to save context
                        match_lines.append("  ... [truncated additional matches in this symbol]")
                        break
            
            lines_str = "\n".join(match_lines)
            formatted_results.append(f"- {sym_type.capitalize()} `{name}` in `{file_path}`:\n{lines_str}")
            
        return f"Found matches in {len(results)} nodes:\n\n" + "\n\n".join(formatted_results)

    def get_dependency_subgraph_code(self, symbol_id: str, repo_id: str, max_depth: int = 3, direction: str = "both") -> Dict[str, Any]:
        """
        Consolidates complete source code for a symbol and all its dependencies
        across multiple hops into a unified bundle.
        """
        col = self._get_collection(repo_id)
        target_collection_name = f"{repo_id}_symbols"
        symbol_ids_to_fetch = {symbol_id}

        # 1. Downstream Traversal (Callees)
        if direction in ("downstream", "both"):
            down_pipe = [
                {"$match": {"_id": symbol_id}},
                {
                    "$graphLookup": {
                        "from": target_collection_name,
                        "startWith": "$calls",
                        "connectFromField": "calls",
                        "connectToField": "_id",
                        "as": "callees",
                        "maxDepth": max_depth - 1
                    }
                }
            ]
            down_res = list(col.aggregate(down_pipe))
            if down_res:
                for doc in down_res[0].get("callees", []):
                    symbol_ids_to_fetch.add(doc["_id"])

        # 2. Upstream Traversal (Callers)
        if direction in ("upstream", "both"):
            up_pipe = [
                {"$match": {"_id": symbol_id}},
                {
                    "$graphLookup": {
                        "from": target_collection_name,
                        "startWith": "$_id",
                        "connectFromField": "_id",
                        "connectToField": "calls",
                        "as": "callers",
                        "maxDepth": max_depth - 1
                    }
                }
            ]
            up_res = list(col.aggregate(up_pipe))
            if up_res:
                for doc in up_res[0].get("callers", []):
                    symbol_ids_to_fetch.add(doc["_id"])

        docs = list(col.find(
            {"_id": {"$in": list(symbol_ids_to_fetch)}},
            {"_id": 1, "name": 1, "file_path": 1, "start_line": 1, "end_line": 1, "signature": 1, "source_code": 1, "type": 1}
        ))

        if not docs:
            return {"error": f"Symbol '{symbol_id}' not found in repo '{repo_id}'."}

        return {
            "repo_id": repo_id,
            "root_symbol": symbol_id,
            "total_symbols_retrieved": len(docs),
            "code_manifest": [
                {
                    "symbol_id": d["_id"],
                    "file_path": d.get("file_path", ""),
                    "type": d.get("type", ""),
                    "lines": f"{d.get('start_line')}-{d.get('end_line')}",
                    "signature": d.get("signature", ""),
                    "source_code": d.get("source_code", "")
                }
                for d in docs
            ]
        }