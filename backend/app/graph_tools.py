from typing import Dict, List, Any
from pymongo.database import Database

class CodeGraphTools:
    def __init__(self, db: Database):
        self.db = db

    def _get_collection(self, repo_id: str):
        return self.db[f"{repo_id}_symbols"]

    def search_symbols(self, query: str, repo_id: str, limit: int = 8) -> List[Dict[str, Any]]:
        """Finds entry-point symbols by name, signature, or file path in a specific repo collection."""
        col = self._get_collection(repo_id)
        regex = {"$regex": query, "$options": "i"}
        cursor = col.find(
            {"$or": [{"_id": regex}, {"name": regex}, {"file_path": regex}, {"docstring": regex}]},
            {"source_code": 0, "imports": 0}
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
                    "from": target_collection_name,  # MUST point to the repo-specific collection
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
        impacted_files = sorted(list({d["file_path"] for d in dependents} | {root.get("file_path", "")}))

        return {
            "repo_id": repo_id,
            "target_symbol": symbol_id,
            "total_affected_symbols": len(dependents),
            "impacted_files": [f for f in impacted_files if f],
            "dependency_chain": [
                {
                    "symbol": d["_id"],
                    "file": d["file_path"],
                    "lines": f"{d.get('start_line')}-{d.get('end_line')}",
                    "hops_from_target": d["hop_distance"] + 1,
                    "signature": d.get("signature", "")
                }
                for d in dependents
            ]
        }

    def get_downstream_dependencies(self, symbol_id: str, repo_id: str, max_depth: int = 2) -> Dict[str, Any]:
        """
        Forward Graph Traversal: Finds everything `symbol_id` calls downstream in {repo_id}_symbols.
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

        callees = sorted(results[0].get("downstream_callees", []), key=lambda x: x["hop_distance"])
        return {
            "repo_id": repo_id,
            "target_symbol": symbol_id,
            "downstream_callees": [
                {
                    "symbol": c["_id"],
                    "file": c["file_path"],
                    "hops": c["hop_distance"] + 1,
                    "signature": c.get("signature", "")
                }
                for c in callees
            ]
        }