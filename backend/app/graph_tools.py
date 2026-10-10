from typing import Dict, List, Any
from pymongo.database import Database

class CodeGraphTools:
    def __init__(self, db: Database):
        
        self.db = db
        self.symbols = db.symbols  # Assuming a collection named 'symbols' exists

    def search_symbols(self, repo_id: str, query: str, limit: int = 8) -> List[Dict[str, Any]]:
        """Finds entry-point symbols by name, signature, or file path."""
        col = self.db[f"{repo_id}_symbols"]
        regex = {"$regex": query, "$options": "i"}
        regex = {"$regex": query, "$options": "i"}
        cursor = col.find(
            {"$or": [{"_id": regex}, {"name": regex}, {"file_path": regex}, {"docstring": regex}]},
            {"source_code": 0, "imports": 0}
        ).limit(limit)
        return list(cursor)

    def get_symbol_details(self, repo_id: str, symbol_id: str) -> Dict[str, Any]:
            col = self.db[f"{repo_id}_symbols"]
            doc = col.find_one({"_id": symbol_id})
            if not doc:
                return {"error": f"Symbol '{symbol_id}' not found in repo '{repo_id}'."}
            return doc

    def get_upstream_blast_radius(self, repo_id: str, symbol_id: str, max_depth: int = 3) -> Dict[str, Any]:
        """
        Reverse Graph Traversal: Finds every function/class in the repo that depends on `symbol_id`
        up to `max_depth` hops away, plus the exact list of impacted files.
        """
        pipeline = [
            {"$match": {"_id": symbol_id}},
            {
                "$graphLookup": {
                    "from": f"{repo_id}_symbols",
                    "startWith": "$_id",
                    "connectFromField": "_id",
                    "connectToField": "calls", # Reverse edge lookup: who has my _id in their `calls` array?
                    "as": "upstream_dependents",
                    "maxDepth": max_depth - 1,
                    "depthField": "hop_distance"
                }
            }
        ]
        results = list(self.symbols.aggregate(pipeline))
        if not results:
            return {"error": f"Symbol '{symbol_id}' not found."}

        root = results[0]
        dependents = sorted(root.get("upstream_dependents", []), key=lambda x: x["hop_distance"])
        
        impacted_files = sorted(list({d["file_path"] for d in dependents} | {root["file_path"]}))
        
        return {
            "target_symbol": symbol_id,
            "total_affected_symbols": len(dependents),
            "impacted_files": impacted_files,
            "dependency_chain": [
                {
                    "symbol": d["_id"],
                    "file": d["file_path"],
                    "lines": f"{d['start_line']}-{d['end_line']}",
                    "hops_from_target": d["hop_distance"] + 1,
                    "signature": d["signature"]
                }
                for d in dependents
            ]
        }

    def get_downstream_dependencies(self, repo_id: str, symbol_id: str, max_depth: int = 2) -> Dict[str, Any]:
        """
        Forward Graph Traversal: Finds everything `symbol_id` calls downstream.
        """
        pipeline = [
            {"$match": {"_id": symbol_id}},
            {
                "$graphLookup": {
                    "from": f"{repo_id}_symbols",
                    "startWith": "$calls",
                    "connectFromField": "calls",
                    "connectToField": "_id", # Forward edge lookup: follow the `calls` array to `_id`
                    "as": "downstream_callees",
                    "maxDepth": max_depth - 1,
                    "depthField": "hop_distance"
                }
            }
        ]
        results = list(self.symbols.aggregate(pipeline))
        if not results:
            return {"error": f"Symbol '{symbol_id}' not found."}

        callees = sorted(results[0].get("downstream_callees", []), key=lambda x: x["hop_distance"])
        return {
            "target_symbol": symbol_id,
            "downstream_callees": [
                {
                    "symbol": c["_id"],
                    "file": c["file_path"],
                    "hops": c["hop_distance"] + 1,
                    "signature": c["signature"]
                }
                for c in callees
            ]
        }