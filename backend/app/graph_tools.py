from typing import Dict, List, Any, Optional
from pymongo.database import Database
import re

try:
    from app.embedding_service import generate_query_embedding, rank_by_cosine_similarity
except ImportError:
    try:
        from embedding_service import generate_query_embedding, rank_by_cosine_similarity
    except ImportError:
        generate_query_embedding = None
        rank_by_cosine_similarity = None


class CodeGraphTools:
    def __init__(self, db: Database):
        self.db = db

    def _get_collection(self, repo_id: str):
        clean_repo_id = repo_id.replace("_symbols", "")
        return self.db[f"{clean_repo_id}_symbols"]

    def semantic_search_symbols(self, query: str, repo_id: str, limit: int = 10, min_score: float = 0.25) -> List[Dict[str, Any]]:
        """
        Performs semantic vector search over codebase symbols using local dense embeddings.
        Maps conceptual intent (e.g. 'catalog configuration') to matching symbols (e.g. 'AiModelConfigService').
        """
        if not generate_query_embedding or not rank_by_cosine_similarity:
            return []

        col = self._get_collection(repo_id)
        try:
            query_vector = generate_query_embedding(query)
        except Exception:
            return []

        if not query_vector:
            return []

        # 1. Try Atlas Vector Search if configured
        try:
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": "vector_index",
                        "path": "embedding",
                        "queryVector": query_vector,
                        "numCandidates": limit * 5,
                        "limit": limit
                    }
                },
                {
                    "$project": {
                        "_id": 1,
                        "name": 1,
                        "type": 1,
                        "file_path": 1,
                        "signature": 1,
                        "start_line": 1,
                        "end_line": 1,
                        "docstring": 1,
                        "score": {"$meta": "vectorSearchScore"}
                    }
                }
            ]
            results = list(col.aggregate(pipeline))
            if results:
                return results
        except Exception:
            pass  # Fall back to local vector search

        # 2. Local in-memory cosine similarity search
        projection = {
            "_id": 1,
            "name": 1,
            "type": 1,
            "file_path": 1,
            "signature": 1,
            "start_line": 1,
            "end_line": 1,
            "docstring": 1,
            "embedding": 1
        }
        cursor = col.find({"embedding": {"$exists": True, "$type": "array"}}, projection)
        candidates = list(cursor)
        if not candidates:
            return []

        candidate_embeddings = [c["embedding"] for c in candidates]
        ranked = rank_by_cosine_similarity(query_vector, candidate_embeddings, top_k=limit, min_score=min_score)

        results = []
        for idx, score in ranked:
            doc = dict(candidates[idx])
            doc.pop("embedding", None)
            doc["score"] = round(score, 4)
            results.append(doc)

        return results

    def search_symbols(self, query: str, repo_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Hybrid symbol search: Combines deterministic lexical matching with semantic vector search.
        Finds exact symbol identifiers as well as conceptual intent matches.
        """
        col = self._get_collection(repo_id)
        escaped_query = re.escape(query)
        regex = {"$regex": escaped_query, "$options": "i"}
        
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
        )
        if hasattr(cursor, "limit"):
            cursor = cursor.limit(limit)
        lexical_hits = list(cursor)[:limit]

        # Semantic vector search
        semantic_hits = self.semantic_search_symbols(query=query, repo_id=repo_id, limit=limit)

        # Merge results, prioritizing exact/lexical hits but enriching with semantic hits
        seen_ids = set()
        merged = []

        for item in lexical_hits:
            if item["_id"] not in seen_ids:
                seen_ids.add(item["_id"])
                item["match_type"] = "lexical"
                merged.append(item)

        for item in semantic_hits:
            if item["_id"] not in seen_ids:
                seen_ids.add(item["_id"])
                item["match_type"] = "semantic"
                merged.append(item)

        return merged[:limit]

    def resolve_intent_context(self, intent: str, repo_id: str, max_hops: int = 1, top_k: int = 3) -> Dict[str, Any]:
        """
        High-Performance Compound Retrieval:
        1. Resolves conceptual intent to top seed symbols using vector embeddings.
        2. Traverses call graphs in MongoDB via $graphLookup to extract direct dependencies.
        3. Assembles a deduplicated code manifest in a single ~100ms round trip.
        """
        col = self._get_collection(repo_id)
        clean_repo_id = repo_id.replace("_symbols", "")
        target_collection_name = f"{clean_repo_id}_symbols"

        # 1. Discover top entry seeds semantically
        seeds = self.semantic_search_symbols(query=intent, repo_id=repo_id, limit=top_k)
        
        # If semantic search yields nothing (e.g. index without embeddings), fallback to lexical
        if not seeds:
            words = [w for w in re.split(r'\W+', intent) if len(w) > 3]
            for w in words[:3]:
                cur = col.find(
                    {"$or": [{"name": {"$regex": re.escape(w), "$options": "i"}}]},
                    {"_id": 1, "name": 1, "type": 1, "file_path": 1, "signature": 1, "start_line": 1, "end_line": 1}
                )
                if hasattr(cur, "limit"):
                    cur = cur.limit(top_k)
                lexical = list(cur)[:top_k]
                if lexical:
                    seeds.extend(lexical)
                    break

        if not seeds:
            return {
                "repo_id": repo_id,
                "intent": intent,
                "message": f"No relevant symbols found for intent '{intent}'. Try alternative keywords.",
                "matched_seeds": [],
                "code_manifest": []
            }

        seed_ids = [s["_id"] for s in seeds]

        # 2. Traverse downstream dependencies for all seeds in parallel via $graphLookup
        pipeline = [
            {"$match": {"_id": {"$in": seed_ids}}},
            {
                "$graphLookup": {
                    "from": target_collection_name,
                    "startWith": "$calls",
                    "connectFromField": "calls",
                    "connectToField": "_id",
                    "as": "callees",
                    "maxDepth": max_hops - 1,
                    "depthField": "hop_distance"
                }
            }
        ]
        
        lookup_results = list(col.aggregate(pipeline))
        
        # Collect all symbol IDs (seeds + callees)
        all_symbol_ids = set(seed_ids)
        for row in lookup_results:
            for callee in row.get("callees", []):
                all_symbol_ids.add(callee["_id"])

        # 3. Fetch full code docs for all symbols in a single query
        code_docs = list(col.find(
            {"_id": {"$in": list(all_symbol_ids)}},
            {"_id": 1, "name": 1, "type": 1, "file_path": 1, "signature": 1, "start_line": 1, "end_line": 1, "source_code": 1, "calls": 1}
        ))

        manifest = []
        for d in code_docs:
            is_seed = d["_id"] in seed_ids
            manifest.append({
                "symbol_id": d["_id"],
                "role": "seed_entry_point" if is_seed else "direct_dependency",
                "name": d.get("name", ""),
                "type": d.get("type", ""),
                "file_path": d.get("file_path", ""),
                "lines": f"{d.get('start_line')}-{d.get('end_line')}",
                "signature": d.get("signature", ""),
                "calls": d.get("calls", []),
                "source_code": d.get("source_code", "")
            })

        # Sort seeds first, then dependencies
        manifest.sort(key=lambda m: (0 if m["role"] == "seed_entry_point" else 1, m["file_path"]))

        return {
            "repo_id": repo_id,
            "intent": intent,
            "total_symbols_retrieved": len(manifest),
            "matched_seeds": [
                {
                    "symbol_id": s["_id"],
                    "name": s.get("name"),
                    "type": s.get("type"),
                    "file_path": s.get("file_path"),
                    "score": s.get("score")
                }
                for s in seeds
            ],
            "code_manifest": manifest
        }

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
                    "signature": d.get("signature", "")
                }
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

    def get_repo_filemap(self, repo_id: str) -> Dict[str, Any]:
        """
        Returns a structured map of all files and their top-level symbols for a repository.
        Enables host AI agents to quickly understand the whole codebase architecture.
        """
        col = self._get_collection(repo_id)
        cursor = col.find(
            {},
            {"_id": 1, "name": 1, "type": 1, "file_path": 1, "signature": 1, "start_line": 1, "end_line": 1}
        )

        file_map: Dict[str, list] = {}
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

        for fp in file_map:
            file_map[fp].sort(key=lambda s: int((s.get("lines") or "0").split("-")[0] or 0))

        return {
            "repo_id": repo_id,
            "total_files": len(file_map),
            "total_symbols": sum(len(v) for v in file_map.values()),
            "file_map": file_map,
        }

    def read_file_content(self, file_path: str, repo_id: str) -> str:
        """
        Reads module or file content directly from the AST database index.
        """
        col = self._get_collection(repo_id)

        # 1. Try to fetch the full module file
        doc = col.find_one({"file_path": file_path, "type": "module"})
        if doc:
            code = doc.get("source_code", "")
            calls = doc.get("calls", [])
            deps_str = "\n".join([f"- {c}" for c in calls]) if calls else "None"
            return f"File: {file_path}\n\nCode:\n```python\n{code}\n```\n\nDependencies (Calls):\n{deps_str}"

        # 2. Fallback: concatenate symbol source codes if module doc is missing
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

    def find_files_by_name(self, pattern: str, repo_id: str) -> List[str]:
        """
        Finds file paths matching a glob/regex pattern from the database index.
        """
        col = self._get_collection(repo_id)
        mongo_regex = pattern.replace(".", "\\.").replace("*", ".*").replace("?", ".")
        return col.distinct("file_path", {"file_path": {"$regex": mongo_regex, "$options": "i"}})

    def execute_graph_query(self, repo_id: str, pipeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Executes an aggregation pipeline against {repo_id}_symbols.
        """
        col = self._get_collection(repo_id)
        return list(col.aggregate(pipeline))
