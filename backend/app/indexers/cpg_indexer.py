import ast
import os
from pathlib import Path
from typing import Callable, Dict, List, Set, Optional
from pymongo import MongoClient, UpdateOne
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from repo_manager import sanitize_repo_id
except ImportError:
    try:
        from app.repo_manager import sanitize_repo_id
    except ImportError:
        def sanitize_repo_id(path_or_name: str) -> str:
            name = Path(path_or_name).name
            clean_name = re.sub(r'[^a-zA-Z0-9_-]', '_', name).strip('_').lower()
            return clean_name or "default_repo"

class RepoGraphIndexer:
    def __init__(
        self,
        repo_path: str,
        repo_id: str,
        mongo_uri: str = "mongodb://localhost:27017",
        db_name: str = "prode_engine",
        progress_callback: Optional[Callable[[Dict], None]] = None,
    ):
        self.repo_path = Path(repo_path).resolve()
        self.repo_id = repo_id
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.symbols_col = self.db[f"{self.repo_id}_symbols"]
        self.progress_callback = progress_callback
        
        self.known_symbols: Set[str] = set()
        self.file_imports: Dict[str, Dict[str, str]] = {}
        self.class_methods: Dict[str, Set[str]] = {}
        self.file_variable_types: Dict[str, Dict[str, str]] = {}

    def _file_to_module(self, file_path: Path) -> str:
        rel = file_path.relative_to(self.repo_path).with_suffix("")
        parts = list(rel.parts)
        if parts[-1] == "__init__":
            parts.pop()
        return ".".join(parts)

    def _report_progress(self, **progress):
        if self.progress_callback:
            self.progress_callback(progress)

    def index_repository(self):
        py_files = [
            p for p in self.repo_path.rglob("*.py")
            if not any(part.startswith(".") or part in ("venv", "__pycache__", "node_modules", "tests") for part in p.parts)
        ]

        print(f"[Pass 1] Concurrently parsing {len(py_files)} files...")
        self._report_progress(
            phase="parsing",
            total_files=len(py_files),
            files_completed=0,
            parsed_files=0,
            current_file=None,
            percent=1 if py_files else 20,
        )

        parsed_trees: Dict[Path, tuple] = {}
        parsed_files = 0

        # Concurrently read and parse AST trees using ThreadPoolExecutor (safe & low memory overhead)
        def parse_single_file(file_path: Path):
            try:
                source = file_path.read_text(encoding="utf-8")
                tree = ast.parse(source, filename=str(file_path))
                return file_path, tree, source, None
            except Exception as e:
                return file_path, None, None, e

        max_workers = min(32, (os.cpu_count() or 1) * 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {executor.submit(parse_single_file, fp): fp for fp in py_files}
            
            for future in as_completed(future_to_file):
                file_path, tree, source, error = future.result()
                parsed_files += 1
                if error:
                    print(f"Skipping {file_path}: {error}")
                else:
                    parsed_trees[file_path] = (tree, source)
                
                if parsed_files % 10 == 0 or parsed_files == len(py_files):
                    self._report_progress(
                        phase="parsing",
                        total_files=len(py_files),
                        files_completed=0,
                        parsed_files=parsed_files,
                        current_file=str(file_path.relative_to(self.repo_path)),
                        percent=5 + int(15 * parsed_files / max(len(py_files), 1)),
                    )

        # Pass 1 Discovery (sequential over pre-parsed trees for safety and correctness)
        for file_path, (tree, source) in parsed_trees.items():
            self._pass_one_discover(file_path, tree)

        print(f"[Pass 2] Extracting symbols for {len(self.known_symbols)} symbols...")

        all_docs = []
        files_completed = 0

        for file_path, (tree, source) in parsed_trees.items():
            docs = self._pass_two_extract(file_path, tree, source)
            all_docs.extend(docs)
            files_completed += 1
            self._report_progress(
                phase="parsing",
                total_files=len(py_files),
                files_completed=files_completed,
                parsed_files=parsed_files,
                current_file=str(file_path.relative_to(self.repo_path)),
                percent=5 + int(15 * files_completed / max(len(py_files), 1)),
            )

        batch_ops = []
        total_indexed = 0
        db_batch_size = 100
        for doc in all_docs:
            batch_ops.append(
                UpdateOne(
                    {"_id": doc["_id"]},
                    {"$set": doc, "$unset": {"embedding": ""}},
                    upsert=True,
                )
            )
            if len(batch_ops) >= db_batch_size:
                self.symbols_col.bulk_write(batch_ops)
                total_indexed += len(batch_ops)
                batch_ops = []

        if batch_ops:
            self.symbols_col.bulk_write(batch_ops)
            total_indexed += len(batch_ops)

        if total_indexed > 0:
            self.symbols_col.create_index("calls")
            self.symbols_col.create_index("file_path")
            self.symbols_col.create_index("name")
            self.symbols_col.create_index("type")
            print(f"Successfully indexed {total_indexed} symbols into collection {self.repo_id}_symbols.")

        self._report_progress(
            phase="parsed",
            total_files=len(py_files),
            files_completed=files_completed,
            parsed_files=parsed_files,
            current_file=None,
            percent=20,
            symbols_indexed=total_indexed,
        )

    def generate_missing_embeddings(
        self,
        batch_size: int = 16,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> Dict[str, int | bool]:
        """Embed MongoDB symbols without vectors, persisting each batch for safe resumption."""
        if batch_size < 1:
            raise ValueError("Embedding batch_size must be greater than zero.")

        try:
            from app.embedding_service import build_symbol_embedding_text, generate_embeddings
        except ImportError:
            from embedding_service import build_symbol_embedding_text, generate_embeddings

        missing_filter = {
            "$or": [
                {"embedding": {"$exists": False}},
                {"embedding": None},
            ]
        }
        total_symbols = self.symbols_col.count_documents({})
        initially_missing = self.symbols_col.count_documents(missing_filter)
        already_embedded = total_symbols - initially_missing
        completed = already_embedded
        processed_this_run = 0
        all_files = set(self.symbols_col.distinct("file_path"))
        missing_files = sorted(
            self.symbols_col.distinct("file_path", missing_filter),
            key=lambda value: str(value or ""),
        )
        files_completed = len(all_files) - len(missing_files)

        def embed_batch(documents: List[Dict], file_path: str) -> None:
            nonlocal completed, processed_this_run
            if not documents:
                return

            texts = [build_symbol_embedding_text(doc) for doc in documents]
            vectors = generate_embeddings(texts, batch_size=batch_size)
            if len(vectors) != len(documents) or any(not vector for vector in vectors):
                raise RuntimeError(
                    f"Embedding service returned {len(vectors)} vectors for "
                    f"{len(documents)} symbols."
                )

            operations = [
                UpdateOne(
                    {"_id": doc["_id"], **missing_filter},
                    {"$set": {"embedding": vector}},
                )
                for doc, vector in zip(documents, vectors)
            ]
            result = self.symbols_col.bulk_write(operations, ordered=False)
            written = result.modified_count
            completed += written
            processed_this_run += written
            self._report_progress(
                phase="embedding",
                symbols_indexed=total_symbols,
                embeddings_generated=completed,
                embedding_symbols_completed=completed,
                embedding_symbols_total=total_symbols,
                embedding_files_completed=files_completed,
                embedding_files_total=len(all_files),
                current_file=file_path,
                percent=20 + int(75 * completed / max(total_symbols, 1)),
            )

        stopped = False
        for file_path in missing_files:
            if should_stop and should_stop():
                stopped = True
                break
            file_filter = {
                "file_path": file_path,
                **missing_filter,
            }
            batch = []
            for doc in self.symbols_col.find(file_filter):
                batch.append(doc)
                if len(batch) == batch_size:
                    if should_stop and should_stop():
                        stopped = True
                        break
                    embed_batch(batch, file_path)
                    batch = []
                    if should_stop and should_stop():
                        stopped = True
                        break

            if stopped:
                break
            if batch:
                if should_stop and should_stop():
                    stopped = True
                    break
                embed_batch(batch, file_path)
            files_completed += 1

        remaining = self.symbols_col.count_documents(missing_filter)
        stopped = bool(remaining and stopped)
        return {
            "embeddings_generated": processed_this_run,
            "embedding_symbols_completed": total_symbols - remaining,
            "embedding_symbols_total": total_symbols,
            "embedding_files_completed": files_completed,
            "embedding_files_total": len(all_files),
            "remaining": remaining,
            "stopped": stopped,
        }

    def _pass_one_discover(self, file_path: Path, tree: ast.AST):
        mod_name = self._file_to_module(file_path)
        self.file_imports[mod_name] = {}
        self.known_symbols.add(mod_name)

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    local_name = alias.asname or alias.name
                    self.file_imports[mod_name][local_name] = alias.name
            elif isinstance(node, ast.ImportFrom):
                base_mod = node.module or ""
                if node.level > 0:
                    parent_parts = mod_name.split(".")[:-node.level]
                    base_mod = ".".join(parent_parts + ([base_mod] if base_mod else []))
                for alias in node.names:
                    local_name = alias.asname or alias.name
                    self.file_imports[mod_name][local_name] = f"{base_mod}:{alias.name}"
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                fqn = f"{mod_name}:{node.name}"
                self.known_symbols.add(fqn)
                self.file_imports[mod_name][node.name] = fqn
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        fqn = f"{mod_name}:{target.id}"
                        self.known_symbols.add(fqn)
                        self.file_imports[mod_name][target.id] = fqn
            elif isinstance(node, ast.ClassDef):
                class_fqn = f"{mod_name}:{node.name}"
                self.known_symbols.add(class_fqn)
                self.file_imports[mod_name][node.name] = class_fqn
                self.class_methods[class_fqn] = set()

                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        method_fqn = f"{class_fqn}.{item.name}"
                        self.known_symbols.add(method_fqn)
                        self.class_methods[class_fqn].add(item.name)

    def _pass_two_extract(self, file_path: Path, tree: ast.AST, source: str) -> List[Dict]:
        mod_name = self._file_to_module(file_path)
        rel_path = str(file_path.relative_to(self.repo_path))
        lines = source.splitlines()
        docs = []

        self._extract_variable_types(tree, mod_name)
        docs.append(self._build_module_doc(tree, mod_name, rel_path, lines))

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                docs.append(self._build_symbol_doc(
                    node, f"{mod_name}:{node.name}", "function", mod_name, None, rel_path, lines
                ))
            elif isinstance(node, ast.ClassDef):
                class_fqn = f"{mod_name}:{node.name}"
                bases = [self._resolve_name(ast.unparse(b), mod_name, None) for b in node.bases]
                method_fqns = [f"{class_fqn}.{item.name}" for item in node.body if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))]
                
                docs.append({
                    "_id": class_fqn,
                    "name": node.name,
                    "type": "class",
                    "file_path": rel_path,
                    "start_line": node.lineno,
                    "end_line": node.end_lineno,
                    "signature": f"class {node.name}({', '.join(ast.unparse(b) for b in node.bases)}):",
                    "docstring": ast.get_docstring(node) or "",
                    "source_code": "\n".join(lines[node.lineno - 1 : node.end_lineno]),
                    "calls": method_fqns,
                    "is_hub": len(method_fqns) > 2,
                    "out_degree": len(method_fqns),
                    "inherits": [b for b in bases if b],
                    "imports": list(self.file_imports[mod_name].values())
                })

                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        method_fqn = f"{class_fqn}.{item.name}"
                        docs.append(self._build_symbol_doc(
                            item, method_fqn, "method", mod_name, class_fqn, rel_path, lines
                        ))

            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        var_fqn = f"{mod_name}:{target.id}"
                        docs.append({
                            "_id": var_fqn,
                            "name": target.id,
                            "type": "variable",
                            "file_path": rel_path,
                            "start_line": node.lineno,
                            "end_line": node.end_lineno,
                            "signature": ast.unparse(node).split('\n')[0],
                            "docstring": "",
                            "source_code": "\n".join(lines[node.lineno - 1 : node.end_lineno]),
                            "calls": [],
                            "is_hub": False,
                            "out_degree": 0,
                            "inherits": [],
                            "imports": []
                        })
        return docs

    def _build_module_doc(self, tree: ast.AST, mod_name: str, rel_path: str, lines: List[str]) -> Dict:
        calls = set()
        local_scope_types = {}

        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                for child in ast.walk(node):
                    if isinstance(child, ast.Assign) and isinstance(child.value, ast.Call):
                        func_name = self._extract_call_name(child.value.func)
                        if func_name:
                            resolved_type = self._resolve_name(func_name, mod_name, None)
                            if resolved_type:
                                for target in child.targets:
                                    if isinstance(target, ast.Name):
                                        local_scope_types[target.id] = resolved_type
                                calls.add(resolved_type)
                    
                    elif isinstance(child, ast.Call):
                        raw_call = self._extract_call_name(child.func)
                        if raw_call:
                            parts = raw_call.split('.')
                            if len(parts) > 1 and parts[0] in local_scope_types:
                                target_method = f"{local_scope_types[parts[0]]}.{parts[1]}"
                                if target_method in self.known_symbols:
                                    calls.add(target_method)
                                    continue
                            resolved = self._resolve_name(raw_call, mod_name, None)
                            if resolved:
                                calls.add(resolved)

                    elif isinstance(node, ast.Assign):
                        for target in node.targets:
                            if isinstance(target, ast.Name):
                                calls.add(f"{mod_name}:{target.id}")
            
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                for decorator in node.decorator_list:
                    for child in ast.walk(decorator):
                        if isinstance(child, ast.Call):
                            raw_call = self._extract_call_name(child.func)
                            if raw_call:
                                resolved = self._resolve_name(raw_call, mod_name, None)
                                if resolved:
                                    calls.add(resolved)

        unique_calls = sorted(list(calls))
        return {
            "_id": mod_name,
            "name": mod_name.split('.')[-1],
            "type": "module",
            "file_path": rel_path,
            "arguments": [],
            "return_type": "Any",
            "external_io": self._flag_external_io(tree),
            "start_line": 1,
            "end_line": len(lines),
            "signature": f"module {mod_name}",
            "docstring": ast.get_docstring(tree) or "",
            "source_code": "\n".join(lines),
            "calls": unique_calls,
            "is_hub": len(unique_calls) > 2,
            "out_degree": len(unique_calls),
            "inherits": [],
            "imports": list(self.file_imports[mod_name].values())
        }

    def _extract_variable_types(self, tree: ast.AST, mod_name: str):
        self.file_variable_types[mod_name] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                func_name = self._extract_call_name(node.value.func)
                if func_name:
                    resolved_class = self._resolve_name(func_name, mod_name, None)
                    if resolved_class in self.known_symbols:
                        for target in node.targets:
                            if isinstance(target, ast.Name):
                                self.file_variable_types[mod_name][target.id] = resolved_class
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for arg in node.args.args:
                    if arg.annotation:
                        type_name = self._extract_annotation_name(arg.annotation)
                        if type_name:
                            resolved_type = self._resolve_name(type_name, mod_name, None)
                            if resolved_type in self.known_symbols:
                                self.file_variable_types[mod_name][arg.arg] = resolved_type

    def _extract_annotation_name(self, node: ast.AST) -> Optional[str]:
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return node.attr
        elif isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Name):
            return node.slice.id
        return None

    def _build_symbol_doc(self, node: ast.AST, fqn: str, sym_type: str, mod_name: str, class_fqn: Optional[str], rel_path: str, lines: List[str]) -> Dict:
        calls = set()
        local_scope_types = {}

        for child in ast.walk(node):
            if isinstance(child, ast.Assign) and isinstance(child.value, ast.Call):
                func_name = self._extract_call_name(child.value.func)
                if func_name:
                    resolved_type = self._resolve_name(func_name, mod_name, class_fqn)
                    if resolved_type:
                        for target in child.targets:
                            if isinstance(target, ast.Name):
                                local_scope_types[target.id] = resolved_type
                        calls.add(resolved_type)
            elif isinstance(child, ast.Call):
                raw_call = self._extract_call_name(child.func)
                if raw_call:
                    parts = raw_call.split('.')
                    if len(parts) > 1 and parts[0] in local_scope_types:
                        target_method = f"{local_scope_types[parts[0]]}.{parts[1]}"
                        if target_method in self.known_symbols:
                            calls.add(target_method)
                            continue
                    resolved = self._resolve_name(raw_call, mod_name, class_fqn)
                    if resolved:
                        calls.add(resolved)

        arguments = []
        if hasattr(node, 'args') and hasattr(node.args, 'args'):
            arguments = [arg.arg for arg in node.args.args]

        return_type = "Any"
        if getattr(node, "returns", None):
            return_type = self._extract_annotation_name(node.returns) or "Any"

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args_str = ast.unparse(node.args)
            ret_str = f" -> {ast.unparse(node.returns)}" if getattr(node, "returns", None) else ""
            prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
            signature = f"{prefix} {node.name}({args_str}){ret_str}:"
        else:
            signature = f"class {getattr(node, 'name', 'Unknown')}:"

        unique_calls = sorted(list(calls))
        
        return {
            "_id": fqn,
            "name": getattr(node, "name", "unknown"),
            "type": sym_type,
            "file_path": rel_path,
            "start_line": getattr(node, "lineno", 1),
            "end_line": getattr(node, "end_lineno", 1),
            "signature": signature,
            "docstring": ast.get_docstring(node) or "",
            "source_code": "\n".join(lines[getattr(node, "lineno", 1) - 1 : getattr(node, "end_lineno", 1)]),
            "calls": unique_calls,
            "is_hub": len(unique_calls) > 2,
            "out_degree": len(unique_calls),
            "inherits": [],
            "imports": list(self.file_imports.get(mod_name, {}).values()),
            "arguments": arguments,
            "return_type": return_type,
            "external_io": self._flag_external_io(node)
        }

    def _flag_external_io(self, node: ast.AST) -> List[str]:
        io_flags = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                call_name = self._extract_call_name(child.func)
                if not call_name:
                    continue
                call_name_lower = call_name.lower()
                if any(kw in call_name_lower for kw in ['requests.', 'httpx.', 'session.get', 'session.post', 'aiohttp', 'urllib']):
                    io_flags.add("HTTP")
                if any(kw in call_name_lower for kw in ['.execute', 'cursor.', 'session.query', 'session.commit', 'mongoclient', 'db.', 'pymongo']):
                    io_flags.add("DATABASE")
        return sorted(list(io_flags))
    
    def _extract_call_name(self, func_node: ast.AST) -> Optional[str]:
        if isinstance(func_node, ast.Name):
            return func_node.id
        elif isinstance(func_node, ast.Attribute):
            value = self._extract_call_name(func_node.value)
            return f"{value}.{func_node.attr}" if value else func_node.attr
        return None

    def _resolve_name(self, raw_name: str, mod_name: str, class_fqn: Optional[str]) -> Optional[str]:
        parts = raw_name.split(".")
        head = parts[0]
        local_map = self.file_imports.get(mod_name, {})
        var_types = self.file_variable_types.get(mod_name, {})

        if head == "self" and class_fqn and len(parts) > 1:
            candidate = f"{class_fqn}.{parts[1]}"
            if candidate in self.known_symbols:
                return candidate

        if head in local_map:
            mapped = local_map[head]
            if len(parts) == 1:
                if mapped in self.known_symbols:
                    return mapped
            else:
                tail = ".".join(parts[1:])
                for sep in (".", ":"):
                    candidate = f"{mapped}{sep}{tail}"
                    if candidate in self.known_symbols:
                        return candidate

        if len(parts) > 1:
            method_name = parts[-1]
            if head in var_types:
                class_candidate = var_types[head]
                target_fqn = f"{class_candidate}.{method_name}"
                if target_fqn in self.known_symbols:
                    return target_fqn

        leaf_name = parts[-1]
        matches = [s for s in self.known_symbols if s.endswith(f".{leaf_name}") or s.endswith(f":{leaf_name}")]
        if len(matches) == 1:
            return matches[0]

        return None