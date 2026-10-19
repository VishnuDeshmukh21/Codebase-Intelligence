import ast
import os
from pathlib import Path
from typing import Dict, List, Set, Optional
from pymongo import MongoClient, UpdateOne
import re

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
    def __init__(self, repo_path: str, repo_id: str, mongo_uri: str = "mongodb://localhost:27017", db_name: str = "prode_engine"):
        self.repo_path = Path(repo_path).resolve()
        self.repo_id = repo_id
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.symbols_col = self.db[f"{self.repo_id}_symbols"]
        
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

    def index_repository(self):
        py_files = [
            p for p in self.repo_path.rglob("*.py")
            if not any(part.startswith(".") or part in ("venv", "__pycache__", "node_modules", "tests") for part in p.parts)
        ]

        print(f"[Pass 1] Discovering symbols across {len(py_files)} files...")
        parsed_trees: Dict[Path, ast.AST] = {}
        for file_path in py_files:
            try:
                source = file_path.read_text(encoding="utf-8")
                tree = ast.parse(source, filename=str(file_path))
                parsed_trees[file_path] = (tree, source)
                self._pass_one_discover(file_path, tree)
            except Exception as e:
                print(f"Skipping {file_path}: {e}")

        print(f"[Pass 2] Extracting resolved call edges for {len(self.known_symbols)} symbols...")
        bulk_ops = []
        for file_path, (tree, source) in parsed_trees.items():
            docs = self._pass_two_extract(file_path, tree, source)
            for doc in docs:
                bulk_ops.append(UpdateOne({"_id": doc["_id"]}, {"$set": doc}, upsert=True))

        if bulk_ops:
            self.symbols_col.bulk_write(bulk_ops)
            self.symbols_col.create_index("calls")
            self.symbols_col.create_index("file_path")
            self.symbols_col.create_index("name")
            print(f"Indexed {len(bulk_ops)} symbols into collection {self.repo_id}_symbols.")

    def _pass_one_discover(self, file_path: Path, tree: ast.AST):
        mod_name = self._file_to_module(file_path)
        self.file_imports[mod_name] = {}
        
        # Register the module itself as a known symbol
        self.known_symbols.add(mod_name)

        for node in ast.iter_child_nodes(tree):
            # Track imports: `import x.y as z` or `from x import y as z`
            if isinstance(node, ast.Import):
                for alias in node.names:
                    local_name = alias.asname or alias.name
                    self.file_imports[mod_name][local_name] = alias.name
            elif isinstance(node, ast.ImportFrom):
                base_mod = node.module or ""
                # Handle relative imports (e.g., `from . import utils` or `from ..models import User`)
                if node.level > 0:
                    parent_parts = mod_name.split(".")[:-node.level]
                    base_mod = ".".join(parent_parts + ([base_mod] if base_mod else []))
                for alias in node.names:
                    local_name = alias.asname or alias.name
                    self.file_imports[mod_name][local_name] = f"{base_mod}:{alias.name}"

            # Track top-level functions
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                fqn = f"{mod_name}:{node.name}"
                self.known_symbols.add(fqn)
                self.file_imports[mod_name][node.name] = fqn

            # Track top-level variables/constants (e.g., mcp = FastMCP())
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        fqn = f"{mod_name}:{target.id}"
                        self.known_symbols.add(fqn)
                        self.file_imports[mod_name][target.id] = fqn

            # Track classes and their methods
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

        # Extract variable assignments and type hints for context scoping
        self._extract_variable_types(tree, mod_name)

        # 1. Create a Root Module Node (captures top-level execution like main.py)
        docs.append(self._build_module_doc(tree, mod_name, rel_path, lines))

        # 2. Extract Functions and Classes
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                docs.append(self._build_symbol_doc(
                    node, f"{mod_name}:{node.name}", "function", mod_name, None, rel_path, lines
                ))
            elif isinstance(node, ast.ClassDef):
                class_fqn = f"{mod_name}:{node.name}"
                bases = [self._resolve_name(ast.unparse(b), mod_name, None) for b in node.bases]
                
                # Capture all method FQNs to link the class to its children structurally
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

                # Method documents
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
                            "signature": ast.unparse(node).split('\n')[0], # Store the assignment line
                            "docstring": "",
                            "source_code": "\n".join(lines[node.lineno - 1 : node.end_lineno]),
                            "calls": [], # Variables don't have structural children
                            "is_hub": False,
                            "out_degree": 0,
                            "inherits": [],
                            "imports": []
                        })
        return docs

    def _build_module_doc(self, tree: ast.AST, mod_name: str, rel_path: str, lines: List[str]) -> Dict:
        """Parses the top-level execution scope of a file to capture module-level dependencies."""
        calls = set()
        local_scope_types = {}

        for node in tree.body:
            # Skip inner contents of classes and functions, we only want global scope
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
            
            # Special case: Catch decorators on top-level functions (e.g., @mcp.tool())
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
        """Scans the AST to build a dictionary mapping local variable names to their Class FQNs."""
        self.file_variable_types[mod_name] = {}
        
        for node in ast.walk(tree):
            # 1. Track Instantiations: e.g., graph_tools = CodeGraphTools(db)
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                func_name = self._extract_call_name(node.value.func)
                if func_name:
                    resolved_class = self._resolve_name(func_name, mod_name, None)
                    if resolved_class in self.known_symbols:
                        for target in node.targets:
                            if isinstance(target, ast.Name):
                                self.file_variable_types[mod_name][target.id] = resolved_class

            # 2. Track Argument Type Hints: e.g., def execute(tools: CodeGraphTools):
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for arg in node.args.args:
                    if arg.annotation:
                        type_name = self._extract_annotation_name(arg.annotation)
                        if type_name:
                            resolved_type = self._resolve_name(type_name, mod_name, None)
                            if resolved_type in self.known_symbols:
                                self.file_variable_types[mod_name][arg.arg] = resolved_type

    def _extract_annotation_name(self, node: ast.AST) -> Optional[str]:
        """Safely extracts the string name from type annotation nodes."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return node.attr
        # Basic support for generic wrappers like Optional[CodeGraphTools]
        elif isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Name):
            return node.slice.id
        return None

    def _build_symbol_doc(self, node: ast.AST, fqn: str, sym_type: str, mod_name: str, class_fqn: Optional[str], rel_path: str, lines: List[str]) -> Dict:
        calls = set()
        local_scope_types = {}

        for child in ast.walk(node):
            # Track local assignments (e.g., my_var = SomeClass())
            if isinstance(child, ast.Assign) and isinstance(child.value, ast.Call):
                func_name = self._extract_call_name(child.value.func)
                if func_name:
                    resolved_type = self._resolve_name(func_name, mod_name, class_fqn)
                    if resolved_type:
                        for target in child.targets:
                            if isinstance(target, ast.Name):
                                local_scope_types[target.id] = resolved_type
                        calls.add(resolved_type) # Add the initialization as a dependency
            
            # Track method calls on those local variables (e.g., my_var.do_something())
            elif isinstance(child, ast.Call):
                raw_call = self._extract_call_name(child.func)
                if raw_call:
                    parts = raw_call.split('.')
                    # If the base variable was initialized locally, resolve its method
                    if len(parts) > 1 and parts[0] in local_scope_types:
                        target_method = f"{local_scope_types[parts[0]]}.{parts[1]}"
                        if target_method in self.known_symbols:
                            calls.add(target_method)
                            continue
                            
                    resolved = self._resolve_name(raw_call, mod_name, class_fqn)
                    if resolved:
                        calls.add(resolved)

        args_str = ast.unparse(node.args)
        ret_str = f" -> {ast.unparse(node.returns)}" if getattr(node, "returns", None) else ""
        prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
        signature = f"{prefix} {node.name}({args_str}){ret_str}:"

        unique_calls = sorted(list(calls))
        return {
            "_id": fqn,
            "name": node.name,
            "type": sym_type,
            "file_path": rel_path,
            "start_line": node.lineno,
            "end_line": node.end_lineno,
            "signature": signature,
            "docstring": ast.get_docstring(node) or "",
            "source_code": "\n".join(lines[node.lineno - 1 : node.end_lineno]),
            "calls": unique_calls,
            "is_hub": len(unique_calls) > 2,
            "out_degree": len(unique_calls),
            "inherits": [],
            "imports": list(self.file_imports[mod_name].values())
        }

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

        # Case 1: `self.method()` inside a class
        if head == "self" and class_fqn and len(parts) > 1:
            candidate = f"{class_fqn}.{parts[1]}"
            if candidate in self.known_symbols:
                return candidate

        # Case 2: Direct match in current file's import/symbol table
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

        # Case 3: Instance Variable tracking via Type Hints / Assignments
        if len(parts) > 1:
            method_name = parts[-1]
            if head in var_types:
                class_candidate = var_types[head]
                target_fqn = f"{class_candidate}.{method_name}"
                if target_fqn in self.known_symbols:
                    return target_fqn

        # Case 4: Suffix fallback (Handles import path mismatches AND unknown types)
        leaf_name = parts[-1]
        matches = [s for s in self.known_symbols if s.endswith(f".{leaf_name}") or s.endswith(f":{leaf_name}")]
        if len(matches) == 1:
            return matches[0]

        return None

    def _flag_external_io(self, node: ast.AST) -> List[str]:
        """Scan a subtree for obvious HTTP or database calls."""
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