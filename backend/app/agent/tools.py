# agent/tools.py
"""
Bridges the existing CodeGraphTools into LangGraph-compatible tool callables.

Each method is a thin, typed wrapper that:
1. Calls the underlying graph_tools logic
2. Returns a plain dict (JSON-serialisable)
3. Catches and surfaces errors so the agent can self-correct
"""
from __future__ import annotations

import traceback
from typing import Any

from graph_tools import CodeGraphTools


class AgentToolkit:
    """
    A stateful toolbox the agent nodes call directly.
    Built on top of CodeGraphTools (already wired to MongoDB).
    """

    TOOL_NAMES = [
        "search_symbols",
        "get_symbol_details",
        "get_upstream_blast_radius",
        "get_downstream_dependencies",
        "get_dependency_subgraph_code",
        "read_file_content",
        "search_codebase_text",
    ]

    def __init__(self, graph_tools: CodeGraphTools):
        self._gt = graph_tools

    def call(self, tool_name: str, tool_args: dict[str, Any]) -> dict[str, Any]:
        """
        Dispatch a tool call by name and return a structured result dict.
        Always returns {\"tool\": str, \"args\": dict, \"result\": Any, \"error\": str|None}.
        """
        if tool_name not in self.TOOL_NAMES:
            return self._error(tool_name, tool_args, f"Unknown tool '{tool_name}'.")

        handler = getattr(self, f"_{tool_name}", None)
        if handler is None:
            return self._error(tool_name, tool_args, f"Handler for '{tool_name}' not implemented.")

        try:
            result = handler(**tool_args)
            return {"tool": tool_name, "args": tool_args, "result": result, "error": None}
        except Exception as exc:
            return self._error(tool_name, tool_args, str(exc), traceback.format_exc())

    # ------------------------------------------------------------------
    # Private handlers — one per MCP tool
    # ------------------------------------------------------------------

    def _search_symbols(self, query: str, repo_id: str, limit: int = 10) -> list[dict]:
        return self._gt.search_symbols(query=query, repo_id=repo_id, limit=limit)

    def _get_symbol_details(self, symbol_id: str, repo_id: str) -> dict:
        return self._gt.get_symbol_details(symbol_id=symbol_id, repo_id=repo_id)

    def _get_upstream_blast_radius(
        self, symbol_id: str, repo_id: str, max_depth: int = 3
    ) -> dict:
        return self._gt.get_upstream_blast_radius(
            symbol_id=symbol_id, repo_id=repo_id, max_depth=max_depth
        )

    def _get_downstream_dependencies(
        self, symbol_id: str, repo_id: str, max_depth: int = 2
    ) -> dict:
        return self._gt.get_downstream_dependencies(
            symbol_id=symbol_id, repo_id=repo_id, max_depth=max_depth
        )

    def _get_dependency_subgraph_code(
        self,
        symbol_id: str,
        repo_id: str,
        max_depth: int = 2,
        direction: str = "both",
    ) -> dict:
        return self._gt.get_dependency_subgraph_code(
            symbol_id=symbol_id,
            repo_id=repo_id,
            max_depth=max_depth,
            direction=direction,
        )

    def _read_file_content(self, file_path: str, repo_id: str) -> dict:
        col = self._gt._get_collection(repo_id)
        doc = col.find_one({"file_path": file_path, "type": "module"})
        if doc:
            return {
                "file_path": file_path,
                "source_code": doc.get("source_code", ""),
                "calls": doc.get("calls", []),
            }
        symbols = list(col.find({"file_path": file_path}).sort("start_line", 1))
        if not symbols:
            raise ValueError(f"File '{file_path}' not found in the index.")
        return {
            "file_path": file_path,
            "source_code": "\n\n".join(
                f"# Symbol: {s['_id']}\n{s.get('source_code', '')}"
                for s in symbols
                if s.get("source_code")
            ),
            "calls": list({c for s in symbols for c in s.get("calls", [])}),
        }

    def _search_codebase_text(self, pattern: str, repo_id: str) -> str:
        return self._gt.search_codebase_text(pattern=pattern, repo_id=repo_id)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _error(tool_name: str, args: dict, message: str, tb: str = "") -> dict[str, Any]:
        return {
            "tool": tool_name,
            "args": args,
            "result": None,
            "error": message,
            "traceback": tb,
        }

