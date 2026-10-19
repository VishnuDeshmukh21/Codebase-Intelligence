# eval/harness.py
"""
Evaluation harness for the Codebase Planning Agent.

Measures how accurately the agent finds the *right* symbols and files
when given a planning task with known ground truth.

Why this matters
----------------
LLM-based agents are non-deterministic. Without a structured eval loop,
you have no signal on whether a prompt change made things better or worse.
This harness gives you a repeatable, quantitative pass-rate metric.

Metrics
-------
symbol_recall   : fraction of expected symbols found in agent observations
file_recall     : fraction of expected files found in agent observations
pass_rate       : fraction of cases where symbol_recall >= threshold (default 0.75)

Usage
-----
    cases = [
        EvalCase(
            id="blast_radius_auth",
            task="What breaks if we remove the authenticate() function?",
            repo_id="my_repo",
            expected_symbols=["auth.service:authenticate", "api.routes:login_view"],
            expected_files=["auth/service.py", "api/routes.py"],
        ),
        ...
    ]
    harness = AgentEvalHarness(app=agent_app, thread_config={"configurable": {"thread_id": "eval"}})
    report = harness.run(cases)
    print(report.summary())
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

import structlog

log = structlog.get_logger()


@dataclass
class EvalCase:
    """One test case with ground truth."""
    id: str
    task: str
    repo_id: str
    expected_symbols: list[str]   # Symbols the agent MUST find
    expected_files: list[str]     # Files the agent MUST reference
    description: str = ""         # Human-readable test description


@dataclass
class CaseResult:
    """Result for a single eval case."""
    case_id: str
    task: str
    symbol_recall: float
    file_recall: float
    passed: bool
    found_symbols: list[str]
    found_files: list[str]
    duration_seconds: float
    error: str = ""


@dataclass
class EvalReport:
    """Aggregated evaluation report across all cases."""
    results: list[CaseResult]
    threshold: float
    total_cases: int = field(init=False)
    passed_cases: int = field(init=False)
    pass_rate: float = field(init=False)
    avg_symbol_recall: float = field(init=False)
    avg_file_recall: float = field(init=False)

    def __post_init__(self):
        self.total_cases = len(self.results)
        self.passed_cases = sum(1 for r in self.results if r.passed)
        self.pass_rate = self.passed_cases / self.total_cases if self.total_cases else 0.0
        self.avg_symbol_recall = (
            sum(r.symbol_recall for r in self.results) / self.total_cases
            if self.total_cases else 0.0
        )
        self.avg_file_recall = (
            sum(r.file_recall for r in self.results) / self.total_cases
            if self.total_cases else 0.0
        )

    def summary(self) -> str:
        lines = [
            "=" * 60,
            f"EVAL REPORT  ({self.passed_cases}/{self.total_cases} passed, threshold={self.threshold:.0%})",
            f"Pass rate       : {self.pass_rate:.1%}",
            f"Avg symbol recall: {self.avg_symbol_recall:.1%}",
            f"Avg file recall  : {self.avg_file_recall:.1%}",
            "=" * 60,
        ]
        for r in self.results:
            status = "✅ PASS" if r.passed else "❌ FAIL"
            lines.append(
                f"{status}  [{r.case_id}]  sym={r.symbol_recall:.0%} "
                f"file={r.file_recall:.0%}  {r.duration_seconds:.1f}s"
            )
            if r.error:
                lines.append(f"       ERROR: {r.error}")
        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "pass_rate": self.pass_rate,
            "avg_symbol_recall": self.avg_symbol_recall,
            "avg_file_recall": self.avg_file_recall,
            "threshold": self.threshold,
            "results": [
                {
                    "case_id": r.case_id,
                    "passed": r.passed,
                    "symbol_recall": r.symbol_recall,
                    "file_recall": r.file_recall,
                    "duration_seconds": r.duration_seconds,
                    "error": r.error,
                }
                for r in self.results
            ],
        }


class AgentEvalHarness:
    """
    Runs a compiled LangGraph agent against a list of EvalCases
    and returns a structured EvalReport.
    """

    def __init__(self, app, threshold: float = 0.75):
        """
        Parameters
        ----------
        app        : Compiled LangGraph application (from build_graph())
        threshold  : Minimum symbol_recall to count a case as "passed"
        """
        self.app = app
        self.threshold = threshold

    def run(self, cases: list[EvalCase]) -> EvalReport:
        """Execute all cases and return a full report."""
        results = []
        for case in cases:
            result = self._run_case(case)
            results.append(result)
            log.info(
                "eval.case_complete",
                case_id=case.id,
                passed=result.passed,
                symbol_recall=f"{result.symbol_recall:.0%}",
            )

        report = EvalReport(results=results, threshold=self.threshold)
        log.info("eval.complete", pass_rate=f"{report.pass_rate:.0%}")
        return report

    def _run_case(self, case: EvalCase) -> CaseResult:
        """Run a single eval case end-to-end (no HITL — auto-approve)."""
        import uuid
        from agent.state import AgentState

        thread_config = {"configurable": {"thread_id": f"eval-{uuid.uuid4()}"}}
        initial_state = AgentState.initial(task=case.task, repo_id=case.repo_id)

        start = time.perf_counter()
        try:
            # Stream through all nodes; collect final state
            final_state: dict[str, Any] = {}
            for chunk in self.app.stream(initial_state, thread_config):
                final_state.update(chunk.get(list(chunk.keys())[-1], {}))

            # Auto-approve if graph paused at human_review
            if final_state.get("awaiting_human"):
                self.app.update_state(
                    thread_config,
                    {"human_feedback": "approve"},
                    as_node="human_review",
                )
                for chunk in self.app.stream(None, thread_config):
                    final_state.update(chunk.get(list(chunk.keys())[-1], {}))

            duration = time.perf_counter() - start
            found_symbols, found_files = self._extract_found(final_state)

            symbol_recall = _recall(found_symbols, case.expected_symbols)
            file_recall = _recall(found_files, case.expected_files)

            return CaseResult(
                case_id=case.id,
                task=case.task,
                symbol_recall=symbol_recall,
                file_recall=file_recall,
                passed=symbol_recall >= self.threshold,
                found_symbols=list(found_symbols),
                found_files=list(found_files),
                duration_seconds=duration,
            )

        except Exception as exc:
            duration = time.perf_counter() - start
            log.error("eval.case_error", case_id=case.id, error=str(exc))
            return CaseResult(
                case_id=case.id,
                task=case.task,
                symbol_recall=0.0,
                file_recall=0.0,
                passed=False,
                found_symbols=[],
                found_files=[],
                duration_seconds=duration,
                error=str(exc),
            )

    @staticmethod
    def _extract_found(state: dict) -> tuple[set[str], set[str]]:
        """
        Mines the observations list for symbol IDs and file paths
        that the agent actually retrieved during its run.
        """
        found_symbols: set[str] = set()
        found_files: set[str] = set()

        for obs in state.get("observations", []):
            result = obs.get("result")
            if not result:
                continue

            # search_symbols returns a list of {_id, file_path, ...}
            if isinstance(result, list):
                for item in result:
                    if isinstance(item, dict):
                        if "_id" in item:
                            found_symbols.add(item["_id"])
                        if "file_path" in item:
                            found_files.add(item["file_path"])

            # blast_radius / downstream return nested dicts
            elif isinstance(result, dict):
                # Direct symbol hits
                if "_id" in result:
                    found_symbols.add(result["_id"])
                if "file_path" in result:
                    found_files.add(result["file_path"])

                # dependency_chain / downstream_callees
                for chain_key in ("dependency_chain", "downstream_callees", "code_manifest"):
                    for entry in result.get(chain_key, []):
                        if isinstance(entry, dict):
                            sym = entry.get("symbol") or entry.get("symbol_id") or entry.get("_id")
                            if sym:
                                found_symbols.add(sym)
                            fp = entry.get("file") or entry.get("file_path")
                            if fp:
                                found_files.add(fp)

                # impacted_files
                for fp in result.get("impacted_files", []):
                    found_files.add(fp)

        return found_symbols, found_files


def _recall(found: set[str], expected: list[str]) -> float:
    """Fraction of expected items that appear anywhere in found."""
    if not expected:
        return 1.0
    hits = sum(
        1 for exp in expected
        if any(exp in f or f in exp for f in found)
    )
    return hits / len(expected)

