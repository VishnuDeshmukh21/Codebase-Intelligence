#!/usr/bin/env python
# run_agent.py
"""
CLI entry point for the Codebase Planning Agent.

Usage
-----
    # Basic run (auto-approves human review for non-interactive use)
    python run_agent.py --task "What breaks if I remove authenticate()?" --repo-id my_repo

    # Interactive HITL mode (pauses for your review before generating spec)
    python run_agent.py --task "Migrate auth to OAuth2" --repo-id my_repo --interactive

    # Run the eval harness
    python run_agent.py --eval --repo-id my_repo
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import structlog
from dotenv import load_dotenv
from pymongo import MongoClient

# Add app/ to the path so imports work from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app"))

from app.agent.graph import build_graph
from app.agent.state import AgentState
from app.agent.tools import AgentToolkit
from app.eval.harness import AgentEvalHarness, EvalCase
from app.graph_tools import CodeGraphTools
from app.llm_service.factory import LLMFactory
from app.memory.mongo_checkpointer import build_mongo_checkpointer

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "app", ".env"))

log = structlog.get_logger()


def build_app(mongo_uri: str):
    """Wire up the agent graph with all its dependencies."""
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=10000)
    db = client["prode_engine"]
    graph_tools = CodeGraphTools(db)
    toolkit = AgentToolkit(graph_tools)
    llm_client = LLMFactory.get_client()
    checkpointer = build_mongo_checkpointer(mongo_uri, client=client)
    app = build_graph(llm_client=llm_client, toolkit=toolkit, checkpointer=checkpointer)
    return app, llm_client


def run_interactive(app, task: str, repo_id: str):
    """Run with human-in-the-loop review gate."""
    import uuid
    thread_config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    initial_state = AgentState.initial(task=task, repo_id=repo_id)

    print(f"\n{'='*60}")
    print(f"  CODEBASE PLANNING AGENT")
    print(f"  Task   : {task}")
    print(f"  Repo   : {repo_id}")
    print(f"{'='*60}\n")

    # Stream until the graph pauses at human_review
    for chunk in app.stream(initial_state, thread_config, stream_mode="updates"):
        node_name = list(chunk.keys())[0]
        node_state = chunk[node_name]
        _print_chunk(node_name, node_state)

    # Check if we're paused
    current = app.get_state(thread_config)
    if current.next and "human_review" in current.next:
        print("\n" + "="*60)
        print("  HUMAN REVIEW REQUIRED")
        print("="*60)
        for msg in current.values.get("messages", [])[-1:]:
            print(msg.content)
        print("\nRespond with: 'approve' | 'revise: <feedback>' | 'stop'")
        feedback = input("> ").strip()

        app.update_state(
            thread_config,
            {"human_feedback": feedback, "awaiting_human": False},
            as_node="human_review",
        )

        for chunk in app.stream(None, thread_config, stream_mode="updates"):
            node_name = list(chunk.keys())[0]
            node_state = chunk[node_name]
            _print_chunk(node_name, node_state)

    # Print final spec
    final = app.get_state(thread_config)
    spec = final.values.get("final_spec", "")
    if spec:
        print("\n" + "="*60)
        print("  FINAL SPEC")
        print("="*60)
        print(spec)

        output_file = f"spec_{repo_id}.md"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(spec)
        print(f"\n✅ Spec saved to {output_file}")


def run_auto(app, task: str, repo_id: str):
    """Run without HITL — auto-approves at the review gate (useful for eval/CI)."""
    import uuid
    thread_config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    initial_state = AgentState.initial(task=task, repo_id=repo_id)

    for chunk in app.stream(initial_state, thread_config, stream_mode="updates"):
        node_name = list(chunk.keys())[0]
        node_state = chunk[node_name]
        _print_chunk(node_name, node_state)

    current = app.get_state(thread_config)
    if current.next and "human_review" in current.next:
        app.update_state(
            thread_config,
            {"human_feedback": "approve", "awaiting_human": False},
            as_node="human_review",
        )
        for chunk in app.stream(None, thread_config, stream_mode="updates"):
            node_name = list(chunk.keys())[0]
            _print_chunk(node_name, chunk[node_name])

    final = app.get_state(thread_config)
    return final.values.get("final_spec", "")


def run_eval(app, repo_id: str):
    """Run the evaluation harness with built-in test cases."""
    # Extend this list with real cases from YOUR indexed repos
    cases = [
        EvalCase(
            id="blast_radius_basic",
            task="What will break if I change the CodeGraphTools class?",
            repo_id=repo_id,
            expected_symbols=[f"graph_tools:CodeGraphTools"],
            expected_files=["app/graph_tools.py"],
            description="Blast radius on the central graph tools class",
        ),
        EvalCase(
            id="downstream_deps",
            task="What does the index_repository function depend on?",
            repo_id=repo_id,
            expected_symbols=[f"indexers.cpg_indexer:RepoGraphIndexer"],
            expected_files=["app/indexers/cpg_indexer.py"],
            description="Downstream dependency tracing",
        ),
        EvalCase(
            id="search_entry_points",
            task="Find all MCP tool functions in the codebase",
            repo_id=repo_id,
            expected_symbols=["main:search_symbols", "main:index_repository"],
            expected_files=["app/main.py"],
            description="Entry point discovery",
        ),
    ]

    harness = AgentEvalHarness(app=app, threshold=0.75)
    report = harness.run(cases)
    print(report.summary())

    # Save report to JSON
    report_path = f"eval_report_{repo_id}.json"
    with open(report_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)
    print(f"\n📊 Full report saved to {report_path}")
    return report


def _print_chunk(node_name: str, state: dict):
    """Pretty-print each node's output as the agent streams."""
    icons = {
        "planner": "🧠",
        "tool_caller": "🔧",
        "reflector": "🪞",
        "human_review": "👤",
        "spec_writer": "📝",
    }
    icon = icons.get(node_name, "▶")
    print(f"\n{icon}  [{node_name.upper()}]")

    if node_name == "planner" and state.get("plan"):
        for i, step in enumerate(state["plan"]):
            print(f"   Step {i+1}: {step.description} (tool: {step.tool_name})")

    elif node_name == "tool_caller" and state.get("observations"):
        last = state["observations"][-1]
        err = last.get("error")
        if err:
            print(f"   ❌ {last['tool']} failed: {err}")
        else:
            result_preview = str(last.get("result", ""))[:200]
            print(f"   ✅ {last['tool']} → {result_preview}...")

    elif node_name == "reflector" and state.get("reflection"):
        print(f"   {state['reflection'][:300]}")

    elif node_name == "spec_writer" and state.get("final_spec"):
        print("   Spec generated ✅")


def main():
    parser = argparse.ArgumentParser(description="Codebase Planning Agent CLI")
    parser.add_argument("--task", type=str, help="Planning task to execute")
    parser.add_argument("--repo-id", type=str, required=True, help="MongoDB repo ID to query")
    parser.add_argument("--interactive", action="store_true", help="Enable HITL review gate")
    parser.add_argument("--eval", action="store_true", help="Run evaluation harness")
    args = parser.parse_args()

    mongo_uri = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
    app, _ = build_app(mongo_uri)

    if args.eval:
        run_eval(app, repo_id=args.repo_id)
    elif args.task:
        if args.interactive:
            run_interactive(app, task=args.task, repo_id=args.repo_id)
        else:
            spec = run_auto(app, task=args.task, repo_id=args.repo_id)
            if spec:
                print("\n" + "="*60)
                print(spec)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

