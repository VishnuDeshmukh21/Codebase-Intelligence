#!/usr/bin/env python
# test_client.py
"""
Example Python client demonstrating how a frontend or client application
interacts with the Codebase Intelligence Agent server via HTTP and Server-Sent Events (SSE).

Usage:
    # 1. Start the server in one terminal:
    #    python -m uvicorn app.server:app --port 8000 --reload
    #
    # 2. Run this test client in another terminal:
    #    python test_client.py --task "What will break if we refactor CodeGraphTools?" --repo-id "codebase-intelligence_symbols"
"""
from __future__ import annotations

import argparse
import json
import sys
import httpx


SERVER_URL = "http://localhost:8000"

ICONS = {
    "planner": "🧠",
    "tool_caller": "🔧",
    "reflector": "🪞",
    "human_review": "👤",
    "spec_writer": "📝",
    "session_start": "🚀",
    "awaiting_human": "⏸️",
    "completed": "✅",
    "error": "❌",
}


def parse_sse_line(line: str) -> tuple[str | None, str | None]:
    if line.startswith("event: "):
        return "event", line[7:].strip()
    elif line.startswith("data: "):
        return "data", line[6:].strip()
    return None, None


def stream_response(response: httpx.Response):
    """Parses SSE stream and prints events in real-time."""
    current_event = None
    thread_id = None
    awaiting_human_data = None
    final_spec = None

    for raw_line in response.iter_lines():
        if not raw_line:
            continue
        line = raw_line.strip()
        kind, value = parse_sse_line(line)

        if kind == "event":
            current_event = value
        elif kind == "data" and current_event:
            try:
                data = json.loads(value)
            except json.JSONDecodeError:
                data = {"raw": value}

            icon = ICONS.get(current_event, "▶")

            if current_event == "session_start":
                thread_id = data.get("thread_id")
                print(f"\n{icon} Session started! Thread ID: {thread_id}")
                print(f"   Task   : {data.get('task')}")
                print(f"   Repo   : {data.get('repo_id')}\n")

            elif current_event == "node_update":
                node = data.get("node")
                node_icon = ICONS.get(node, "▶")
                print(f"\n{node_icon} [{node.upper()}]")

                if node == "planner" and data.get("plan"):
                    for i, step in enumerate(data["plan"]):
                        print(f"   Step {i+1}: {step.get('description')} (tool: {step.get('tool_name')})")

                elif node == "tool_caller":
                    err = data.get("error")
                    if err:
                        print(f"   ❌ {data.get('tool')} failed: {err}")
                    else:
                        preview = data.get("result_preview", "")
                        print(f"   ✅ {data.get('tool')} -> {preview[:160]}...")

                elif node == "reflector":
                    print(f"   Critique: {data.get('reflection', '')[:250]}...")

                elif node == "spec_writer":
                    print(f"   Spec generated!")

            elif current_event == "awaiting_human":
                awaiting_human_data = data
                thread_id = data.get("thread_id")
                print(f"\n{icon} [HUMAN REVIEW REQUIRED]")
                print(f"   {data.get('prompt', '')}")
                if data.get("reasoning"):
                    print(f"   Strategy: {data.get('reasoning')[:160]}...")
                if data.get("plan"):
                    print(f"   Plan: {len(data['plan'])} steps queued/executed")
                if data.get("reflection"):
                    print(f"   Reflection: {data.get('reflection')[:160]}...")
                print("\n   Enter 'approve', 'revise: <feedback>', or 'stop'")

            elif current_event == "completed":
                print(f"\n{icon} Agent finished execution!")
                if data.get("final_spec"):
                    final_spec = data["final_spec"]
                    print("\n" + "=" * 60)
                    print("  FINAL SPECIFICATION")
                    print("=" * 60)
                    print(final_spec[:800] + ("..." if len(final_spec) > 800 else ""))

            elif current_event == "error":
                print(f"\n❌ Server error: {data.get('error')}")

            current_event = None

    return thread_id, awaiting_human_data, final_spec


def run_chat(task: str, repo_id: str, interactive: bool = True):
    print(f"Connecting to Agent Server at {SERVER_URL}...")

    # Check health
    with httpx.Client(timeout=10.0) as client:
        try:
            r = client.get(f"{SERVER_URL}/health")
            if r.status_code != 200:
                print(f"Server health check failed: {r.text}")
                return
        except Exception as e:
            print(f"Failed to connect to server at {SERVER_URL}: {e}")
            print("Make sure to start the server first via: uvicorn app.server:app --port 8000")
            return

    # Start stream
    stream_url = f"{SERVER_URL}/api/agent/stream"
    payload = {
        "task": task,
        "repo_id": repo_id,
        "interactive": interactive,
    }

    with httpx.Client(timeout=300.0) as client:
        with client.stream("POST", stream_url, json=payload) as response:
            thread_id, awaiting_human_data, final_spec = stream_response(response)

        # Handle Human Review if paused
        while awaiting_human_data and thread_id:
            user_input = input("\nYour decision > ").strip()
            if not user_input:
                user_input = "approve"

            feedback_url = f"{SERVER_URL}/api/agent/feedback"
            feedback_payload = {
                "thread_id": thread_id,
                "feedback": user_input,
            }

            with client.stream("POST", feedback_url, json=feedback_payload) as feedback_resp:
                thread_id, awaiting_human_data, final_spec = stream_response(feedback_resp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test client for Codebase Intelligence Agent server")
    parser.add_argument("--task", type=str, default="What will break if we refactor CodeGraphTools into a separate service?")
    parser.add_argument("--repo-id", type=str, default="codebase-intelligence_symbols")
    parser.add_argument("--non-interactive", action="store_true", help="Auto-approve human review gate")
    args = parser.parse_args()

    run_chat(task=args.task, repo_id=args.repo_id, interactive=not args.non_interactive)

