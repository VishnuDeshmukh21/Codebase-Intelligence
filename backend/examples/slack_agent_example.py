# examples/slack_agent_example.py
"""
Slack Agent Integration Example for CodeGraph MCP Server.

How to plug Codebase Intelligence into Slack (or any other agent):
------------------------------------------------------------------
1. CodeGraph MCP Server runs as a plug-and-play process (via stdio or SSE).
2. It requires ZERO LLM keys (no Gemini, no server-side keys).
3. The Slack Bot uses ITS OWN model (e.g., OpenAI GPT-4o, Anthropic Claude, or local Ollama).
4. CodeGraph MCP server deterministically indexes code, calculates blast-radius,
   and provides AST dependency maps directly to the Slack Bot's model.

Usage:
    # 1. Run simulation demo (tests full MCP connection + agent tool calling):
    python examples/slack_agent_example.py --simulate "What breaks if I change dispatch_request in flask?"

    # 2. Or run real Slack Bolt app (requires SLACK_BOT_TOKEN and SLACK_APP_TOKEN):
    python examples/slack_agent_example.py --run-slack
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Any, Dict, List

# Safe UTF-8 encoding on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Path to the MCP server
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


async def get_mcp_tools(session: ClientSession) -> List[Dict[str, Any]]:
    """Fetches tool definitions from CodeGraph MCP server."""
    tools_result = await session.list_tools()
    openai_tools = []
    for tool in tools_result.tools:
        openai_tools.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.inputSchema or {"type": "object", "properties": {}},
            },
        })
    return openai_tools


async def run_agent_turn(session: ClientSession, user_message: str, api_key: str):
    """
    Demonstrates how an agent host connects to CodeGraph MCP,
    discovers available tools, and executes them.
    """
    print(f"\n[Slack Incoming Message]: \"{user_message}\"")
    print("-" * 64)

    # 1. Discover tools from CodeGraph MCP
    tools = await session.list_tools()
    print(f"[+] Connected to CodeGraph MCP Server ({len(tools.tools)} deterministic tools available):")
    for t in tools.tools[:6]:
        print(f"    - {t.name}: {t.description[:60]}...")
    if len(tools.tools) > 6:
        print(f"    - ... and {len(tools.tools) - 6} more tools.")

    # 2. Host Agent Decision Logic:
    # In a production Slack bot, you pass `tools` to:
    #   openai.chat.completions.create(model="gpt-4o", tools=tools, messages=[...])
    #   or anthropic.messages.create(model="claude-3-7-sonnet-20250219", tools=tools, ...)
    # Here we demonstrate calling the deterministic blast radius tool:
    print("\n[AI Agent Reasoning]:")
    print("  Host model determined it needs to run blast radius analysis.")
    print("  Dispatching tool call -> get_upstream_blast_radius(symbol_id='dispatch_request', repo_id='flask')")

    # 3. Call tool on MCP Server
    tool_result = await session.call_tool(
        "get_upstream_blast_radius",
        {"symbol_id": "dispatch_request", "repo_id": "flask", "max_depth": 3},
    )

    raw_response = tool_result.content[0].text if tool_result.content else "{}"
    print(f"\n[CodeGraph MCP Deterministic Response]:")
    print(raw_response[:300] + ("\n... [truncated for display]" if len(raw_response) > 300 else ""))

    # 4. Final Slack Reply
    print("\n[Slack Bot Response Posted to Channel]:")
    print("  *Impact Analysis for `dispatch_request` in repo `flask`:*")
    print("  - Analyzed via deterministic Code Property Graph.")
    print("  - Traversed upstream callers and dependent routes.")
    print("  - Ready to provide recommended test targets and PRD spec.")
    print("=" * 64 + "\n")


async def run_client_simulation(query: str, api_key: str):
    """Runs a simulated Slack agent over stdio MCP transport."""
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "app.main", "--transport", "stdio"],
        env={
            **os.environ,
            "PYTHONPATH": BACKEND_DIR,
            "CODEGRAPH_API_KEY": api_key,
        },
    )

    print("\n" + "=" * 64)
    print("  PLUG-AND-PLAY MCP SLACK AGENT SIMULATION")
    print("=" * 64)
    print(f"  Target Server : {sys.executable} -m app.main")
    print(f"  Transport     : stdio")
    print(f"  API Key       : {api_key[:14]}...")
    print("  LLM Model     : Host's Own Model (Zero Gemini/server keys needed)")

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            await run_agent_turn(session, query, api_key)


def run_slack_bolt_listener():
    """
    Template for connecting a live Slack app (using slack_bolt) to the MCP server.
    """
    try:
        from slack_bolt.async_app import AsyncApp
        from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
    except ImportError:
        print(
            "[-] 'slack_bolt' is not installed. To run live Slack integration, install it:\n"
            "    pip install slack-bolt\n"
            "Or run the simulation mode using:\n"
            "    python examples/slack_agent_example.py --simulate \"What breaks if I change X?\"",
            file=sys.stderr,
        )
        sys.exit(1)

    slack_app = AsyncApp(token=os.environ.get("SLACK_BOT_TOKEN"))

    @slack_app.event("app_mention")
    async def handle_mention(event, say):
        text = event.get("text", "")
        # Connect to MCP server or reuse persistent session:
        # Run agent turn and post answer back
        await say(f"Received request: {text}. Querying CodeGraph MCP tools...")

    print("[+] Starting Slack Bolt Socket Mode Handler...")
    handler = AsyncSocketModeHandler(slack_app, os.environ.get("SLACK_APP_TOKEN"))
    asyncio.run(handler.start_async())


def main():
    parser = argparse.ArgumentParser(description="CodeGraph MCP Slack Agent Integration")
    parser.add_argument(
        "--simulate",
        default="What breaks if I change dispatch_request in flask?",
        help="Simulate a Slack user message asking an architectural/code query",
    )
    parser.add_argument("--run-slack", action="store_true", help="Launch live Slack Bolt SocketMode listener")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("CODEGRAPH_API_KEY", "cg_live_fd49fb83f68b1ae8fda15c3239b9adfa48b2d8c3"),
        help="CODEGRAPH_API_KEY for MCP authentication",
    )

    args = parser.parse_args()

    if args.run_slack:
        run_slack_bolt_listener()
    else:
        asyncio.run(run_client_simulation(args.simulate, args.api_key))


if __name__ == "__main__":
    main()

