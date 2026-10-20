# 💬 Slack Bot & External Agent Integration Guide

This guide explains how to plug the **Codebase Intelligence MCP Server** directly into a Slack Bot, Discord Bot, or any existing conversational AI agent.

---

## 💡 How It Works

```
┌────────────────────────────────────────────────────────┐
│                   YOUR SLACK WORKSPACE                 │
│                                                        │
│  User: "@CodeBot what breaks if I change               │
│         Flask.dispatch_request?"                       │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│               YOUR SLACK BOT APPLICATION               │
│  (Powered by YOUR model: GPT-4o, Claude 3.7, etc.)     │
│                                                        │
│  1. Receives Slack mention                             │
│  2. Calls CodeGraph MCP Server over stdio or SSE       │
│  3. LLM decides to call `get_upstream_blast_radius`    │
└───────────────────────────┬────────────────────────────┘
                            │ MCP Protocol (JSON-RPC)
                            │ Header / Env: CODEGRAPH_API_KEY
                            ▼
┌────────────────────────────────────────────────────────┐
│           CODEBASE INTELLIGENCE MCP SERVER             │
│                                                        │
│  • ZERO Gemini / LLM keys required                     │
│  • 100% deterministic AST & Code Property Graph (CPG)  │
│  • MongoDB `$graphLookup` blast-radius traversal       │
│  • Returns affected symbols, callers, and files        │
└────────────────────────────────────────────────────────┘
```

---

## 🚀 Quickstart: Running the Included Slack Agent Example

The repository includes a runnable Slack agent implementation at [`backend/examples/slack_agent_example.py`](../backend/examples/slack_agent_example.py).

### Step 1: Generate an API Key for your Slack Bot
Generate a plug-and-play API key using the built-in CLI:

```bash
cd backend
python -m app.auth_cli create-key --name "Slack Bot" --tenant "engineering"
```

Output:
```text
====================================================================
  [KEY] CODEGRAPH MCP API KEY CREATED
====================================================================
  Key ID      : key_511d219491cc
  Name        : Slack Bot
  Tenant      : engineering
  Status      : active
  SECRET KEY  : cg_live_fd49fb83f68b1ae8fda15c3239b9adfa48b2d8c3
====================================================================
```

### Step 2: Run the Simulation Demo
Test the agent interaction directly from your terminal:

```bash
python examples/slack_agent_example.py --simulate "What breaks if I change dispatch_request in flask?"
```

The script will:
1. Launch `python -m app.main` as an MCP sub-process via `stdio`.
2. Authenticate using your `CODEGRAPH_API_KEY`.
3. Discover the 12 deterministic CodeGraph tools.
4. Execute `get_upstream_blast_radius`.
5. Return the structured blast-radius result.

---

## 🔌 Connecting to a Live Slack App (Slack Bolt)

### 1. Install Slack Bolt & MCP SDK
```bash
pip install slack-bolt mcp openai
```

### 2. Configure Environment Variables
```env
SLACK_BOT_TOKEN=xoxb-your-slack-bot-token
SLACK_APP_TOKEN=xapp-your-slack-app-token
OPENAI_API_KEY=your-openai-key-or-anthropic-key
CODEGRAPH_API_KEY=cg_live_your_key_here
MONGODB_URI=mongodb://localhost:27017
```

### 3. Complete Slack Bot Integration Code

```python
import os
import asyncio
from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import openai

app = AsyncApp(token=os.environ.get("SLACK_BOT_TOKEN"))
ai_client = openai.AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

SERVER_PARAMS = StdioServerParameters(
    command="python",
    args=["-m", "app.main"],
    cwd="/path/to/codegraph-mcp/backend",
    env={
        **os.environ,
        "CODEGRAPH_API_KEY": os.environ.get("CODEGRAPH_API_KEY"),
        "MONGODB_URI": os.environ.get("MONGODB_URI"),
    }
)

@app.event("app_mention")
async def handle_mention(event, say):
    user_prompt = event.get("text", "")
    await say(f"🔍 Analyzing codebase for: *{user_prompt}*...")

    async with stdio_client(SERVER_PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # Fetch tools from CodeGraph MCP
            tools_list = await session.list_tools()
            openai_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.inputSchema
                    }
                }
                for t in tools_list.tools
            ]

            # Ask your model with the CodeGraph tools
            response = await ai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a senior codebase intelligence agent. Use the tools to answer questions with precision."},
                    {"role": "user", "content": user_prompt}
                ],
                tools=openai_tools
            )

            # If the model requested a tool call, execute it on MCP server
            message = response.choices[0].message
            if message.tool_calls:
                tool_call = message.tool_calls[0]
                import json
                args = json.loads(tool_call.function.arguments)
                tool_result = await session.call_tool(tool_call.function.name, args)
                
                # Format final answer
                final_res = await ai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "Summarize the blast radius and architectural impact for a Slack channel."},
                        {"role": "user", "content": user_prompt},
                        message,
                        {"role": "tool", "tool_call_id": tool_call.id, "content": tool_result.content[0].text}
                    ]
                )
                await say(final_res.choices[0].message.content)
            else:
                await say(message.content)

if __name__ == "__main__":
    handler = AsyncSocketModeHandler(app, os.environ.get("SLACK_APP_TOKEN"))
    asyncio.run(handler.start_async())
```

---

## 🌐 Remote Integration via SSE (Server-Sent Events)

If your Slack bot runs on a different server or container, run CodeGraph MCP with HTTP/SSE transport:

```bash
python -m app.main --transport sse --port 8000 --host 0.0.0.0
```

Connect using standard MCP SSE client:
```python
from mcp.client.sse import sse_client
from mcp import ClientSession

async with sse_client("http://your-server-ip:8000/sse", headers={"Authorization": f"Bearer {CODEGRAPH_API_KEY}"}) as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()
        ...
```

