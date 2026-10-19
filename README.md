# Codebase Intelligence: Deterministic AST & Graph Planning Agent

Codebase Intelligence is an enterprise-grade agentic planning service and Model Context Protocol (MCP) server that transforms Python codebases into queryable, deterministic graph databases. 

Standard LLM RAG pipelines rely on arbitrary text chunking and vector similarity, leading to hallucinations and missing dependencies. Codebase Intelligence bypasses this entirely by parsing the **Abstract Syntax Tree (AST)** to extract strict programmatic relationships (callers, callees, variables, imports, arguments, I/O) and storing them in MongoDB. LangGraph AI agents can then write deterministic graph queries (`$graphLookup`) to perform 100% accurate blast-radius calculations and architectural planning.

## 🚀 Core Features

- **Public Git Repository Indexing:** Point the service to any public Git repository URL (e.g. GitHub, GitLab, Bitbucket). The service automatically performs shallow clones (`--depth 1`) in seconds, tracks commit/branch metadata, and maps AST symbols into MongoDB collections.
- **Deterministic AST Graphing:** Two-pass parsing resolves classes, methods, and functions to their Fully Qualified Names (FQNs) and maps structural edges, ignoring arbitrary text files.
- **Official Google GenAI SDK (`google-genai`):** Modern Gemini integration powered by the official Google GenAI SDK with support for `gemini-3.8-flash`, robust `.env` handling, and zero legacy deprecation warnings.
- **Failover LLM Routing:** Built-in resilience with support for Google Gemini and Hugging Face providers.
- **LangGraph Multi-Step Reasoning Agent:** Automated planning, tool calling, graph reflection, human-in-the-loop (HITL) review gates, and comprehensive architectural PRD/spec generation.
- **Interactive Web Cockpit UI & Real-Time SSE:** Real-time event streaming (`/api/agent/stream`) and interactive web dashboard for indexing repos and chatting with the planning agent.
- **MCP-Native:** Exposes deterministic tools directly to AI coding agents (Claude Code, Cursor, VS Code) via FastMCP.

## 📋 Prerequisites

- **Python 3.10+** (tested on Python 3.12 - 3.14)
- **Git** (for cloning public repositories)
- **MongoDB Atlas** or a local MongoDB instance.
- **Google Gemini API Key** (from [Google AI Studio](https://aistudio.google.com/))

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/codegraph-mcp.git
   cd codegraph-mcp/backend
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables:**
   Copy `.env.example` to `app/.env` (or project root):
   ```env
   MONGODB_URI=mongodb+srv://<username>:<password>@<cluster>.mongodb.net/?retryWrites=true&w=majority
   LLM_PROVIDER=gemini
   GEMINI_MODEL=gemini-3.8-flash
   GEMINI_API_KEY=your_gemini_api_key_here
   REPOS_STORAGE_DIR=./storage/repos
   ```

4. **Run the API & Web Cockpit Server:**
   ```bash
   uvicorn app.server:app --host 0.0.0.0 --port 8000 --reload
   ```
   Open `http://localhost:8000` in your browser to access the Web Cockpit.

5. **Index a Public Repository via API:**
   ```bash
   curl -X POST http://localhost:8000/api/repos/index \
     -H "Content-Type: application/json" \
     -d '{"repo_url": "https://github.com/pallets/flask.git"}'
   ```


## 🔌 Connecting to VS Code / AI Agents

This server uses the open **Model Context Protocol (MCP)**, meaning it can be plugged directly into modern AI IDEs and extensions to give your AI assistant deep, graph-level understanding of your code.

### Option 1: VS Code (via Cline / Roo Code extension)
If you are using an MCP-compatible VS Code extension like Cline, add the following to your `cline_mcp.json` file (usually located in `~/.config/Code/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp.json`):

```json
{
  "mcpServers": {
    "CodeGraph": {
      "command": "python",
      "args": [
        "/absolute/path/to/codegraph-mcp/app/main.py"
      ],
      "env": {
        "MONGODB_URI": "mongodb://localhost:27017",
        "GEMINI_API_KEY": "your_api_key"
      }
    }
  }
}
```

### Option 2: Cursor IDE
1. Open Cursor Settings -> Features -> MCP.
2. Click **Add new MCP server**.
3. Set the type to `command`.
4. Name it `CodeGraph`.
5. Command: `python /absolute/path/to/codegraph-mcp/app/main.py`
6. Click Save. Your AI agent can now use the `index_repository`, `get_upstream_blast_radius`, and `plan_architectural_change` tools directly in the chat.

### Option 3: Claude Code (CLI)
Run the following command in your terminal to expose the server to Claude Code:
```bash
claude mcp add CodeGraph python /absolute/path/to/codegraph-mcp/app/main.py
```

## 🧰 Available MCP Tools

Once connected, your AI agent will automatically have access to these deterministic tools:

- `index_repository`: Compiles the target directory into the MongoDB AST graph.
- `search_symbols`: Locates entry points via lightweight IDs.
- `get_symbol_details`: Reads exact source code and direct outgoing calls for a specific ID.
- `get_upstream_blast_radius`: Identifies all parent functions/classes that depend on a target symbol.
- `get_downstream_dependencies`: Traces execution flow to retrieve all nested dependencies.
- `plan_architectural_change`: Converts natural language into a PyMongo `$graphLookup` aggregation pipeline to programmatically map architecture changes.
