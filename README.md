# Codebase Intelligence: Plug-and-Play Model Context Protocol (MCP) Server

[![MCP Native](https://img.shields.io/badge/MCP-Protocol_2.0-blue.svg)](https://modelcontextprotocol.io/)
[![Zero Server LLM Keys](https://img.shields.io/badge/LLM_Keys-Bring_Your_Own_Model-brightgreen.svg)]()
[![Deterministic AST](https://img.shields.io/badge/Graph-Deterministic_AST_CPG-purple.svg)]()
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)]()

**Codebase Intelligence** is a **plug-and-play Model Context Protocol (MCP) server** that turns any Python codebase into a deterministic Code Property Graph (CPG) in MongoDB. 

Instead of running an isolated web service with a separate UI and separate LLM keys, Codebase Intelligence plugs directly into your existing AI applications, IDEs, and bots (Cursor, Claude Desktop, VS Code Cline, Slack bots, custom agents). 

> **Bring Your Own Model:** The MCP server requires **ZERO server-side LLM keys** (no Gemini key, no OpenAI key). It provides 100% deterministic AST parsing, `$graphLookup` blast-radius calculations, and dependency subgraphs. Your host agent (GPT-4o, Claude 3.7 Sonnet, Ollama, etc.) acts as the brain that reasons and calls the tools.

---

## ⚡ Key Highlights

- 🔌 **Plug & Play Everywhere:** Plug directly into Cursor, Claude Desktop, VS Code (Cline/Roo-Code), Claude Code, or any custom agent (e.g. a company Slack Bot).
- 🧠 **Zero Server-Side LLM Dependency:** No Gemini API key or proprietary LLM lock-in. When a client plugs in this server, it utilizes the host's models.
- 🔑 **Plug-and-Play Authentication:** Secure API key validation (`CODEGRAPH_API_KEY`) with a built-in CLI manager (`python -m app.auth_cli create-key`). No web UI or registration portal needed.
- 🌐 **Dual Transports:** Supports standard `stdio` (for local processes and IDEs) and `sse` / `streamable-http` (for network/cloud bot integrations).
- 🎯 **100% Deterministic AST Traversals:** Bypasses vector RAG chunking and hallucinated imports. Uses MongoDB `$graphLookup` for programmatic caller/callee graphs and blast-radius tracing.
- 📦 **Git & Local Repo Indexing:** Point it at public Git URLs (`https://github.com/owner/repo`) or local directories.

---

## 🛠️ Quick Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/codegraph-mcp.git
   cd codegraph-mcp/backend
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment:**
   Set `MONGODB_URI` in `.env` (Atlas or local MongoDB):
   ```env
   MONGODB_URI=mongodb://localhost:27017
   # Optional: set master key or require auth
   CODEGRAPH_AUTH_REQUIRED=true
   ```

---

## 🔑 Plug-and-Play Authentication (Zero UI)

Create, list, and revoke API keys with zero UI via the built-in CLI:

```bash
# Generate an API key for your client or agent
python -m app.auth_cli create-key --name "Slack Bot" --tenant "engineering"

# List active keys
python -m app.auth_cli list-keys

# Verify or revoke a key
python -m app.auth_cli verify-key cg_live_xxxxxxxxxxxxxxxxxxxx
python -m app.auth_cli revoke-key key_511d219491cc
```

---

## 🔌 Plugging into Your Tools

### 1. Cursor IDE
1. Open **Cursor Settings** -> **Features** -> **MCP**.
2. Click **Add new MCP server**.
3. Choose type `command`.
4. Command:
   ```bash
   python -m app.main
   ```
5. In environment variables, add:
   ```json
   {
     "PYTHONPATH": "/absolute/path/to/codegraph-mcp/backend",
     "MONGODB_URI": "mongodb://localhost:27017",
     "CODEGRAPH_API_KEY": "cg_live_your_key_here"
   }
   ```

### 2. Claude Desktop (`claude_desktop_config.json`)
Add to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "CodeGraph": {
      "command": "python",
      "args": ["-m", "app.main"],
      "cwd": "/absolute/path/to/codegraph-mcp/backend",
      "env": {
        "PYTHONPATH": "/absolute/path/to/codegraph-mcp/backend",
        "MONGODB_URI": "mongodb://localhost:27017",
        "CODEGRAPH_API_KEY": "cg_live_your_key_here"
      }
    }
  }
}
```

### 3. VS Code (Cline / Roo-Code Extension)
Add to your `cline_mcp.json`:

```json
{
  "mcpServers": {
    "CodeGraph": {
      "command": "python",
      "args": ["-m", "app.main"],
      "cwd": "/absolute/path/to/codegraph-mcp/backend",
      "env": {
        "PYTHONPATH": "/absolute/path/to/codegraph-mcp/backend",
        "MONGODB_URI": "mongodb://localhost:27017",
        "CODEGRAPH_API_KEY": "cg_live_your_key_here"
      }
    }
  }
}
```

### 4. Claude Code (CLI)
```bash
claude mcp add CodeGraph python -m app.main
```

### 5. Slack Bot / Custom AI Agents
See the full guide in [docs/SLACK_INTEGRATION.md](docs/SLACK_INTEGRATION.md).

Run the included simulation to see an agent connect and query CodeGraph over MCP:
```bash
python examples/slack_agent_example.py --simulate "What breaks if I change dispatch_request in flask?"
```

---

## 🧰 Available MCP Capabilities

### Deterministic Tools
| Tool Name | Description |
|---|---|
| `index_repository` | Shallow clones remote Git repo or scans local path into MongoDB AST symbols. |
| `list_repositories` | Lists all indexed repositories, commit hashes, branches, and symbol counts. |
| `get_repo_filemap` | Generates a complete directory tree + symbol manifest for high-level architecture tasks. |
| `search_symbols` | Exact / regex symbol discovery for classes, functions, and methods. |
| `get_symbol_details` | Retrieves full Python source code, line numbers, docstrings, and direct calls. |
| `get_upstream_blast_radius` | Reverse graph traversal (`$graphLookup`) determining all breaking changes and impacted files. |
| `get_downstream_dependencies` | Traces nested execution flow and retrieves all downstream callees with code. |
| `get_dependency_subgraph_code` | Consolidates complete source code for a symbol and its connected neighborhood. |
| `read_file_content` | Reads file content directly from the AST database index without touching disk. |
| `search_codebase_text` | Fast regex search across all indexed code with line numbers and enclosing symbols. |
| `find_files_by_name` | Glob/regex file path search within the database index. |
| `query_graph` | Safe PyMongo aggregation runner for custom graph traversal queries. |

### MCP Prompts (Workflow Blueprints for Host Models)
- `plan_architectural_change(task, repo_id)`: Blueprints multi-step architectural planning for the host agent.
- `analyze_blast_radius(symbol_id, repo_id)`: Guides the host agent through impact analysis and test planning.
- `codebase_overview(repo_id)`: Guides the host agent to synthesize a complete architectural report.

### MCP Resources
- `codegraph://repositories`: Real-time JSON manifest of indexed repositories.
- `codegraph://repo/{repo_id}/summary`: Repository symbol count and file statistics.

---

## 🧪 Testing

Run the test suite to verify the MCP server, auth manager, and tools:

```bash
cd backend
$env:PYTHONPATH="."
python -m pytest tests/test_mcp_server.py
```
