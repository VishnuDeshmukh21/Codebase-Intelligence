# CodeGraph MCP: Deterministic Codebase Intelligence

CodeGraph MCP is an enterprise-grade Model Context Protocol (MCP) server that transforms Python codebases into queryable, deterministic graph databases. 

Standard LLM RAG pipelines rely on arbitrary text chunking and vector similarity, leading to hallucinations and missing dependencies. CodeGraph bypasses this entirely by parsing the **Abstract Syntax Tree (AST)** to extract strict programmatic relationships (callers, callees, variables, imports) and storing them in MongoDB. AI agents can then write deterministic graph queries (`$graphLookup`) to perform 100% accurate blast-radius calculations and architectural planning.

## 🚀 Core Features

- **Deterministic AST Graphing:** Two-pass parsing resolves classes, methods, and functions to their Fully Qualified Names (FQNs) and maps structural edges, ignoring arbitrary text files.
- **Failover LLM Routing:** Built-in resilience. `plan_architectural_change` utilizes Google Gemini with an automatic, zero-downtime failover to Hugging Face (Meta Llama 3) if rate limits are hit.
- **Upstream Blast Radius:** Perform reverse-graph traversals to see exactly which files and functions will break if a core component is modified.
- **Downstream Dependency Trees:** Map the full execution flow of a specific endpoint or function across multiple micro-services/files.
- **MCP-Native:** Exposes these tools directly to AI coding agents (Claude Code, Cursor, VS Code) so they can formulate enterprise-ready PRDs and specs grounded in mathematical reality.

## 📋 Prerequisites

- **Python 3.10+**
- **MongoDB Atlas** or a local MongoDB instance.
- **API Keys** for Google Gemini (Primary) and Hugging Face (Failover).

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/codegraph-mcp.git
   cd codegraph-mcp
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *(Requires `mcp`, `pymongo`, `google-generativeai`, `huggingface_hub`, `python-dotenv`)*

3. **Configure Environment Variables:**
   Create a `.env` file in the root directory:
   ```env
   MONGODB_URI=mongodb://localhost:27017
   LLM_PROVIDER=gemini
   GEMINI_API_KEY=your_gemini_key_here
   GEMINI_MODEL=gemini-3.8-flash
   HF_TOKEN=your_huggingface_token_here
   HF_MODEL=meta-llama/Meta-Llama-3-8B-Instruct
   ```

4. **Index a Repository manually (Optional but recommended for testing):**
   ```bash
   python run_indexer.py /path/to/your/local/codebase
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
