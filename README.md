## 🚀 Milestone 1: Structural Parsing, AST-Aware Chunking & Deterministic Flow Generation

### 1. AST Repository Ingestion & Parsing
- [ ] Implement a robust multi-pass parser to ingest repository files and build Abstract Syntax Trees (ASTs).
- [ ] Map Fully Qualified Names (FQNs) for functions, methods, classes, and other relevant symbols.
- [ ] Resolve cross-file imports and construct dependency relationships.
- [ ] Validate parsing and dependency resolution across supported file types.

### 2. AST-Aware Structural Chunking
- [ ] Replace naive character- or token-based chunking with AST-aware structural chunking.
- [ ] Segment codebases along logical boundaries, including functions, methods, and classes.
- [ ] Preserve semantic context and structural integrity within each chunk.
- [ ] Handle oversized functions, nested definitions, and edge cases gracefully.

### 3. Deterministic Flow Narrative Generation (Zero-LLM)
- [ ] Generate execution-flow narratives for functions without native docstrings using AST analysis and dependency graph traversal.
- [ ] Construct structured context sentences describing invocation sequences, database service calls, and preprocessing utilities.
- [ ] Ensure deterministic output without LLM latency or API costs.
- [ ] Introduce an extensible `generate_docstring=True` flag to support future LLM-based summarization.
- [ ] Validate generated narratives against representative codebases and complex dependency chains.

### 4. Vector Embedding & Hybrid Searchability
- [ ] Transform structurally enriched symbol documents and flow narratives into dense vector embeddings.
- [ ] Index and store embeddings for efficient retrieval.
- [ ] Implement hybrid semantic and structural search across the codebase.
- [ ] Validate retrieval quality, indexing performance, and search latency.

### 📅 Milestone Completion Log

| Milestone | Status | Completed On |
|---|---|---|
| Milestone 1 | ⏳ In Progress | Not completed |

**Completion timestamp format:** `YYYY-MM-DD HH:MM IST`

<!-- Update the relevant task to the following format when completed:
- [x] Task description — ✅ Completed: 2026-10-12 14:30 IST
-->
