# Task Summary
This specification details the plan and impact analysis for refactoring the `CodeGraphTools` class out of the monolithic backend (`backend/app/graph_tools.py`) into an independent, standalone Graph Intelligence service.

Currently, `CodeGraphTools` runs in-process inside the backend application, directly querying the underlying database via a shared `Database` handle. Extracting it into a separate service requires introducing a remote communication boundary (REST/gRPC), managed database access for the new service, and updating client call sites within `backend/app/main.py`.

---

# Impacted Symbols

| Symbol ID | File Path | Line Numbers | Type / Signature | Impact |
| :--- | :--- | :--- | :--- | :--- |
| `backend.app.graph_tools:CodeGraphTools` | `backend/app/graph_tools.py` | 4–254 | `class CodeGraphTools()` | **Moved / Refactored**: Extracted into separate microservice application. |
| `backend.app.graph_tools:CodeGraphTools.__init__` | `backend/app/graph_tools.py` | 5–6 | `def __init__(self, db: Database):` | **Modified**: Direct DB injection replaced by standalone service configuration. |
| `backend.app.graph_tools:CodeGraphTools._get_collection` | `backend/app/graph_tools.py` | 8–9 | `def _get_collection(self, repo_id: str):` | **Moved**: Internal service data access helper. |
| `backend.app.graph_tools:CodeGraphTools.get_downstream_dependencies` | `backend/app/graph_tools.py` | 11–136 | `def get_downstream_dependencies(self, symbol_id: str, repo_id: str, max_depth: int=2) -> Dict[str, Any]:` | **Exposed as Endpoint**: HTTP POST/GET endpoint `/api/v1/graph/downstream-dependencies`. |
| `backend.app.graph_tools:CodeGraphTools.get_dependency_subgraph_code` | `backend/app/graph_tools.py` | 182–254 | `def get_dependency_subgraph_code(self, symbol_id: str, repo_id: str, max_depth: int=3, direction: str='both') -> Dict[str, Any]:` | **Exposed as Endpoint**: HTTP POST/GET endpoint `/api/v1/graph/subgraph-code`. |
| `backend.app.main` (Module imports) | `backend/app/main.py` | 9, 34 | Line 9: `from graph_tools import CodeGraphTools`<br>Line 34: `graph_tools = CodeGraphTools(db)` | **Modified**: In-memory instantiation replaced by HTTP/RPC client stub `CodeGraphServiceClient`. |
| `backend.app.indexers.cpg_indexer:RepoGraphIndexer` | `backend/app/indexers/cpg_indexer.py` | 261, 271, 287 | Docstrings / AST pattern examples referencing `CodeGraphTools` | **Documentation Update**: Update class name comments if renamed or updated. |

---

# Dependency Analysis

1. **Database Access Coupling**:
   - `CodeGraphTools.__init__` currently accepts a direct `Database` handle.
   - Moving to a standalone service means the new service must manage its own database connection pool or data store instance, preventing shared in-memory DB handle passing.

2. **Synchronous In-Memory Calls**:
   - `backend/app/main.py` directly instantiates `CodeGraphTools(db)` on line 34 and invokes methods synchronously.
   - Replacing direct method execution with network calls introduces latency and potential network failure modes (timeouts, 5xx responses).

3. **Data Transfer & Payload Contracts**:
   - Graph traversal methods like `get_dependency_subgraph_code` return nested `Dict[str, Any]` objects representing deep code subgraphs.
   - Dict return types need strict OpenAPI/Pydantic schemas to ensure contract safety and proper JSON serialization across the service boundary.

---

# Proposed Changes

### Step 1: Define API Contracts and Pydantic Schemas
Define request and response schemas in a shared library or within the new service interface:
- `GraphTraversalRequest(symbol_id: str, repo_id: str, max_depth: int = 2, direction: str = "both")`
- `GraphTraversalResponse(nodes: List[Dict], edges: List[Dict], metadata: Dict)`

### Step 2: Create the Standalone Graph Service
- Extract `backend/app/graph_tools.py` into a new repository/directory `services/code_graph_service`.
- Wrap `CodeGraphTools` within a FastAPI / gRPC web application server.
- Configure independent database client connection logic inside the new service lifecycle.

### Step 3: Implement `CodeGraphServiceClient` Client Adapter
- Create `backend/app/clients/graph_service_client.py` in the primary backend.
- Implement methods matching the original `CodeGraphTools` signature (`get_downstream_dependencies`, `get_dependency_subgraph_code`) using an HTTP client (e.g., `httpx.AsyncClient`).

### Step 4: Refactor `backend/app/main.py`
- Remove import on line 9: `from graph_tools import CodeGraphTools`.
- Replace line 34 instantiation (`graph_tools = CodeGraphTools(db)`) with:
  ```python
  from clients.graph_service_client import CodeGraphServiceClient

  graph_tools = CodeGraphServiceClient(service_url=settings.GRAPH_SERVICE_URL)
  ```

### Step 5: Environment & Configuration Configuration
- Add `GRAPH_SERVICE_URL` and `GRAPH_SERVICE_TIMEOUT` to backend environment configuration (`backend/app/config.py`).

---

# Risk Assessment

| Risk | Impact | Likelihood | Mitigation Strategy |
| :--- | :--- | :--- | :--- |
| **Increased API Latency** | High | High | Implement response caching (e.g., Redis) for graph queries and use HTTP connection pooling/gRPC. |
| **Payload Size Serialization Overhead** | Medium | High | Compress response payloads (gzip) or use Protocol Buffers for large subgraph traversals. |
| **Service Outages / Availability** | High | Low | Implement circuit breaker patterns, client retries, and fallback graceful degradation in `CodeGraphServiceClient`. |
| **Database Connection Exhaustion** | High | Medium | Isolate database roles or connection pools for the new service to prevent contention with primary API backend queries. |

---

# Rollback Plan

1. **Feature Flag / Interface Fallback**:
   Maintain a fallback flag (`ENABLE_REMOTE_GRAPH_SERVICE=False`) in `backend/app/config.py`.
2. **Dual-Path Initialization**:
   In `backend/app/main.py`, support conditional instantiation:
   ```python
   if settings.ENABLE_REMOTE_GRAPH_SERVICE:
       graph_tools = CodeGraphServiceClient(service_url=settings.GRAPH_SERVICE_URL)
   else:
       from graph_tools import CodeGraphTools
       graph_tools = CodeGraphTools(db)
   ```
3. **Emergency Revert Strategy**:
   - Set `ENABLE_REMOTE_GRAPH_SERVICE=False` via environment variables without requiring code re-deployments if the standalone service experiences downtime or performance degradation.
   - Revert backend PR if persistent issues occur during blue/green rollout.