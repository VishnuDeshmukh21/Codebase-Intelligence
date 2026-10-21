"""
Tests for CodeGraph MCP Server, Authentication, and Tool Registry.
"""
from __future__ import annotations

import asyncio
import os
from app.auth_mcp import MCPAuthManager
from app.main import mcp


def test_mcp_registered_tools():
    """Verify all 12 deterministic CodeGraph tools are properly registered."""
    async def _run():
        tools = await mcp.list_tools()
        tool_names = {t.name for t in tools}

        expected_tools = {
            "index_repository",
            "list_repositories",
            "get_repo_filemap",
            "search_symbols",
            "semantic_search_symbols",
            "resolve_intent_context",
            "get_symbol_details",
            "get_upstream_blast_radius",
            "get_downstream_dependencies",
            "get_dependency_subgraph_code",
            "read_file_content",
            "search_codebase_text",
            "find_files_by_name",
            "query_graph",
        }

        assert expected_tools.issubset(tool_names), f"Missing tools: {expected_tools - tool_names}"

    asyncio.run(_run())


def test_mcp_registered_prompts():
    """Verify all MCP prompts are registered for host LLM orchestration."""
    async def _run():
        prompts = await mcp.list_prompts()
        prompt_names = {p.name for p in prompts}

        assert "plan_architectural_change" in prompt_names
        assert "analyze_blast_radius" in prompt_names
        assert "codebase_overview" in prompt_names

    asyncio.run(_run())


def test_mcp_registered_resources():
    """Verify MCP resources are exposed."""
    async def _run():
        resources = await mcp.list_resources()
        uris = {r.uri for r in resources}
        assert "codegraph://repositories" in uris

    asyncio.run(_run())


def test_auth_manager_lifecycle():
    """Test API key creation, verification, and revocation."""
    manager = MCPAuthManager(None)

    # 1. Create Key
    raw_key, info = manager.generate_api_key(name="PyTest Client", tenant_id="test_suite")
    assert raw_key.startswith("cg_live_")
    assert info["name"] == "PyTest Client"
    assert info["status"] == "active"

    # 2. Verify Valid Key
    is_valid, err, doc = manager.verify_api_key(raw_key)
    assert is_valid is True
    assert err is None
    assert doc["tenant_id"] == "test_suite"

    # 3. Verify Invalid Key
    is_valid, err, _ = manager.verify_api_key("cg_live_invalidkey1234567890")
    assert is_valid is False
    assert "Invalid API key" in err

    # 4. Revoke Key
    revoked = manager.revoke_api_key(info["_id"])
    assert revoked is True

    # 5. Verify Revoked Key
    is_valid, err, _ = manager.verify_api_key(raw_key)
    assert is_valid is False
    assert "status 'revoked'" in err


def test_mcp_bearer_auth_middleware():
    """Verify that HTTP middleware automatically extracts Bearer token and validates it."""
    from starlette.testclient import TestClient
    from app.server import app
    from app.main import _check_auth

    manager = MCPAuthManager(None)
    raw_key, info = manager.generate_api_key(name="Bearer Test Client")

    @app.get("/_test_mcp_auth")
    def _test_route():
        return {"err": _check_auth()}

    client = TestClient(app)

    try:
        os.environ["CODEGRAPH_AUTH_REQUIRED"] = "true"
        os.environ.pop("CODEGRAPH_API_KEY", None)

        # 1. Without header -> Fails
        res_no_auth = client.get("/_test_mcp_auth")
        assert "[Authentication Error]" in res_no_auth.json()["err"]

        # 2. With Bearer header -> Succeeds
        res_bearer = client.get("/_test_mcp_auth", headers={"Authorization": f"Bearer {raw_key}"})
        assert res_bearer.json()["err"] is None

        # 3. With X-API-Key header -> Succeeds
        res_x_key = client.get("/_test_mcp_auth", headers={"X-API-Key": raw_key})
        assert res_x_key.json()["err"] is None

        # 4. With query parameter -> Succeeds
        res_query = client.get(f"/_test_mcp_auth?api_key={raw_key}")
        assert res_query.json()["err"] is None

        # 5. With invalid Bearer header -> Fails
        res_invalid = client.get("/_test_mcp_auth", headers={"Authorization": "Bearer cg_live_invalidkey123"})
        assert "Invalid API key" in res_invalid.json()["err"]
    finally:
        manager.revoke_api_key(info["_id"])


def test_tool_call_auth_enforcement():
    """Verify that tool calls reject unauthorized access when auth is required."""
    async def _run():
        os.environ["CODEGRAPH_AUTH_REQUIRED"] = "true"
        os.environ.pop("CODEGRAPH_API_KEY", None)
        os.environ.pop("CODEGRAPH_MASTER_KEY", None)

        # Should return authentication error
        result = await mcp.call_tool("find_files_by_name", {"pattern": "*.py", "repo_id": "test_repo"})
        response_text = result.content[0].text
        assert "[Authentication Error]" in response_text

    asyncio.run(_run())

