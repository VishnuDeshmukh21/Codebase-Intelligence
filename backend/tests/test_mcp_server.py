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

