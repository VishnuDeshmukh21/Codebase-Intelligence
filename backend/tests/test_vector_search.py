"""
Tests for Local Semantic Embeddings, Hybrid Search, and Compound Intent Resolution.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from app.embedding_service import (
    build_symbol_embedding_text,
    generate_embeddings,
    generate_query_embedding,
    rank_by_cosine_similarity,
)
from app.graph_tools import CodeGraphTools
from app.agent.tools import AgentToolkit


def test_symbol_embedding_text_builder():
    """Verify high-signal semantic text formatting from AST symbol doc."""
    doc = {
        "_id": "app.config:AiModelConfigService",
        "name": "AiModelConfigService",
        "type": "class",
        "file_path": "app/config.py",
        "signature": "class AiModelConfigService:",
        "docstring": "Manages model catalog, API keys, and model parameters.",
        "calls": ["app.utils:load_yaml", "app.auth:verify_key"],
    }
    text = build_symbol_embedding_text(doc)
    assert "Class: AiModelConfigService" in text
    assert "Path: app/config.py" in text
    assert "Manages model catalog" in text
    assert "Interacts with: load_yaml, verify_key" in text


def test_embedding_generation_and_cosine_ranking():
    """Verify that semantic vectors correctly rank intent against conceptually similar symbols."""
    # Symbols: one about catalog/model config, one about DB pool, one about user auth
    symbol_docs = [
        {
            "_id": "app.config:AiModelConfigService",
            "name": "AiModelConfigService",
            "type": "class",
            "file_path": "app/config.py",
            "signature": "class AiModelConfigService:",
            "docstring": "Loads provider catalog configurations and sets parameters for models.",
            "calls": [],
        },
        {
            "_id": "app.db:DatabasePoolManager",
            "name": "DatabasePoolManager",
            "type": "class",
            "file_path": "app/db.py",
            "signature": "class DatabasePoolManager:",
            "docstring": "Maintains TCP connection pool and handles postgres transactions.",
            "calls": [],
        },
    ]

    texts = [build_symbol_embedding_text(d) for d in symbol_docs]
    embeddings = generate_embeddings(texts)
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 384
    assert len(embeddings[1]) == 384

    # Query: "catalog configuration" - neither exact word is in the class name!
    query = "catalog configuration"
    q_vec = generate_query_embedding(query)
    assert len(q_vec) == 384

    ranked = rank_by_cosine_similarity(q_vec, embeddings, top_k=2)
    assert len(ranked) == 2

    top_idx, top_score = ranked[0]
    second_idx, second_score = ranked[1]

    # AiModelConfigService must score significantly higher than DatabasePoolManager
    assert top_idx == 0
    assert top_score > second_score
    assert top_score > 0.5


def test_graph_tools_semantic_and_hybrid_search():
    """Verify semantic_search_symbols and hybrid search_symbols on mock Mongo collection."""
    mock_db = MagicMock()
    mock_col = MagicMock()
    mock_db.__getitem__.return_value = mock_col

    # Prepare dummy docs with precomputed embeddings
    doc_config = {
        "_id": "app.config:AiModelConfigService",
        "name": "AiModelConfigService",
        "type": "class",
        "file_path": "app/config.py",
        "signature": "class AiModelConfigService:",
        "start_line": 10,
        "end_line": 50,
        "docstring": "Manages model catalog and configurations.",
        "embedding": generate_query_embedding("Manages model catalog and configurations."),
    }

    doc_db = {
        "_id": "app.db:DatabasePoolManager",
        "name": "DatabasePoolManager",
        "type": "class",
        "file_path": "app/db.py",
        "signature": "class DatabasePoolManager:",
        "start_line": 1,
        "end_line": 40,
        "docstring": "Handles database pools.",
        "embedding": generate_query_embedding("Handles database pools."),
    }

    # Simulate local Mongo find
    mock_col.aggregate.side_effect = Exception("No Atlas Search")
    mock_col.find.return_value = [doc_config, doc_db]

    tools = CodeGraphTools(mock_db)

    # 1. Semantic search
    semantic_results = tools.semantic_search_symbols("catalog config", "test_repo", limit=5)
    assert len(semantic_results) > 0
    assert semantic_results[0]["_id"] == "app.config:AiModelConfigService"
    assert "embedding" not in semantic_results[0]
    assert "score" in semantic_results[0]

    # 2. Hybrid search_symbols (lexical + semantic)
    mock_col.find.side_effect = [
        [],  # Lexical regex finds 0 hits (because class name doesn't match 'catalog config')
        [doc_config, doc_db],  # Semantic candidate retrieval
    ]
    hybrid_results = tools.search_symbols("catalog config", "test_repo", limit=5)
    assert len(hybrid_results) > 0
    assert hybrid_results[0]["_id"] == "app.config:AiModelConfigService"
    assert hybrid_results[0]["match_type"] == "semantic"


def test_resolve_intent_context_compound():
    """Verify resolve_intent_context executes single-call seed lookup + graph dependency extraction."""
    mock_db = MagicMock()
    mock_col = MagicMock()
    mock_db.__getitem__.return_value = mock_col

    doc_seed = {
        "_id": "app.config:AiModelConfigService",
        "name": "AiModelConfigService",
        "type": "class",
        "file_path": "app/config.py",
        "signature": "class AiModelConfigService:",
        "start_line": 10,
        "end_line": 50,
        "docstring": "Catalog config loader",
        "embedding": generate_query_embedding("catalog config"),
        "calls": ["app.utils:load_json"],
        "source_code": "class AiModelConfigService:\n    pass",
    }

    doc_callee = {
        "_id": "app.utils:load_json",
        "name": "load_json",
        "type": "function",
        "file_path": "app/utils.py",
        "signature": "def load_json(path: str):",
        "start_line": 1,
        "end_line": 15,
        "calls": [],
        "source_code": "def load_json(path: str):\n    return {}",
    }

    # Setup mocks
    # Call 1 (semantic candidates):
    mock_col.aggregate.side_effect = [
        Exception("No Atlas"),  # Atlas check fail
        [{"callees": [doc_callee]}],  # $graphLookup aggregation
    ]
    mock_col.find.side_effect = [
        [doc_seed],  # semantic candidates
        [doc_seed, doc_callee],  # code docs fetch
    ]

    tools = CodeGraphTools(mock_db)
    result = tools.resolve_intent_context(intent="catalog config", repo_id="test_repo", max_hops=1, top_k=2)

    assert result["intent"] == "catalog config"
    assert result["total_symbols_retrieved"] == 2
    assert len(result["matched_seeds"]) == 1
    assert result["matched_seeds"][0]["symbol_id"] == "app.config:AiModelConfigService"
    
    roles = {m["symbol_id"]: m["role"] for m in result["code_manifest"]}
    assert roles["app.config:AiModelConfigService"] == "seed_entry_point"
    assert roles["app.utils:load_json"] == "direct_dependency"


def test_agent_toolkit_dispatch():
    """Verify that AgentToolkit dispatches the new tools correctly."""
    mock_gt = MagicMock()
    mock_gt.semantic_search_symbols.return_value = [{"_id": "test_sym", "score": 0.9}]
    mock_gt.resolve_intent_context.return_value = {"intent": "test", "total_symbols_retrieved": 1}

    toolkit = AgentToolkit(mock_gt)

    # 1. semantic_search_symbols dispatch
    res1 = toolkit.call("semantic_search_symbols", {"query": "auth tokens", "repo_id": "my_repo"})
    assert res1["error"] is None
    assert res1["result"][0]["_id"] == "test_sym"

    # 2. resolve_intent_context dispatch
    res2 = toolkit.call("resolve_intent_context", {"intent": "auth tokens", "repo_id": "my_repo"})
    assert res2["error"] is None
    assert res2["result"]["intent"] == "test"

