# agent/graph.py
"""
Builds and compiles the LangGraph state machine for the Codebase Planning Agent.

Architecture
------------
                    ┌─────────────┐
      task ──────►  │   planner   │ ◄── human_feedback (revise loop)
                    └──────┬──────┘
                           │ route_after_planner
                    ┌──────▼──────┐
                    │ tool_caller │
                    └──────┬──────┘
                           │ route_after_tool
                    ┌──────▼──────┐      ┌──────────────┐
                    │  reflector  │ ───► │ human_review │ (HITL gate)
                    └─────────────┘      └──────┬───────┘
                                                │ route_after_human
                                         ┌──────▼──────┐
                                         │ spec_writer │
                                         └──────┬──────┘
                                                │
                                               END

The graph is compiled with interrupt_before=["human_review"] so execution
pauses for human input before writing the final spec.
"""
from __future__ import annotations

import functools
from typing import Any

import structlog
from langgraph.graph import END, StateGraph

from .nodes import (
    human_review_node,
    planner_node,
    reflector_node,
    route_after_human,
    route_after_planner,
    route_after_tool,
    spec_writer_node,
    tool_caller_node,
)
from .state import AgentState
from .tools import AgentToolkit

log = structlog.get_logger()


def build_graph(llm_client, toolkit: AgentToolkit, checkpointer=None):
    """
    Constructs and compiles the planning agent graph.

    Parameters
    ----------
    llm_client  : Any BaseLLMProvider instance (Gemini, HuggingFace, etc.)
    toolkit     : AgentToolkit wrapping your CodeGraphTools
    checkpointer: Optional LangGraph checkpointer (e.g. MongoDBSaver).
                  If None, the graph runs without persistence (useful in tests).

    Returns
    -------
    Compiled LangGraph application ready for .invoke() / .stream()
    """
    builder = StateGraph(AgentState)

    # ── Bind runtime dependencies into each node via functools.partial ──────
    _planner = functools.partial(planner_node, llm_client=llm_client)
    _tool_caller = functools.partial(tool_caller_node, toolkit=toolkit)
    _reflector = functools.partial(reflector_node, llm_client=llm_client)
    _spec_writer = functools.partial(spec_writer_node, llm_client=llm_client)

    # ── Register nodes ───────────────────────────────────────────────────────
    builder.add_node("planner", _planner)
    builder.add_node("tool_caller", _tool_caller)
    builder.add_node("reflector", _reflector)
    builder.add_node("human_review", human_review_node)
    builder.add_node("spec_writer", _spec_writer)

    # ── Entry point ──────────────────────────────────────────────────────────
    builder.set_entry_point("planner")

    # ── Conditional edges ────────────────────────────────────────────────────
    builder.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "execute": "tool_caller",
            "error": END,
        },
    )

    builder.add_conditional_edges(
        "tool_caller",
        route_after_tool,
        {
            "reflect": "reflector",
            "human_review": "human_review",
        },
    )

    # After reflecting: always go back to tool_caller to run next step
    builder.add_edge("reflector", "tool_caller")

    builder.add_conditional_edges(
        "human_review",
        route_after_human,
        {
            "write_spec": "spec_writer",
            "replan": "planner",
            "stop": END,
        },
    )

    builder.add_edge("spec_writer", END)

    # ── Compile ──────────────────────────────────────────────────────────────
    compile_kwargs: dict[str, Any] = {}
    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer
        # Pause BEFORE human_review so the caller can inject feedback
        compile_kwargs["interrupt_before"] = ["human_review"]

    app = builder.compile(**compile_kwargs)
    log.info("agent_graph.compiled", has_checkpointer=checkpointer is not None)
    return app


def create_agent(mongo_uri: str, llm_client, graph_tools) -> tuple:
    """
    Convenience factory that wires everything together.

    Returns (app, thread_config) where thread_config is a dict you pass
    to every .invoke() / .stream() call to maintain conversation continuity.

    Usage
    -----
    app, thread = create_agent(mongo_uri, llm_client, graph_tools)
    state = AgentState.initial(task="Migrate auth to OAuth2", repo_id="my_repo")
    for chunk in app.stream(state, thread):
        print(chunk)
    """
    try:
        from memory.mongo_checkpointer import build_mongo_checkpointer
    except ImportError:
        from app.memory.mongo_checkpointer import build_mongo_checkpointer

    toolkit = AgentToolkit(graph_tools)
    checkpointer = build_mongo_checkpointer(mongo_uri)
    app = build_graph(llm_client=llm_client, toolkit=toolkit, checkpointer=checkpointer)

    import uuid
    thread_id = str(uuid.uuid4())
    thread_config = {"configurable": {"thread_id": thread_id}}

    log.info("agent.created", thread_id=thread_id)
    return app, thread_config

