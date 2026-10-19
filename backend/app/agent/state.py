# agent/state.py
"""
Agent state definition.
All fields flow through the LangGraph state machine.
"""
from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Annotated, Any, Sequence

from langchain_core.messages import BaseMessage


@dataclass
class PlanStep:
    """A single step in the agent's execution plan."""
    description: str
    tool_name: str          # Which MCP tool to call
    tool_args: dict         # Arguments for that tool
    rationale: str          # Why this step is needed
    status: str = "pending" # pending | done | skipped
    result: Any = None      # Tool output after execution


class AgentState(dict):
    """
    Typed state container for the planning agent.
    Uses dict so LangGraph can merge states across nodes.

    Fields
    ------
    messages        : Full conversation history (auto-merged by LangGraph)
    task            : Original user task string
    repo_id         : Target repository ID in MongoDB
    plan            : Ordered list of PlanSteps the planner generated
    current_step    : Index into `plan` for the current execution cursor
    observations    : Accumulated tool results (symbol data, blast radii, etc.)
    reflection      : Self-critique text produced after each tool call
    iteration       : How many planner-tool-reflect loops have run
    awaiting_human  : When True the graph pauses for human review
    human_feedback  : Text the human typed after reviewing the plan
    final_spec      : The finished PRD / migration spec produced at the end
    error           : Last unrecoverable error message, if any
    """

    messages: Annotated[Sequence[BaseMessage], operator.add]
    task: str
    repo_id: str
    plan: list[PlanStep]
    current_step: int
    observations: list[dict[str, Any]]
    reflection: str
    iteration: int
    awaiting_human: bool
    human_feedback: str
    final_spec: str
    error: str

    @classmethod
    def initial(cls, task: str, repo_id: str) -> "AgentState":
        """Factory: create a fresh state for a new planning session."""
        return cls(
            messages=[],
            task=task,
            repo_id=repo_id,
            plan=[],
            current_step=0,
            observations=[],
            reflection="",
            iteration=0,
            awaiting_human=False,
            human_feedback="",
            final_spec="",
            error="",
        )
