# agent/nodes.py
"""
LangGraph node functions.

Each function takes the current AgentState, performs its job,
and returns a dict of state fields to update (partial updates are fine).

Nodes:
  planner_node       - LLM generates / revises the plan
  tool_caller_node   - Executes the next step's tool call
  reflector_node     - LLM self-critiques the tool result
  human_review_node  - Formats the plan for human approval (HITL gate)
  spec_writer_node   - LLM produces the final Markdown PRD/spec
"""
from __future__ import annotations

import json
import re
import textwrap
from dataclasses import asdict
from typing import Any

import structlog
from langchain_core.messages import AIMessage, HumanMessage

from .prompts import (
    HUMAN_REVIEW_PROMPT,
    PLANNER_SYSTEM,
    REFLECTOR_SYSTEM,
    SPEC_WRITER_SYSTEM,
)
from .state import AgentState, PlanStep
from .tools import AgentToolkit

log = structlog.get_logger()

MAX_ITERATIONS = 12  # Safety ceiling to prevent infinite loops


# ---------------------------------------------------------------------------
# Planner Node
# ---------------------------------------------------------------------------

def planner_node(state: AgentState, llm_client) -> dict[str, Any]:
    """
    Calls the LLM to produce or revise a plan.

    Builds a rich user prompt that includes:
    - The original task
    - Any prior observations already collected
    - Human feedback (if this is a revision loop)
    - The reflection from the last tool call
    """
    iteration = state.get("iteration", 0)
    if iteration >= MAX_ITERATIONS:
        log.warning("planner_node.max_iterations_reached", iteration=iteration)
        return {"error": f"Exceeded max iterations ({MAX_ITERATIONS}). Terminating."}

    observations = state.get("observations", [])
    reflection = state.get("reflection", "")
    human_feedback = state.get("human_feedback", "")
    task = state["task"]
    repo_id = state["repo_id"]

    # Build the contextual user message
    parts = [f"Task: {task}", f"Repository ID: {repo_id}"]

    if observations:
        obs_summary = _format_observations(observations[-3:])  # Last 3 to avoid context overflow
        parts.append(f"\nObservations so far:\n{obs_summary}")

    if reflection:
        parts.append(f"\nReflection on last action:\n{reflection}")

    if human_feedback:
        parts.append(f"\nHuman reviewer feedback (MUST incorporate):\n{human_feedback}")

    user_prompt = "\n".join(parts)

    log.info("planner_node.calling_llm", iteration=iteration, task=task[:80])
    raw = llm_client.generate(
        prompt=user_prompt,
        system_prompt=PLANNER_SYSTEM,
        temperature=0.2,
    )

    # Strip accidental markdown fences
    clean = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
    clean = re.sub(r"\s*```$", "", clean.strip(), flags=re.MULTILINE)

    try:
        parsed = json.loads(clean)
    except json.JSONDecodeError as exc:
        log.error("planner_node.json_parse_failed", error=str(exc), raw=raw[:300])
        return {"error": f"Planner produced invalid JSON: {exc}. Raw: {raw[:200]}"}

    raw_steps = parsed.get("plan", [])
    new_plan = [
        PlanStep(
            description=s["description"],
            tool_name=s["tool_name"],
            tool_args=s["tool_args"],
            rationale=s.get("rationale", ""),
        )
        for s in raw_steps
    ]

    log.info("planner_node.plan_generated", steps=len(new_plan), reasoning=parsed.get("reasoning", "")[:120])

    return {
        "plan": new_plan,
        "current_step": 0,
        "iteration": iteration + 1,
        "human_feedback": "",  # Clear feedback after incorporating
        "messages": [
            HumanMessage(content=user_prompt),
            AIMessage(content=raw),
        ],
    }


# ---------------------------------------------------------------------------
# Tool Caller Node
# ---------------------------------------------------------------------------

def tool_caller_node(state: AgentState, toolkit: AgentToolkit) -> dict[str, Any]:
    """
    Executes the tool specified in plan[current_step].
    Marks the step as done and appends the observation.
    """
    plan = state.get("plan", [])
    current_step = state.get("current_step", 0)

    if current_step >= len(plan):
        log.info("tool_caller_node.all_steps_done", total_steps=len(plan))
        return {}  # Signal to router that plan is exhausted

    step = plan[current_step]
    log.info(
        "tool_caller_node.executing",
        step=current_step,
        tool=step.tool_name,
        args=step.tool_args,
    )

    observation = toolkit.call(tool_name=step.tool_name, tool_args=step.tool_args)

    if observation.get("error"):
        log.warning(
            "tool_caller_node.tool_error",
            tool=step.tool_name,
            error=observation["error"],
        )

    # Mark step as done (PlanStep is a dataclass, so create updated copy)
    updated_step = PlanStep(
        description=step.description,
        tool_name=step.tool_name,
        tool_args=step.tool_args,
        rationale=step.rationale,
        status="done",
        result=observation.get("result"),
    )
    updated_plan = list(plan)
    updated_plan[current_step] = updated_step

    return {
        "plan": updated_plan,
        "current_step": current_step + 1,
        "observations": state.get("observations", []) + [observation],
    }


# ---------------------------------------------------------------------------
# Reflector Node
# ---------------------------------------------------------------------------

def reflector_node(state: AgentState, llm_client) -> dict[str, Any]:
    """
    Produces a short self-critique of the latest tool call result.
    This reflection is fed back into the planner on the next iteration.
    """
    observations = state.get("observations", [])
    if not observations:
        return {"reflection": "No observations yet."}

    latest = observations[-1]
    task = state["task"]
    plan = state.get("plan", [])

    plan_summary = "\n".join(
        f"  Step {i+1}: [{s.status}] {s.description}" for i, s in enumerate(plan)
    )

    user_prompt = textwrap.dedent(f"""
        Task: {task}

        Current Plan:
        {plan_summary}

        Latest tool call: {latest.get('tool')} with args {latest.get('args')}
        Result:
        {json.dumps(latest.get('result'), indent=2, default=str)[:2000]}

        Error (if any): {latest.get('error', 'None')}
    """).strip()

    log.info("reflector_node.calling_llm")
    reflection = llm_client.generate(
        prompt=user_prompt,
        system_prompt=REFLECTOR_SYSTEM,
        temperature=0.3,
    )

    log.info("reflector_node.done", reflection_preview=reflection[:120])
    return {"reflection": reflection}


# ---------------------------------------------------------------------------
# Human Review Node  (HITL gate)
# ---------------------------------------------------------------------------

def human_review_node(state: AgentState) -> dict[str, Any]:
    """
    Formats the current plan and reflection for human review.
    The graph is interrupted HERE (interrupt_before=["human_review"]),
    so this node just prepares the display message.

    After the human calls graph.update_state() with their response,
    execution resumes from this node's output.
    """
    plan = state.get("plan", [])
    plan_summary = "\n".join(
        f"  Step {i+1}: [{s.status}] {s.description} (tool: {s.tool_name})"
        for i, s in enumerate(plan)
    )

    prompt_text = HUMAN_REVIEW_PROMPT.format(
        task=state.get("task", ""),
        plan_summary=plan_summary,
        reflection=state.get("reflection", "No reflection yet."),
    )

    log.info("human_review_node.awaiting_human_input")
    return {
        "awaiting_human": True,
        "messages": [AIMessage(content=prompt_text)],
    }


# ---------------------------------------------------------------------------
# Spec Writer Node
# ---------------------------------------------------------------------------

def spec_writer_node(state: AgentState, llm_client) -> dict[str, Any]:
    """
    Synthesises all observations and reflections into a final Markdown spec.
    This is the terminal node — its output is the agent's deliverable.
    """
    observations = state.get("observations", [])
    obs_text = _format_observations(observations)

    plan = state.get("plan", [])
    plan_summary = "\n".join(
        f"  Step {i+1}: [{s.status}] {s.description}\n    Rationale: {s.rationale}"
        for i, s in enumerate(plan)
    )

    user_prompt = textwrap.dedent(f"""
        Task: {state['task']}
        Repository: {state['repo_id']}

        Execution Plan:
        {plan_summary}

        All Graph Analysis Observations:
        {obs_text}

        Final Reflection:
        {state.get('reflection', '')}
    """).strip()

    log.info("spec_writer_node.generating_spec")
    spec = llm_client.generate(
        prompt=user_prompt,
        system_prompt=SPEC_WRITER_SYSTEM,
        temperature=0.4,
    )

    log.info("spec_writer_node.done", spec_length=len(spec))
    return {
        "final_spec": spec,
        "awaiting_human": False,
        "messages": [AIMessage(content=spec)],
    }


# ---------------------------------------------------------------------------
# Router helpers (called by add_conditional_edges in graph.py)
# ---------------------------------------------------------------------------

def route_after_planner(state: AgentState) -> str:
    """Decides what happens after the planner produces a new plan."""
    if state.get("error"):
        return "error"
    plan = state.get("plan", [])
    if not plan:
        return "error"
    return "execute"


def route_after_tool(state: AgentState) -> str:
    """Decides what happens after a tool executes."""
    plan = state.get("plan", [])
    current_step = state.get("current_step", 0)
    iteration = state.get("iteration", 0)

    # All plan steps have been executed
    if current_step >= len(plan):
        return "human_review"

    # Still have steps to run — reflect then continue
    return "reflect"


def route_after_human(state: AgentState) -> str:
    """Routes based on what the human typed after reviewing."""
    feedback = state.get("human_feedback", "").strip().lower()
    if feedback == "approve" or feedback == "":
        return "write_spec"
    if feedback == "stop":
        return "stop"
    # Any "revise: ..." feedback goes back to planner
    return "replan"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _format_observations(observations: list[dict]) -> str:
    """Compact string representation of tool call observations."""
    parts = []
    for i, obs in enumerate(observations):
        tool = obs.get("tool", "unknown")
        error = obs.get("error")
        result = obs.get("result")

        if error:
            parts.append(f"[{i+1}] {tool} → ERROR: {error}")
        else:
            result_str = json.dumps(result, indent=2, default=str)
            # Cap each observation to avoid context overflow
            if len(result_str) > 1500:
                result_str = result_str[:1500] + "\n... [truncated]"
            parts.append(f"[{i+1}] {tool}:\n{result_str}")

    return "\n\n".join(parts) if parts else "No observations yet."

