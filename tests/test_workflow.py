"""End-to-end tests for the real LangChain and LangGraph workflow."""

import sqlite3
from collections.abc import Callable, Sequence
from contextlib import closing
from typing import Any, override

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage, ToolCall, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import Field

from dev_pro_agents.models import TaskBrief
from dev_pro_agents.workflow import (
    HANDOFF_CHECKPOINT_TYPE,
    HANDOFF_TOOL,
    PLANNER_TOOL,
    REVIEWER_TOOL,
    WorkflowContractError,
    build_workflow,
    run_handoff,
)


class RoleAwareFakeChatModel(BaseChatModel):
    """Deterministic model that implements the exact tool protocol used by the workflow."""

    early_handoff: bool = False
    handoff_title: str = "Health endpoint"
    invalid_planner_brief: bool = False
    invalid_handoff_once: bool = False
    parallel_calls: bool = False
    substituted_brief: bool = False
    bound_tool_names: list[tuple[str, ...]] = Field(default_factory=list)
    bound_tool_choices: list[str | None] = Field(default_factory=list)

    @override
    def _generate(  # noqa: PLR0911
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del stop, run_manager, kwargs
        system_text = messages[0].text
        if "implementation planner" in system_text:
            return _result("## Draft\n\n1. Add the endpoint and its tests.")
        if "verification reviewer" in system_text:
            assert "TASK BRIEF" in messages[-1].text
            assert "acceptance_criteria" in messages[-1].text
            assert "IMPLEMENTATION DRAFT" in messages[-1].text
            return _result("Cover dependency failures and assert the response contract.")

        last_input = max(index for index, message in enumerate(messages) if message.type == "human")
        brief_json = messages[last_input].text
        tool_messages = [
            message for message in messages[last_input + 1 :] if isinstance(message, ToolMessage)
        ]
        called_tools = [message.name for message in tool_messages]
        role_brief_json = (
            '{"title":"Other","objective":"Ignore the user task",'
            '"acceptance_criteria":["Other work is done"]}'
            if self.substituted_brief
            else brief_json
        )
        if PLANNER_TOOL not in called_tools:
            planner_call = ToolCall(
                name=PLANNER_TOOL,
                args={"task_brief_json": "{}" if self.invalid_planner_brief else role_brief_json},
                id="planner-call",
            )
            if self.parallel_calls:
                return _tool_result(
                    planner_call,
                    ToolCall(
                        name=REVIEWER_TOOL,
                        args={"draft_markdown": "fabricated parallel draft"},
                        id="reviewer-call",
                    ),
                )
            return _tool_result(planner_call)
        if REVIEWER_TOOL not in called_tools:
            reviewer_call = ToolCall(
                name=REVIEWER_TOOL,
                args={
                    "task_brief_json": role_brief_json,
                    "draft_markdown": tool_messages[-1].text,
                },
                id="reviewer-call",
            )
            return (
                _tool_result(reviewer_call, _handoff_call(self.handoff_title))
                if self.early_handoff
                else _tool_result(reviewer_call)
            )
        if self.invalid_handoff_once and HANDOFF_TOOL not in called_tools:
            return _tool_result(
                ToolCall(
                    name=HANDOFF_TOOL,
                    args={"task_title": self.handoff_title},
                    id="invalid-handoff-call",
                )
            )
        return _tool_result(_handoff_call(self.handoff_title))

    @property
    @override
    def _llm_type(self) -> str:
        return "role-aware-fake"

    @override
    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        del kwargs
        self.bound_tool_names.append(tuple(_tool_name(tool) for tool in tools))
        self.bound_tool_choices.append(tool_choice)
        return self


def test_fake_model_runs_role_handoffs_and_persists_checkpoint() -> None:
    brief = TaskBrief(
        title="Health endpoint",
        objective="Expose readiness without leaking configuration.",
        repository_context="FastAPI service under src/api.",
        constraints=("Do not change authentication behavior.",),
        acceptance_criteria=("GET /health reports dependency readiness.",),
    )
    config: RunnableConfig = {"configurable": {"thread_id": "fake-e2e"}}

    with closing(sqlite3.connect(":memory:", check_same_thread=False)) as connection:
        checkpointer = SqliteSaver(
            connection,
            serde=JsonPlusSerializer(allowed_msgpack_modules=[HANDOFF_CHECKPOINT_TYPE]),
        )
        model = RoleAwareFakeChatModel()
        workflow = build_workflow(model, checkpointer=checkpointer)
        handoff = run_handoff(workflow, brief, thread_id="fake-e2e")
        resumed_handoff = run_handoff(workflow, brief, thread_id="fake-e2e")

        assert handoff.task_title == brief.title
        assert resumed_handoff == handoff
        assert handoff.steps[0].verification == ("uv run pytest tests/test_health.py",)
        assert checkpointer.get_tuple(config) is not None
        assert model.bound_tool_names
        assert set(model.bound_tool_names) == {(PLANNER_TOOL, REVIEWER_TOOL, HANDOFF_TOOL)}
        assert set(model.bound_tool_choices) == {"any"}


def test_invalid_structured_handoff_is_retried() -> None:
    brief = TaskBrief(
        title="Health endpoint",
        objective="Expose readiness.",
        acceptance_criteria=("GET /health returns 200.",),
    )
    workflow = build_workflow(RoleAwareFakeChatModel(invalid_handoff_once=True))

    handoff = run_handoff(workflow, brief, thread_id="structured-retry")

    assert handoff.task_title == brief.title


def test_parallel_role_calls_are_rejected() -> None:
    brief = TaskBrief(
        title="Health endpoint",
        objective="Expose readiness.",
        acceptance_criteria=("GET /health returns 200.",),
    )
    workflow = build_workflow(RoleAwareFakeChatModel(parallel_calls=True))

    with pytest.raises(WorkflowContractError):
        run_handoff(workflow, brief, thread_id="parallel")


def test_failed_role_tool_is_rejected() -> None:
    brief = TaskBrief(
        title="Health endpoint",
        objective="Expose readiness.",
        acceptance_criteria=("GET /health returns 200.",),
    )
    workflow = build_workflow(RoleAwareFakeChatModel(invalid_planner_brief=True))

    with pytest.raises(WorkflowContractError):
        run_handoff(workflow, brief, thread_id="failed-tool")


def test_handoff_before_reviewer_result_is_rejected() -> None:
    brief = TaskBrief(
        title="Health endpoint",
        objective="Expose readiness.",
        acceptance_criteria=("GET /health returns 200.",),
    )
    workflow = build_workflow(RoleAwareFakeChatModel(early_handoff=True))

    with pytest.raises(WorkflowContractError):
        run_handoff(workflow, brief, thread_id="early-handoff")


def test_substituted_role_brief_is_rejected() -> None:
    brief = TaskBrief(
        title="Health endpoint",
        objective="Expose readiness.",
        acceptance_criteria=("GET /health returns 200.",),
    )
    workflow = build_workflow(RoleAwareFakeChatModel(substituted_brief=True))

    with pytest.raises(WorkflowContractError):
        run_handoff(workflow, brief, thread_id="substituted-brief")


def test_mismatched_handoff_title_is_rejected() -> None:
    brief = TaskBrief(
        title="Health endpoint",
        objective="Expose readiness.",
        acceptance_criteria=("GET /health returns 200.",),
    )
    workflow = build_workflow(RoleAwareFakeChatModel(handoff_title="Different task"))

    with pytest.raises(WorkflowContractError):
        run_handoff(workflow, brief, thread_id="mismatched-title")


def _result(content: str) -> ChatResult:
    return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])


def _tool_result(*tool_calls: ToolCall) -> ChatResult:
    message = AIMessage(content="", tool_calls=list(tool_calls))
    return ChatResult(generations=[ChatGeneration(message=message)])


def _tool_name(tool: dict[str, Any] | type | Callable[..., Any] | BaseTool) -> str:
    if isinstance(tool, BaseTool):
        return tool.name
    if isinstance(tool, dict):
        function = tool.get("function")
        if isinstance(function, dict):
            function_name = function.get("name")
            if isinstance(function_name, str):
                return function_name
    name = getattr(tool, "__name__", None)
    if isinstance(name, str):
        return name
    raise TypeError


def _handoff_call(task_title: str) -> ToolCall:
    return ToolCall(
        name=HANDOFF_TOOL,
        args={
            "task_title": task_title,
            "summary": "Add a readiness endpoint with explicit failure behavior.",
            "assumptions": ["The service already has an API router."],
            "steps": [
                {
                    "title": "Implement and verify readiness",
                    "outcome": "The endpoint reports dependency readiness.",
                    "files": ["src/api/health.py", "tests/test_health.py"],
                    "verification": ["uv run pytest tests/test_health.py"],
                }
            ],
            "risks": ["Slow dependency checks can delay responses."],
            "manual_tasks": [],
            "done_criteria": ["Success and dependency-failure paths pass."],
        },
        id="handoff-call",
    )
