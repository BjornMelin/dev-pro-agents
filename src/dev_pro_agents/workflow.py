"""LangChain v1 role handoffs for read-only implementation planning."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, ClassVar, Protocol, cast

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolCall, ToolMessage
from langchain_core.tools import tool

from dev_pro_agents.models import ImplementationHandoff, TaskBrief

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig
    from langgraph.checkpoint.base import BaseCheckpointSaver

PLANNER_TOOL = "implementation_planner"
REVIEWER_TOOL = "verification_reviewer"
HANDOFF_TOOL = ImplementationHandoff.__name__
BLANK_THREAD_ERROR = "thread_id must not be blank"

PLANNER_PROMPT = """You are an implementation planner. Produce a concise Markdown draft that
maps the supplied task brief to ordered, repository-specific changes. Name likely files,
preserve stated constraints, and attach a concrete verification command or observation to each
step. Do not execute commands, access external systems, or claim that work has already happened.
"""

REVIEWER_PROMPT = """You are a verification reviewer. Inspect the supplied implementation draft
for omitted acceptance criteria, unsafe assumptions, unverifiable claims, and missing failure-path
tests. Return concise corrections only. Do not execute commands or access external systems.
"""

COORDINATOR_PROMPT = f"""You produce a validated implementation handoff from a typed task brief.
You MUST call `{PLANNER_TOOL}` first with the complete task brief JSON. Then call
`{REVIEWER_TOOL}` with that same task brief JSON and the planner's full draft. Incorporate the
review, then call the `{HANDOFF_TOOL}` structured-output tool exactly once. Do not call any other
tools, and do not claim the implementation has run. Put missing credentials, deployments, or human
approvals in manual_tasks.
"""

ModelLike = str | BaseChatModel


class Workflow(Protocol):
    """Minimal compiled-agent surface used by the public runner."""

    def invoke(
        self,
        state_input: Mapping[str, object],
        config: RunnableConfig | None = None,
    ) -> Mapping[str, object]:
        """Run the workflow and return its final state."""
        ...


class WorkflowContractError(RuntimeError):
    """Raised when an agent skips a required role handoff or emits invalid output."""

    MISSING_HANDOFF: ClassVar[str] = "workflow did not complete the required role handoff"
    INVALID_HANDOFF: ClassVar[str] = "workflow returned an invalid implementation handoff"
    MISSING_MESSAGES: ClassVar[str] = "workflow state did not contain messages"
    INVALID_MESSAGE: ClassVar[str] = "workflow state contained an invalid message"
    NO_ROLE_MESSAGES: ClassVar[str] = "role agent returned no messages"
    NO_ROLE_TEXT: ClassVar[str] = "role agent returned no text"


def build_workflow(
    model: ModelLike,
    *,
    checkpointer: BaseCheckpointSaver[str] | None = None,
) -> Workflow:
    """Build the canonical planner-reviewer workflow with injected runtime dependencies."""
    planner = create_agent(model=model, tools=(), system_prompt=PLANNER_PROMPT)
    reviewer = create_agent(model=model, tools=(), system_prompt=REVIEWER_PROMPT)

    @tool(PLANNER_TOOL)
    def implementation_planner(task_brief_json: str) -> str:
        """Draft an ordered implementation plan from the complete TaskBrief JSON."""
        brief = TaskBrief.model_validate_json(task_brief_json)
        result = planner.invoke({"messages": [_brief_message(brief)]})
        return _last_message_text(cast("Mapping[str, object]", result))

    @tool(REVIEWER_TOOL)
    def verification_reviewer(task_brief_json: str, draft_markdown: str) -> str:
        """Review a planner draft for safety, completeness, and verifiability."""
        brief = TaskBrief.model_validate_json(task_brief_json)
        review_input = (
            f"TASK BRIEF\n{brief.model_dump_json(indent=2)}\n\n"
            f"IMPLEMENTATION DRAFT\n{draft_markdown}"
        )
        result = reviewer.invoke({"messages": [{"role": "user", "content": review_input}]})
        return _last_message_text(cast("Mapping[str, object]", result))

    graph = create_agent(
        model=model,
        tools=(implementation_planner, verification_reviewer),
        system_prompt=COORDINATOR_PROMPT,
        response_format=ToolStrategy(ImplementationHandoff.model_json_schema()),
        checkpointer=checkpointer,
    )
    return cast("Workflow", graph)


def run_handoff(workflow: Workflow, brief: TaskBrief, *, thread_id: str) -> ImplementationHandoff:
    """Run a task brief and enforce the planner-to-reviewer handoff contract."""
    if not thread_id.strip():
        raise ValueError(BLANK_THREAD_ERROR)
    result = workflow.invoke(
        {"messages": [_brief_message(brief)]},
        {"configurable": {"thread_id": thread_id}},
    )
    messages = _messages(result)
    if not _has_causal_role_handoff(messages, brief):
        raise WorkflowContractError(WorkflowContractError.MISSING_HANDOFF)

    structured = result.get("structured_response")
    if isinstance(structured, ImplementationHandoff):
        handoff = structured
    else:
        try:
            handoff = ImplementationHandoff.model_validate(structured)
        except ValueError as error:
            raise WorkflowContractError(WorkflowContractError.INVALID_HANDOFF) from error
    if handoff.task_title != brief.title:
        raise WorkflowContractError(WorkflowContractError.INVALID_HANDOFF)
    return handoff


def _brief_message(brief: TaskBrief) -> dict[str, str]:
    return {"role": "user", "content": brief.model_dump_json(indent=2)}


def _messages(result: Mapping[str, object]) -> Sequence[BaseMessage]:
    raw_messages = result.get("messages")
    if not isinstance(raw_messages, Sequence) or isinstance(raw_messages, str):
        raise WorkflowContractError(WorkflowContractError.MISSING_MESSAGES)
    if not all(isinstance(message, BaseMessage) for message in raw_messages):
        raise WorkflowContractError(WorkflowContractError.INVALID_MESSAGE)
    return cast("Sequence[BaseMessage]", raw_messages)


def _current_messages(messages: Sequence[BaseMessage]) -> Sequence[BaseMessage]:
    last_input = max(
        (index for index, message in enumerate(messages) if isinstance(message, HumanMessage)),
        default=-1,
    )
    return messages[last_input + 1 :]


def _has_causal_role_handoff(messages: Sequence[BaseMessage], brief: TaskBrief) -> bool:
    current = _current_messages(messages)
    planner_calls = _tool_calls(current, PLANNER_TOOL)
    reviewer_calls = _tool_calls(current, REVIEWER_TOOL)
    handoff_calls = _tool_calls(current, HANDOFF_TOOL)
    if len(planner_calls) != 1 or len(reviewer_calls) != 1 or len(handoff_calls) != 1:
        return False

    planner_index, planner_call = planner_calls[0]
    reviewer_index, reviewer_call = reviewer_calls[0]
    handoff_index, handoff_call = handoff_calls[0]
    planner_call_id = planner_call.get("id")
    reviewer_call_id = reviewer_call.get("id")
    handoff_call_id = handoff_call.get("id")
    if planner_call_id is None or reviewer_call_id is None or handoff_call_id is None:
        return False
    planner_results = _tool_results(current, PLANNER_TOOL, planner_call_id)
    reviewer_results = _tool_results(current, REVIEWER_TOOL, reviewer_call_id)
    handoff_results = _tool_results(current, HANDOFF_TOOL, handoff_call_id)
    if len(planner_results) != 1 or len(reviewer_results) != 1 or len(handoff_results) != 1:
        return False

    planner_result_index, planner_result = planner_results[0]
    reviewer_result_index, reviewer_result = reviewer_results[0]
    handoff_result_index, handoff_result = handoff_results[0]
    reviewed_draft = reviewer_call["args"].get("draft_markdown")
    planned_brief = _validated_brief(planner_call["args"].get("task_brief_json"))
    reviewed_brief = _validated_brief(reviewer_call["args"].get("task_brief_json"))
    return (
        planner_index
        < planner_result_index
        < reviewer_index
        < reviewer_result_index
        < handoff_index
        < handoff_result_index
        and reviewed_draft == planner_result.text
        and planned_brief == brief
        and reviewed_brief == brief
        and planner_result.status == "success"
        and reviewer_result.status == "success"
        and handoff_result.status == "success"
    )


def _validated_brief(value: object) -> TaskBrief | None:
    if not isinstance(value, str):
        return None
    try:
        return TaskBrief.model_validate_json(value)
    except ValueError:
        return None


def _tool_calls(messages: Sequence[BaseMessage], name: str) -> list[tuple[int, ToolCall]]:
    calls: list[tuple[int, ToolCall]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, AIMessage):
            continue
        calls.extend((index, call) for call in message.tool_calls if call["name"] == name)
    return calls


def _tool_results(
    messages: Sequence[BaseMessage],
    name: str,
    call_id: str,
) -> list[tuple[int, ToolMessage]]:
    return [
        (index, message)
        for index, message in enumerate(messages)
        if isinstance(message, ToolMessage)
        and message.name == name
        and message.tool_call_id == call_id
    ]


def _last_message_text(result: Mapping[str, object]) -> str:
    messages = _messages(result)
    if not messages:
        raise WorkflowContractError(WorkflowContractError.NO_ROLE_MESSAGES)
    text = messages[-1].text
    if not text.strip():
        raise WorkflowContractError(WorkflowContractError.NO_ROLE_TEXT)
    return text
