"""Tests for the public handoff contracts."""

import pytest
from pydantic import ValidationError

from dev_pro_agents.models import ImplementationHandoff, ImplementationStep, TaskBrief


def test_task_brief_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        TaskBrief.model_validate(
            {
                "title": "Task",
                "objective": "Objective",
                "acceptance_criteria": ["Done"],
                "unknown": True,
            }
        )


def test_task_brief_rejects_blank_acceptance_criteria() -> None:
    with pytest.raises(ValidationError):
        TaskBrief(
            title="Task",
            objective="Objective",
            acceptance_criteria=("   ",),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("task_title", "Task\n# Injected heading"),
        ("summary", "Summary\n- Injected item"),
        ("assumptions", ("Assumption\n- Injected item",)),
    ],
)
def test_handoff_rejects_multiline_top_level_values(field: str, value: object) -> None:
    payload: dict[str, object] = {
        "task_title": "Task",
        "summary": "Summary",
        "assumptions": (),
        "steps": (
            {
                "title": "Implement",
                "outcome": "Implemented",
                "verification": ("uv run pytest",),
            },
        ),
        "done_criteria": ("Done",),
    }
    payload[field] = value

    with pytest.raises(ValidationError):
        ImplementationHandoff.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("title", "Implement\n## Injected heading"),
        ("outcome", "Implemented\n- Injected item"),
        ("files", ("src/file.py\n- Injected item",)),
        ("verification", ("uv run pytest\n- Injected item",)),
    ],
)
def test_step_rejects_multiline_values(field: str, value: object) -> None:
    payload: dict[str, object] = {
        "title": "Implement",
        "outcome": "Implemented",
        "files": (),
        "verification": ("uv run pytest",),
    }
    payload[field] = value

    with pytest.raises(ValidationError):
        ImplementationStep.model_validate(payload)


def test_markdown_is_stable_and_complete() -> None:
    handoff = ImplementationHandoff(
        task_title="Health endpoint",
        summary="Add a readiness endpoint.",
        assumptions=("The API already exposes a router.",),
        steps=(
            ImplementationStep(
                title="Implement route",
                outcome="The service exposes readiness.",
                files=("src/api/health.py",),
                verification=("uv run pytest tests/test_health.py",),
            ),
        ),
        risks=("Dependency checks may be slow.",),
        manual_tasks=("Configure the deployment probe.",),
        done_criteria=("The probe reports ready.",),
    )

    assert handoff.to_markdown() == (
        "# Health endpoint\n\n"
        "Add a readiness endpoint.\n\n"
        "## Assumptions\n\n"
        "- The API already exposes a router.\n\n"
        "## Implementation\n\n"
        "### 1. Implement route\n\n"
        "The service exposes readiness.\n\n"
        "#### Files\n\n"
        "- src/api/health.py\n\n"
        "#### Verification\n\n"
        "- uv run pytest tests/test_health.py\n\n"
        "## Risks\n\n"
        "- Dependency checks may be slow.\n\n"
        "## Manual tasks\n\n"
        "- Configure the deployment probe.\n\n"
        "## Done criteria\n\n"
        "- The probe reports ready.\n"
    )
