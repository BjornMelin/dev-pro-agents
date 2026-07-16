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
