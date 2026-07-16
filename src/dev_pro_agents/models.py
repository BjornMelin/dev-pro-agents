"""Typed input and output contracts for implementation planning."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

SingleLineText = Annotated[
    str,
    Field(min_length=1, pattern=r"^[^\r\n]+$"),
]


class ContractModel(BaseModel):
    """Strict, immutable base for public workflow contracts."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class TaskBrief(ContractModel):
    """A repository-scoped engineering task and its completion contract."""

    title: SingleLineText = Field(max_length=120)
    objective: str = Field(min_length=1)
    repository_context: str = ""
    constraints: tuple[SingleLineText, ...] = ()
    acceptance_criteria: tuple[SingleLineText, ...] = Field(min_length=1)


class ImplementationStep(ContractModel):
    """One independently verifiable step in an implementation handoff."""

    title: SingleLineText = Field(max_length=120)
    outcome: SingleLineText
    files: tuple[SingleLineText, ...] = ()
    verification: tuple[SingleLineText, ...] = Field(min_length=1)


class ImplementationHandoff(ContractModel):
    """A validated, execution-ready implementation plan."""

    task_title: SingleLineText = Field(max_length=120)
    summary: SingleLineText
    assumptions: tuple[SingleLineText, ...] = ()
    steps: tuple[ImplementationStep, ...] = Field(min_length=1)
    risks: tuple[SingleLineText, ...] = ()
    manual_tasks: tuple[SingleLineText, ...] = ()
    done_criteria: tuple[SingleLineText, ...] = Field(min_length=1)

    def to_markdown(self) -> str:
        """Render a stable Markdown representation of the handoff."""
        sections = [f"# {self.task_title}", "", self.summary]
        sections.extend(_bullet_section("Assumptions", self.assumptions))
        sections.extend(["", "## Implementation"])
        for index, step in enumerate(self.steps, start=1):
            sections.extend(["", f"### {index}. {step.title}", "", step.outcome])
            sections.extend(_bullet_section("Files", step.files, level=4))
            sections.extend(_bullet_section("Verification", step.verification, level=4))
        sections.extend(_bullet_section("Risks", self.risks))
        sections.extend(_bullet_section("Manual tasks", self.manual_tasks))
        sections.extend(_bullet_section("Done criteria", self.done_criteria))
        return "\n".join(sections).strip() + "\n"


def _bullet_section(title: str, items: tuple[str, ...], *, level: int = 2) -> list[str]:
    if not items:
        return []
    heading = "#" * level
    return ["", f"{heading} {title}", "", *(f"- {item}" for item in items)]
