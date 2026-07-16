"""Command-line interface for generating implementation handoffs."""

from __future__ import annotations

import os
import sys
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Never
from uuid import uuid4

import typer
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import ValidationError

from dev_pro_agents.models import TaskBrief
from dev_pro_agents.workflow import build_workflow, run_handoff

EXIT_INPUT = 2
EXIT_CONFIGURATION = 3
EXIT_WORKFLOW = 4
EXIT_OUTPUT = 5
STATE_FILE_MODE = 0o600
STATE_DIRECTORY_MODE = 0o700

app = typer.Typer(
    add_completion=False,
    help="Generate a validated implementation handoff from a typed task brief.",
    no_args_is_help=True,
)


class OutputFormat(StrEnum):
    """Supported handoff serializations."""

    MARKDOWN = "markdown"
    JSON = "json"


def _default_state_path() -> Path:
    configured_root = os.environ.get("XDG_STATE_HOME")
    candidate = Path(configured_root).expanduser() if configured_root else None
    state_root = (
        candidate
        if candidate is not None and candidate.is_absolute()
        else Path.home() / ".local" / "state"
    )
    return state_root / "dev-pro-agents" / "checkpoints.sqlite"


DEFAULT_STATE_PATH = _default_state_path()


@app.callback()
def root() -> None:
    """Generate implementation handoffs without executing repository changes."""


@app.command()
def plan(  # noqa: PLR0913
    brief_path: Annotated[
        Path,
        typer.Argument(help="TaskBrief JSON file, or '-' to read stdin."),
    ],
    output_format: Annotated[
        OutputFormat,
        typer.Option("--format", help="Output serialization."),
    ] = OutputFormat.MARKDOWN,
    model: Annotated[
        str,
        typer.Option(envvar="DEV_PRO_AGENTS_MODEL", help="LangChain provider:model identifier."),
    ] = "openai:gpt-5-mini",
    state_path: Annotated[
        Path,
        typer.Option(help="SQLite checkpoint file."),
    ] = DEFAULT_STATE_PATH,
    thread_id: Annotated[
        str | None,
        typer.Option(help="Checkpoint thread; defaults to a fresh isolated thread."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option(help="Write to this file instead of stdout."),
    ] = None,
) -> None:
    """Generate one validated implementation handoff."""
    brief = _load_brief(brief_path)
    model = model.strip()
    if not model:
        _exit("model must not be blank", EXIT_INPUT)
    provider, separator, model_name = model.partition(":")
    if not separator or not provider or not model_name or provider != provider.lower():
        _exit("model must use a lowercase provider:model identifier", EXIT_INPUT)
    if thread_id is not None:
        thread_id = thread_id.strip()
        if not thread_id:
            _exit("thread ID must not be blank", EXIT_INPUT)
    _validate_output_path(state_path, output)
    if provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        _exit("OPENAI_API_KEY is required for OpenAI models", EXIT_CONFIGURATION)

    try:
        state_path.parent.mkdir(mode=STATE_DIRECTORY_MODE, parents=True, exist_ok=True)
        state_path.touch(mode=STATE_FILE_MODE, exist_ok=True)
        if os.name == "posix":
            state_path.chmod(STATE_FILE_MODE)
        with SqliteSaver.from_conn_string(str(state_path)) as checkpointer:
            workflow = build_workflow(model, checkpointer=checkpointer)
            handoff = run_handoff(workflow, brief, thread_id=thread_id or f"run-{uuid4().hex}")
    except Exception as error:  # noqa: BLE001
        _exit(f"workflow failed: {error}", EXIT_WORKFLOW)

    rendered = (
        handoff.model_dump_json(indent=2) + "\n"
        if output_format is OutputFormat.JSON
        else handoff.to_markdown()
    )
    _write_output(output, rendered)


def _validate_output_path(state_path: Path, output: Path | None) -> None:
    if output is None:
        return
    try:
        if _paths_refer_to_same_file(state_path, output):
            _exit("output must not refer to the checkpoint file", EXIT_INPUT)
    except OSError as error:
        _exit(f"invalid output path: {error}", EXIT_INPUT)


def _write_output(output: Path | None, rendered: str) -> None:
    try:
        if output is None:
            typer.echo(rendered, nl=False)
        else:
            output.write_text(rendered, encoding="utf-8")
    except OSError as error:
        _exit(f"could not write output: {error}", EXIT_OUTPUT)


def _paths_refer_to_same_file(left: Path, right: Path) -> bool:
    left_resolved = left.expanduser().resolve(strict=False)
    right_resolved = right.expanduser().resolve(strict=False)
    if left_resolved == right_resolved:
        return True
    return (
        left_resolved.exists()
        and right_resolved.exists()
        and left_resolved.samefile(right_resolved)
    )


def _load_brief(path: Path) -> TaskBrief:
    try:
        source = sys.stdin.read() if path == Path("-") else path.read_text(encoding="utf-8")
        return TaskBrief.model_validate_json(source)
    except (OSError, UnicodeError, ValidationError) as error:
        _exit(f"invalid task brief: {error}", EXIT_INPUT)


def _exit(message: str, code: int) -> Never:
    typer.echo(message, err=True)
    raise typer.Exit(code)


def main() -> None:
    """Run the CLI application."""
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
