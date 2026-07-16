"""Tests for deterministic command-line failure behavior."""

import json
import os
from pathlib import Path
from typing import Never

import pytest
import typer
from typer.testing import CliRunner

from dev_pro_agents import cli
from dev_pro_agents.cli import (
    EXIT_CONFIGURATION,
    EXIT_INPUT,
    EXIT_OUTPUT,
    EXIT_WORKFLOW,
    STATE_FILE_MODE,
    _default_state_path,
    app,
)
from dev_pro_agents.models import ImplementationHandoff, ImplementationStep, TaskBrief

runner = CliRunner()


class ProviderFailureError(Exception):
    """Synthetic non-runtime provider failure for CLI boundary tests."""


def test_invalid_brief_exits_two(tmp_path: Path) -> None:
    brief_path = _write_brief(tmp_path, {})

    result = runner.invoke(app, ["plan", str(brief_path)])

    assert result.exit_code == EXIT_INPUT
    assert "invalid task brief" in result.stderr


def test_non_utf8_brief_exits_two(tmp_path: Path) -> None:
    brief_path = tmp_path / "brief.json"
    brief_path.write_bytes(b"\xff")

    result = runner.invoke(app, ["plan", str(brief_path)])

    assert result.exit_code == EXIT_INPUT
    assert "invalid task brief" in result.stderr


def test_missing_openai_key_exits_three(tmp_path: Path) -> None:
    brief_path = _write_brief(tmp_path, _valid_brief())

    result = runner.invoke(
        app,
        ["plan", str(brief_path)],
        env={"OPENAI_API_KEY": "", "DEV_PRO_AGENTS_MODEL": ""},
    )

    assert result.exit_code == EXIT_CONFIGURATION
    assert result.stderr == "OPENAI_API_KEY is required for OpenAI models\n"


def test_blank_thread_id_exits_two(tmp_path: Path) -> None:
    brief_path = _write_brief(tmp_path, _valid_brief())

    result = runner.invoke(
        app,
        ["plan", str(brief_path), "--model", "fake:model", "--thread-id", "   "],
    )

    assert result.exit_code == EXIT_INPUT
    assert result.stderr == "thread ID must not be blank\n"


def test_providerless_model_exits_two(tmp_path: Path) -> None:
    brief_path = _write_brief(tmp_path, _valid_brief())

    result = runner.invoke(app, ["plan", str(brief_path), "--model", "gpt-5-mini"])

    assert result.exit_code == EXIT_INPUT
    assert result.stderr == "model must use a lowercase provider:model identifier\n"


def test_relative_xdg_state_home_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", "relative-state")

    assert _default_state_path() == Path.home() / ".local/state/dev-pro-agents/checkpoints.sqlite"


def test_default_state_path_is_resolved_per_invocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    brief_path = _write_brief(tmp_path, _valid_brief())
    state_root = tmp_path / "state"
    provider_error = "provider unavailable"

    def fail_build_workflow(model: object, *, checkpointer: object) -> Never:
        del model, checkpointer
        raise ProviderFailureError(provider_error)

    monkeypatch.setenv("XDG_STATE_HOME", str(state_root))
    monkeypatch.setattr(cli, "build_workflow", fail_build_workflow)
    result = runner.invoke(app, ["plan", str(brief_path), "--model", "fake:model"])

    assert result.exit_code == EXIT_WORKFLOW
    assert (state_root / "dev-pro-agents/checkpoints.sqlite").exists()


def test_checkpoint_provisioning_failure_exits_three(tmp_path: Path) -> None:
    brief_path = _write_brief(tmp_path, _valid_brief())
    blocked_parent = tmp_path / "not-a-directory"
    blocked_parent.write_text("file", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "plan",
            str(brief_path),
            "--model",
            "fake:model",
            "--state-path",
            str(blocked_parent / "state.sqlite"),
        ],
    )

    assert result.exit_code == EXIT_CONFIGURATION
    assert "could not prepare checkpoint" in result.stderr


def test_workflow_failure_exits_four(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    brief_path = _write_brief(tmp_path, _valid_brief())
    provider_error = "provider unavailable"

    def fail_build_workflow(model: object, *, checkpointer: object) -> Never:
        del model, checkpointer
        raise ProviderFailureError(provider_error)

    monkeypatch.setattr(cli, "build_workflow", fail_build_workflow)
    result = runner.invoke(
        app,
        [
            "plan",
            str(brief_path),
            "--model",
            "fake:model",
            "--state-path",
            str(tmp_path / "state.sqlite"),
        ],
    )

    assert result.exit_code == EXIT_WORKFLOW
    assert result.stderr == "workflow failed: provider unavailable\n"
    if os.name == "posix":
        assert (tmp_path / "state.sqlite").stat().st_mode & 0o777 == STATE_FILE_MODE


def test_output_cannot_replace_checkpoint_file(tmp_path: Path) -> None:
    brief_path = _write_brief(tmp_path, _valid_brief())
    state_path = tmp_path / "state.sqlite"

    result = runner.invoke(
        app,
        [
            "plan",
            str(brief_path),
            "--model",
            "fake:model",
            "--state-path",
            str(state_path),
            "--output",
            str(state_path),
        ],
    )

    assert result.exit_code == EXIT_INPUT
    assert result.stderr == "output must not refer to the checkpoint file\n"
    assert not state_path.exists()


def test_relative_output_alias_cannot_replace_checkpoint_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    brief_path = _write_brief(tmp_path, _valid_brief())
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(
        app,
        [
            "plan",
            str(brief_path),
            "--model",
            "fake:model",
            "--state-path",
            "state.sqlite",
            "--output",
            "./state.sqlite",
        ],
    )

    assert result.exit_code == EXIT_INPUT
    assert result.stderr == "output must not refer to the checkpoint file\n"
    assert not (tmp_path / "state.sqlite").exists()


def test_symlink_output_alias_cannot_replace_checkpoint_file(tmp_path: Path) -> None:
    brief_path = _write_brief(tmp_path, _valid_brief())
    state_path = tmp_path / "state.sqlite"
    state_path.write_bytes(b"checkpoint-data")
    output_path = tmp_path / "handoff.md"
    output_path.symlink_to(state_path)

    result = runner.invoke(
        app,
        [
            "plan",
            str(brief_path),
            "--model",
            "fake:model",
            "--state-path",
            str(state_path),
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == EXIT_INPUT
    assert result.stderr == "output must not refer to the checkpoint file\n"
    assert state_path.read_bytes() == b"checkpoint-data"


def test_hardlink_output_alias_cannot_replace_checkpoint_file(tmp_path: Path) -> None:
    brief_path = _write_brief(tmp_path, _valid_brief())
    state_path = tmp_path / "state.sqlite"
    state_path.write_bytes(b"checkpoint-data")
    output_path = tmp_path / "handoff.md"
    output_path.hardlink_to(state_path)

    result = runner.invoke(
        app,
        [
            "plan",
            str(brief_path),
            "--model",
            "fake:model",
            "--state-path",
            str(state_path),
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == EXIT_INPUT
    assert result.stderr == "output must not refer to the checkpoint file\n"
    assert state_path.read_bytes() == b"checkpoint-data"


def test_output_failure_exits_five(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    brief_path = _write_brief(tmp_path, _valid_brief())
    handoff = ImplementationHandoff(
        task_title="Task",
        summary="Summary",
        steps=(
            ImplementationStep(
                title="Step",
                outcome="Outcome",
                verification=("uv run pytest",),
            ),
        ),
        done_criteria=("Done",),
    )

    def fake_build_workflow(model: object, *, checkpointer: object) -> object:
        del model, checkpointer
        return object()

    def fake_run_handoff(
        workflow: object,
        brief: TaskBrief,
        *,
        thread_id: str,
    ) -> ImplementationHandoff:
        del workflow, brief, thread_id
        return handoff

    monkeypatch.setattr(cli, "build_workflow", fake_build_workflow)
    monkeypatch.setattr(cli, "run_handoff", fake_run_handoff)
    result = runner.invoke(
        app,
        [
            "plan",
            str(brief_path),
            "--model",
            "fake:model",
            "--state-path",
            str(tmp_path / "state.sqlite"),
            "--output",
            str(tmp_path),
        ],
    )

    assert result.exit_code == EXIT_OUTPUT
    assert "could not write output" in result.stderr


def test_stdout_failure_exits_five(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    brief_path = _write_brief(tmp_path, _valid_brief())
    handoff = ImplementationHandoff(
        task_title="Task",
        summary="Summary",
        steps=(
            ImplementationStep(
                title="Step",
                outcome="Outcome",
                verification=("uv run pytest",),
            ),
        ),
        done_criteria=("Done",),
    )

    def fake_build_workflow(model: object, *, checkpointer: object) -> object:
        del model, checkpointer
        return object()

    def fake_run_handoff(
        workflow: object,
        brief: TaskBrief,
        *,
        thread_id: str,
    ) -> ImplementationHandoff:
        del workflow, brief, thread_id
        return handoff

    original_echo = typer.echo
    broken_pipe_error = "closed pipe"

    def fail_stdout(
        message: object = None,
        *,
        err: bool = False,
        nl: bool = True,
    ) -> None:
        if not err:
            raise BrokenPipeError(broken_pipe_error)
        original_echo(message, err=err, nl=nl)

    monkeypatch.setattr(cli, "build_workflow", fake_build_workflow)
    monkeypatch.setattr(cli, "run_handoff", fake_run_handoff)
    monkeypatch.setattr(typer, "echo", fail_stdout)
    result = runner.invoke(
        app,
        [
            "plan",
            str(brief_path),
            "--model",
            "fake:model",
            "--state-path",
            str(tmp_path / "state.sqlite"),
        ],
    )

    assert result.exit_code == EXIT_OUTPUT
    assert result.stderr == "could not write output: closed pipe\n"


def _valid_brief() -> dict[str, object]:
    return {
        "title": "Task",
        "objective": "Objective",
        "acceptance_criteria": ["Done"],
    }


def _write_brief(tmp_path: Path, content: dict[str, object]) -> Path:
    brief_path = tmp_path / "brief.json"
    brief_path.write_text(json.dumps(content), encoding="utf-8")
    return brief_path
