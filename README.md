# dev-pro-agents

`dev-pro-agents` turns a strict JSON task brief into a validated implementation handoff. Its
LangChain v1 coordinator delegates planning and verification to two direct role tools, validates
the result with Pydantic, and can persist checkpoints with LangGraph's native SQLite saver.

Version 0.1 is deliberately read-only: it has no repository mutation, subprocess, shell, browser,
scraping, or network tools. Model providers and explicitly configured LangChain observability may
make external requests.

## Install

```bash
uv sync --frozen
```

Set `OPENAI_API_KEY` in the process environment. The CLI uses `openai:gpt-5-mini` by default;
override it with `--model` or `DEV_PRO_AGENTS_MODEL`. CLI model IDs use explicit lowercase
`provider:model` syntax; library callers can inject any `BaseChatModel` directly.

## Task brief

```json
{
  "title": "Add health endpoint",
  "objective": "Expose process readiness without leaking configuration.",
  "repository_context": "FastAPI service under src/api.",
  "constraints": ["Do not change authentication behavior."],
  "acceptance_criteria": ["GET /health returns 200 when dependencies are ready."]
}
```

## CLI

```bash
uv run dev-pro-agents plan brief.json --format markdown
uv run dev-pro-agents plan brief.json --format json --output handoff.json
cat brief.json | uv run dev-pro-agents plan - --thread-id health-endpoint
```

Checkpoint state defaults to `$XDG_STATE_HOME/dev-pro-agents/checkpoints.sqlite`, falling back to
`~/.local/state/dev-pro-agents/checkpoints.sqlite`. Each run gets a fresh isolated thread by default;
pass `--thread-id` to continue a deliberate thread. Exit codes are stable: `0` success, `2` invalid
input or CLI usage, `3` provider or checkpoint configuration, `4` workflow failure, and `5` output
failure.
The checkpoint database contains task briefs and model messages. The CLI creates it with private
file permissions on POSIX systems; still treat it as sensitive local data and remove it when stale.
For safety, `--output` must not name or alias the checkpoint file.

## Library

```python
from langgraph.checkpoint.sqlite import SqliteSaver

from dev_pro_agents import TaskBrief, build_workflow, run_handoff

brief = TaskBrief(
    title="Add health endpoint",
    objective="Expose readiness.",
    acceptance_criteria=("GET /health returns 200 when ready.",),
)
with SqliteSaver.from_conn_string("checkpoints.sqlite") as checkpointer:
    workflow = build_workflow("openai:gpt-5-mini", checkpointer=checkpointer)
    handoff = run_handoff(workflow, brief, thread_id="health-endpoint")
print(handoff.to_markdown())
```

The model and checkpointer are injected, so tests and alternate providers do not require hidden
global state. See [the architecture notes](docs/architecture.md) for the v0.1 boundaries.

## Development

```bash
uv lock --check
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests
uv run pytest
uv build
```

Before the first PyPI release, manually run one credentialed live-provider smoke test, configure
PyPI Trusted Publishing or release credentials, and recheck package-name availability. The local
and CI suites intentionally require neither provider credentials nor a package publication.
