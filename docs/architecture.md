# Architecture

## Canonical flow

1. `TaskBrief` rejects unknown or incomplete input.
2. A LangChain v1 coordinator calls the implementation planner role.
3. The coordinator passes the canonical brief and that draft to the verification reviewer role.
4. `ToolStrategy` validates the final response as `ImplementationHandoff` and returns correctable
   schema errors to the model for retry.
5. The CLI's native SQLite serializer explicitly allowlists that handoff type for safe resume.
6. The caller selects deterministic Markdown or JSON rendering.

The role handoffs are ordinary LangChain tools that accept and return text. They have no filesystem,
process, browser, scraper, database, or deployment capability. Only the coordinator receives the
injected checkpointer; short-lived role calls do not compete for its thread state.

## Dependency boundaries

The library owns contracts, prompts, orchestration, and rendering. LangChain owns agent execution
and structured-output tool binding. LangGraph owns checkpoint persistence. Pydantic owns validation.
The CLI opens a native `SqliteSaver`; the library never creates a database implicitly.

The same injected model is used for all roles. Add per-role model routing only after measured quality
or cost evidence shows that the additional configuration pays for itself.

## Deliberate exclusions

Version 0.1 does not execute a handoff. Mutation, shell, subprocess, network-research, and scraping
tools remain out of scope until their trust boundaries, approval model, confinement, observability,
and failure behavior have a separately reviewed design. The design gate is tracked in
[GitHub issue #1](https://github.com/BjornMelin/dev-pro-agents/issues/1).
