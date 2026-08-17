"""Agent task tools: the model's only way to touch the task store.

Phase 4 — three Cosmos DB task tools, each an Agent Framework `@tool` so the
Foundry agent can call them by name with validated arguments:

  - list_tasks    -> read the caller's tasks   (Azure Cosmos DB).
  - create_task   -> add a task                (Azure Cosmos DB).
  - complete_task -> mark a task done          (Azure Cosmos DB).

Design rules for this phase:
  1. Pydantic `Field` annotations describe every model-visible argument, so the
     model gets a narrow, typed, self-documenting schema.
  2. Identity is NOT a tool argument. The authenticated user id is injected at
     run time through the `FunctionInvocationContext` (`ctx.kwargs["user_id"]`),
     which Agent Framework hides from the JSON schema shown to the model. The
     model can neither see nor set which user/partition it operates on.
  3. Every Cosmos operation is scoped to that injected user id as the partition
     key, so one user's turn can never read or mutate another user's tasks —
     even if the model fabricates an id.
  4. Tools return SMALL, STRUCTURED dicts and CATCH errors as
     {"ok": False, "error": ...} so failures are visible instead of crashing
     the agent loop. Invalid arguments are rejected inside the tool.
"""

import uuid
from datetime import date
from typing import Annotated, Literal

from agent_framework import FunctionInvocationContext, tool
from pydantic import Field

# Fallback partition key value when a run carries no authenticated user id.
# In production every request is authenticated, so this only guards local
# smoke tests that forget to inject identity.
DEFAULT_USER_ID = "default-user"


def _user_id(ctx: FunctionInvocationContext) -> str:
    """Read the authenticated user id injected into the invocation context.

    Supplied per-run via `agent.run(..., function_invocation_kwargs=
    {"user_id": ...})`. Never comes from a model-visible argument.
    """
    return (ctx.kwargs or {}).get("user_id") or DEFAULT_USER_ID


# ---------------------------------------------------------------------------
# list_tasks
# ---------------------------------------------------------------------------
@tool(approval_mode="never_require")
def list_tasks(
    status: Annotated[
        Literal["pending", "done", "all"],
        Field(description="Filter by task status, or 'all' for everything"),
    ] = "all",
    *,
    ctx: FunctionInvocationContext,
) -> dict:
    """List the current user's tasks, optionally filtered by status.

    Use when the user asks what tasks/to-dos they have. Returns a small
    structured list, not raw database records.
    """
    try:
        from .storage import get_tasks_container

        user_id = _user_id(ctx)
        container = get_tasks_container()
        if status == "all":
            query = "SELECT c.id, c.title, c.status, c.due_date FROM c"
            params: list[dict] = []
        else:
            query = (
                "SELECT c.id, c.title, c.status, c.due_date FROM c "
                "WHERE c.status = @status"
            )
            params = [{"name": "@status", "value": status}]

        # partition_key pins the query to the authenticated user's partition;
        # the model cannot widen this to other users.
        items = list(
            container.query_items(
                query=query,
                parameters=params,
                partition_key=user_id,
            )
        )
        return {"ok": True, "count": len(items), "tasks": items}
    except Exception as exc:
        return {"ok": False, "error": f"list_tasks failed: {exc}"}


# ---------------------------------------------------------------------------
# create_task
# ---------------------------------------------------------------------------
@tool(approval_mode="never_require")
def create_task(
    title: Annotated[
        str, Field(min_length=1, description="Short task description")
    ],
    due_date: Annotated[
        str | None, Field(description="Optional ISO date (YYYY-MM-DD)")
    ] = None,
    *,
    ctx: FunctionInvocationContext,
) -> dict:
    """Create a new task for the current user.

    Use when the user asks to add/create/remember a task or to-do. Returns
    the created task's id and fields.
    """
    try:
        from .storage import get_tasks_container

        # Reject invalid arguments inside the tool: an optional due date must
        # be a real ISO calendar date, not free text.
        if due_date is not None:
            try:
                date.fromisoformat(due_date)
            except ValueError:
                return {
                    "ok": False,
                    "error": (
                        f"due_date must be an ISO date (YYYY-MM-DD); "
                        f"got {due_date!r}"
                    ),
                }

        user_id = _user_id(ctx)
        item = {
            "id": str(uuid.uuid4()),
            "user_id": user_id,  # partition key — bound to the caller.
            "title": title,
            "status": "pending",
            "due_date": due_date,
        }
        container = get_tasks_container()
        created = container.create_item(body=item)
        return {
            "ok": True,
            "task": {
                "id": created["id"],
                "title": created["title"],
                "status": created["status"],
                "due_date": created.get("due_date"),
            },
        }
    except Exception as exc:
        return {"ok": False, "error": f"create_task failed: {exc}"}


# ---------------------------------------------------------------------------
# complete_task
# ---------------------------------------------------------------------------
@tool(approval_mode="never_require")
def complete_task(
    id: Annotated[
        str, Field(min_length=1, description="The id of the task to mark as done")
    ],
    *,
    ctx: FunctionInvocationContext,
) -> dict:
    """Mark one of the current user's tasks as done.

    Use when the user says they finished/completed a task. Takes the task id
    (as returned by list_tasks/create_task) and flips its status to 'done'.
    """
    try:
        from .storage import get_tasks_container

        user_id = _user_id(ctx)
        container = get_tasks_container()

        # Point-read scoped to the caller's partition: if the id belongs to
        # another user (or does not exist here), the read misses and we stop.
        try:
            item = container.read_item(item=id, partition_key=user_id)
        except Exception:
            return {"ok": False, "error": f"task {id!r} not found for this user"}

        item["status"] = "done"
        updated = container.replace_item(item=id, body=item)
        return {
            "ok": True,
            "task": {
                "id": updated["id"],
                "title": updated["title"],
                "status": updated["status"],
                "due_date": updated.get("due_date"),
            },
        }
    except Exception as exc:
        return {"ok": False, "error": f"complete_task failed: {exc}"}


# Task tools the agent binds at build time (KB tool is added separately).
TASK_TOOLS = [list_tasks, create_task, complete_task]
