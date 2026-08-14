"""Agent tools: the model's only way to touch the outside world.

Three tools, each a plain function decorated with LangChain's `@tool` so the
agent can call them by name with validated arguments:

  - search_notes  -> read-only knowledge lookup (Azure AI Search / RAG).
  - list_tasks    -> read the user's tasks   (Azure Cosmos DB).
  - create_task   -> add a task              (Azure Cosmos DB).

Design rules for this phase:
  1. Pydantic argument schemas keep the model honest (narrow, typed inputs).
  2. Every tool returns a SMALL, STRUCTURED dict. We never dump full chunk
     text or raw SDK objects into the model's context.
  3. Errors are CAUGHT and returned as {"ok": False, "error": ...} so a
     failure is visible to the model (and the user) instead of crashing the
     agent loop or being silently swallowed.
"""

import uuid

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedStore
from langgraph.store.base import BaseStore
from pydantic import BaseModel, Field
from typing import Annotated, Literal

# Fallback partition key value when a request carries no authenticated user.
# Real user ids arrive per-turn via config["configurable"]["user_id"].
DEFAULT_USER_ID = "default-user"


def _user_id(config: RunnableConfig) -> str:
    """Read the authenticated user id injected into the run config."""
    return (config.get("configurable") or {}).get("user_id") or DEFAULT_USER_ID


def _prefs_namespace(user_id: str) -> tuple[str, str]:
    """Store namespace for a user's preferences (isolates users from each other)."""
    return (user_id, "preferences")

# Keep snippets short so tool output stays token-cheap.
_SNIPPET_CHARS = 240


# ---------------------------------------------------------------------------
# search_notes
# ---------------------------------------------------------------------------
class SearchNotesArgs(BaseModel):
    """Arguments for searching the indexed study notes."""

    query: str = Field(..., min_length=1, description="What to look up in the notes")
    top_k: int = Field(
        default=4, ge=1, le=10, description="How many note chunks to return"
    )


@tool(args_schema=SearchNotesArgs)
def search_notes(query: str, top_k: int = 4) -> dict:
    """Search the user's study notes for relevant information.

    Use ONLY when you need factual knowledge from the notes to answer a
    question. Returns short, citable snippets (not full documents).
    """
    try:
        # Lazy import: avoids pulling Search deps unless the tool is used.
        from .retrieval import retrieve

        chunks = retrieve(query, top_k=top_k)
        results = [
            {
                "source": c.source,
                "title": c.title,
                "chunk_id": c.chunk_id,
                "score": round(c.score, 4),
                "snippet": c.content[:_SNIPPET_CHARS],
            }
            for c in chunks
        ]
        return {"ok": True, "count": len(results), "results": results}
    except Exception as exc:  # never let a retrieval failure kill the loop.
        print(f"search_notes failed: {exc}")
        return {"ok": False, "error": f"search_notes failed: {exc}"}


# ---------------------------------------------------------------------------
# list_tasks
# ---------------------------------------------------------------------------
class ListTasksArgs(BaseModel):
    """Arguments for listing tasks."""

    status: Literal["pending", "done", "all"] = Field(
        default="all", description="Filter by task status, or 'all'"
    )


@tool(args_schema=ListTasksArgs)
def list_tasks(status: str = "all", *, config: RunnableConfig) -> dict:
    """List the user's tasks, optionally filtered by status.

    Use when the user asks what tasks/to-dos they have. Returns a small
    structured list, not raw database records.
    """
    try:
        from .storage import get_tasks_container

        user_id = _user_id(config)
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
class CreateTaskArgs(BaseModel):
    """Arguments for creating a task."""

    title: str = Field(..., min_length=1, description="Short task description")
    due_date: str | None = Field(
        default=None, description="Optional ISO date (YYYY-MM-DD)"
    )


@tool(args_schema=CreateTaskArgs)
def create_task(
    title: str, due_date: str | None = None, *, config: RunnableConfig
) -> dict:
    """Create a new task for the user.

    Use when the user asks to add/create/remember a task or to-do. Returns
    the created task's id and fields.
    """
    try:
        from .storage import get_tasks_container

        item = {
            "id": str(uuid.uuid4()),
            "user_id": _user_id(config),
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
# set_preference / get_preference  (long-term, cross-thread, per-user memory)
# ---------------------------------------------------------------------------
class SetPreferenceArgs(BaseModel):
    """Arguments for saving a durable user preference."""

    key: str = Field(
        ..., min_length=1, description="Preference name, e.g. 'session_duration'"
    )
    value: str = Field(..., min_length=1, description="Preference value, e.g. '25 minutes'")


@tool(args_schema=SetPreferenceArgs)
def set_preference(
    key: str,
    value: str,
    *,
    config: RunnableConfig,
    store: Annotated[BaseStore, InjectedStore()],
) -> dict:
    """Remember a durable preference for the user across all conversations.

    Use when the user states a lasting preference (e.g. their preferred study
    session duration). Persists to the long-term store, namespaced by user.
    """
    try:
        user_id = _user_id(config)
        store.put(_prefs_namespace(user_id), key, {"value": value})
        return {"ok": True, "key": key, "value": value}
    except Exception as exc:
        return {"ok": False, "error": f"set_preference failed: {exc}"}


class GetPreferenceArgs(BaseModel):
    """Arguments for reading a durable user preference."""

    key: str = Field(
        ..., min_length=1, description="Preference name to look up, e.g. 'session_duration'"
    )


@tool(args_schema=GetPreferenceArgs)
def get_preference(
    key: str,
    *,
    config: RunnableConfig,
    store: Annotated[BaseStore, InjectedStore()],
) -> dict:
    """Look up a durable preference the user set earlier (any conversation).

    Use when answering depends on a saved preference. Reads from the
    long-term store, namespaced by user, so only this user's value is visible.
    """
    try:
        user_id = _user_id(config)
        item = store.get(_prefs_namespace(user_id), key)
        print(f"get_preference: item={item}, user_id={user_id}, key={key}")
        if item is None:
            return {"ok": True, "found": False, "key": key}
        return {"ok": True, "found": True, "key": key, "value": item.value.get("value")}
    except Exception as exc:
        return {"ok": False, "error": f"get_preference failed: {exc}"}


# Exported list the agent binds at build time.
ALL_TOOLS = [search_notes, list_tasks, create_task, set_preference, get_preference]
