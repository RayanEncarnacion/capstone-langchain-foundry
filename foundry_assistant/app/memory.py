"""Cross-session user preferences: Foundry Memory Store (primary) + Cosmos fallback.

Phase 6 — the agent keeps two very different kinds of state:

  * Conversation history  — held in the ephemeral in-memory AgentSession
    (see agent.py). Session-specific by design; lost on restart.
  * A durable user preference (a preferred name / nickname) — held OUTSIDE the
    session so it survives a brand-new session.

The preference is stored in a Foundry managed Memory Store when that preview
feature is reachable, scoped to the authenticated user's stable Entra object id
(`oid`). Foundry managed memory is currently preview, so if the store cannot be
reached we transparently fall back to a Cosmos-backed provider that implements
the SAME behaviour against the existing `preferences` container
(partition key `/user_id`). Either way the caller sees one small interface.

Retrieval is deliberately NOT auto-injected into the model context: preferences
surface only when the user asks, via the explicit tools defined at the bottom of
this module (`remember_nickname`, `get_my_preferences`, `forget_my_preferences`).

Environment variables (all optional; see .env.foundry.example):
    MEMORY_STORE_NAME            — Foundry memory store name.
    PREFERENCE_BACKEND           — force "foundry" | "cosmos" | "auto" (default).
    COSMOS_PREFERENCES_CONTAINER — Cosmos container for preferences.
Foundry endpoint/credential are reused from agent.py; Cosmos settings from
storage.py.
"""

import os
from typing import Annotated

from agent_framework import FunctionInvocationContext, tool
from dotenv import load_dotenv
from pydantic import Field

load_dotenv()

# Stable marker so a preference written as free-text memory can be found and
# de-duplicated later. Every nickname memory content starts with this prefix.
_NICKNAME_PREFIX = "Preferred name/nickname: "

# The single preference key this phase supports. Kept as a constant so the
# store layer stays generic (key/value) while the tools stay nickname-specific.
NICKNAME_KEY = "nickname"


class MemorySettings:
    """Preference-store configuration read from the environment."""

    def __init__(self) -> None:
        self.store_name = os.environ.get(
            "MEMORY_STORE_NAME", "foundry-assistant-memory-store"
        )
        # Memory stores are PROJECT-scoped, so they need the
        # .../api/projects/<project> endpoint — not the resource-level endpoint
        # the FoundryChatClient uses. Prefer an explicit override; else derive
        # from FOUNDRY_PROJECT_ENDPOINT + MEMORY_PROJECT_NAME when needed.
        self.project_endpoint = os.environ.get("FOUNDRY_PROJECT_SUFFIXED_ENDPOINT")
        self.project_name = os.environ.get("MEMORY_PROJECT_NAME")
        # auto (default) probes Foundry then falls back to Cosmos.
        self.backend = os.environ.get("PREFERENCE_BACKEND", "auto").lower()
        self.cosmos_container = os.environ.get(
            "COSMOS_PREFERENCES_CONTAINER", "memories"
        )


memory_settings = MemorySettings()


# ---------------------------------------------------------------------------
# Foundry managed Memory Store backend (primary)
# ---------------------------------------------------------------------------
class FoundryPreferenceStore:
    """User preferences backed by a Foundry managed Memory Store (preview).

    Memories are scoped to the authenticated user's id so one user can never
    read or delete another's. A preference is a single USER_PROFILE memory whose
    content begins with `_NICKNAME_PREFIX`; setting it first clears any prior
    nickname memory so exactly one survives.
    """

    backend = "foundry"

    def __init__(self) -> None:
        from azure.ai.projects import AIProjectClient

        from .agent import _credential, settings

        endpoint = self._resolve_endpoint(settings.project_endpoint)
        if not endpoint:
            raise RuntimeError("No Foundry project endpoint for memory store.")
        self._client = AIProjectClient(endpoint=endpoint, credential=_credential)
        self._name = memory_settings.store_name

    @staticmethod
    def _resolve_endpoint(foundry_endpoint: str | None) -> str | None:
        """Return the project-scoped endpoint the memory stores API requires.

        Precedence: explicit FOUNDRY_PROJECT_SUFFIXED_ENDPOINT > FOUNDRY_PROJECT_ENDPOINT
        already containing '/api/projects/' > FOUNDRY_PROJECT_ENDPOINT joined
        with MEMORY_PROJECT_NAME. The chat client's resource-level endpoint is
        left untouched.
        """
        if memory_settings.project_endpoint:
            return memory_settings.project_endpoint
        if not foundry_endpoint:
            return None
        if "/api/projects/" in foundry_endpoint:
            return foundry_endpoint
        if memory_settings.project_name:
            base = foundry_endpoint.rstrip("/")
            return f"{base}/api/projects/{memory_settings.project_name}"
        return foundry_endpoint

    def probe(self) -> None:
        """Raise if the memory store is unreachable (drives the fallback)."""
        self._client.beta.memory_stores.get(self._name)

    def set_preference(self, user_id: str, key: str, value: str) -> None:
        from azure.ai.projects.models import MemoryItemKind

        # Enforce a single value per key: drop existing matches, then create.
        self.delete_preference(user_id, key)
        self._client.beta.memory_stores.create_memory(
            self._name,
            scope=user_id,
            content=f"{_NICKNAME_PREFIX}{value}",
            kind=MemoryItemKind.USER_PROFILE,
        )

    def get_preferences(self, user_id: str) -> list[dict]:
        items = self._client.beta.memory_stores.list_memories(
            self._name, scope=user_id
        )
        prefs: list[dict] = []
        for item in items:
            content = getattr(item, "content", "") or ""
            if content.startswith(_NICKNAME_PREFIX):
                prefs.append(
                    {
                        "key": NICKNAME_KEY,
                        "value": content[len(_NICKNAME_PREFIX):],
                    }
                )
        return prefs

    def delete_preference(self, user_id: str, key: str | None = None) -> int:
        items = list(
            self._client.beta.memory_stores.list_memories(self._name, scope=user_id)
        )
        deleted = 0
        for item in items:
            content = getattr(item, "content", "") or ""
            if content.startswith(_NICKNAME_PREFIX):
                self._client.beta.memory_stores.delete_memory(
                    self._name, getattr(item, "memory_id")
                )
                deleted += 1
        return deleted


# ---------------------------------------------------------------------------
# Cosmos-backed fallback provider (same behaviour, keyless/connection string)
# ---------------------------------------------------------------------------
class CosmosPreferenceStore:
    """User preferences backed by Cosmos DB when managed memory is unavailable.

    One document per (user, key), partitioned by `/user_id`, so reads/writes are
    always pinned to the authenticated user's partition — identical isolation to
    the Foundry-scoped store.
    """

    backend = "cosmos"

    def _container(self):
        from azure.cosmos import CosmosClient, PartitionKey
        from azure.identity import DefaultAzureCredential

        from .storage import cosmos_settings

        cosmos_settings.require()
        if cosmos_settings.connection_string:
            client = CosmosClient.from_connection_string(
                cosmos_settings.connection_string
            )
        else:
            client = CosmosClient(
                url=cosmos_settings.endpoint, credential=DefaultAzureCredential()
            )
        database = client.get_database_client(cosmos_settings.database)
        # Preferences are partitioned by /user_id (same isolation model as
        # tasks). Create on first use so the fallback works without manual
        # provisioning; idempotent when the container already exists.
        return database.create_container_if_not_exists(
            id=memory_settings.cosmos_container,
            partition_key=PartitionKey(path="/user_id"),
        )

    @staticmethod
    def _doc_id(user_id: str, key: str) -> str:
        return f"{user_id}:{key}"

    def set_preference(self, user_id: str, key: str, value: str) -> None:
        self._container().upsert_item(
            {
                "id": self._doc_id(user_id, key),
                "user_id": user_id,
                "key": key,
                "value": value,
            }
        )

    def get_preferences(self, user_id: str) -> list[dict]:
        # `key`/`value` are reserved words in Cosmos SQL, so use quoted
        # property accessors and alias to plain names.
        items = self._container().query_items(
            query='SELECT c["key"] AS pkey, c["value"] AS pvalue FROM c',
            partition_key=user_id,
        )

        return [{"key": i["pkey"], "value": i["pvalue"]} for i in items]

    def delete_preference(self, user_id: str, key: str | None = None) -> int:
        container = self._container()
        if key is not None:
            try:
                container.delete_item(item=self._doc_id(user_id, key), partition_key=user_id)
                return 1
            except Exception:
                return 0
        # No key: clear every preference in the user's partition.
        ids = [
            i["id"]
            for i in container.query_items(
                query="SELECT c.id FROM c", partition_key=user_id
            )
        ]
        for doc_id in ids:
            container.delete_item(item=doc_id, partition_key=user_id)
        return len(ids)


# ---------------------------------------------------------------------------
# Backend selection: managed memory first, Cosmos fallback (cached once)
# ---------------------------------------------------------------------------
_store = None


def get_preference_store():
    """Return the active preference store, choosing a backend once.

    `PREFERENCE_BACKEND` forces a backend; the default ("auto") tries the
    Foundry managed Memory Store and falls back to Cosmos when the preview
    store cannot be reached. The choice is cached for the process lifetime.
    """
    global _store
    if _store is not None:
        return _store

    backend = memory_settings.backend
    if backend == "cosmos":
        _store = CosmosPreferenceStore()
        return _store
    if backend == "foundry":
        _store = FoundryPreferenceStore()
        return _store

    # auto: probe Foundry, fall back to Cosmos on any failure.
    try:
        foundry = FoundryPreferenceStore()
        foundry.probe()
        _store = foundry
    except Exception as exc:
        _store = CosmosPreferenceStore()
    return _store


def _user_id(ctx: FunctionInvocationContext) -> str:
    """Authenticated user id injected per-run (never a model-visible argument)."""
    from .tools import DEFAULT_USER_ID

    return (ctx.kwargs or {}).get("user_id") or DEFAULT_USER_ID


# ---------------------------------------------------------------------------
# Preference tools (retrieval only when the user asks)
# ---------------------------------------------------------------------------
@tool(approval_mode="always_require")
def remember_nickname(
    nickname: Annotated[
        str,
        Field(min_length=1, description="How the user wants to be addressed"),
    ],
    *,
    ctx: FunctionInvocationContext,
) -> dict:
    """Persist the user's preferred name/nickname across sessions.

    Use ONLY when the user explicitly asks you to remember their name or how to
    address them. Stored durably and scoped to this user; it survives new
    sessions. Replaces any previously remembered nickname.
    """
    try:
        store = get_preference_store()
        store.set_preference(_user_id(ctx), NICKNAME_KEY, nickname)
        return {"ok": True, "backend": store.backend, "nickname": nickname}
    except Exception as exc:
        return {"ok": False, "error": f"remember_nickname failed: {exc}"}


@tool(approval_mode="never_require")
def get_my_preferences(*, ctx: FunctionInvocationContext) -> dict:
    """Return the preferences remembered for the current user.

    Use when the user asks what you remember about them or what their preferred
    name is. Returns a small key/value list; empty when nothing is stored.
    """
    try:
        store = get_preference_store()
        prefs = store.get_preferences(_user_id(ctx))
        return {"ok": True, "backend": store.backend, "preferences": prefs}
    except Exception as exc:
        return {"ok": False, "error": f"get_my_preferences failed: {exc}"}


@tool(approval_mode="always_require")
def forget_my_preferences(*, ctx: FunctionInvocationContext) -> dict:
    """Delete the current user's remembered preferences (nickname).

    Use when the user asks you to forget their name/preferences. Removes the
    stored value so it no longer surfaces in future sessions.
    """
    try:
        store = get_preference_store()
        deleted = store.delete_preference(_user_id(ctx), NICKNAME_KEY)
        return {"ok": True, "backend": store.backend, "deleted": deleted}
    except Exception as exc:
        return {"ok": False, "error": f"forget_my_preferences failed: {exc}"}


PREFERENCE_TOOLS = [remember_nickname, get_my_preferences, forget_my_preferences]
