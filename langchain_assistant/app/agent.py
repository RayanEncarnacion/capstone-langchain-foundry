"""The 'agent' for this phase: a direct chat model call to Microsoft Foundry.

No RAG, no tools, no memory yet (those live in retrieval.py / tools.py /
storage.py in later phases). Here we only:
  1. read connection settings from the environment (nothing hard-coded), and
  2. build a LangChain chat client that points at the Foundry model.

We target Foundry's OpenAI-compatible **v1 endpoint**, which the Foundry UI
shows as e.g. https://<resource>.services.ai.azure.com/openai/v1 . Because it
is OpenAI-compatible, we use `ChatOpenAI` with a `base_url` (no api-version).

Auth has two modes:
  - API key  -> set AZURE_OPENAI_API_KEY.
  - Azure AD -> no key; DefaultAzureCredential (keyless, recommended).
Both are sent as `Authorization: Bearer <secret>`, which the v1 endpoint accepts.
"""

import os

from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# Load a local, git-ignored .env so `uv run` picks up secrets in dev.
load_dotenv()

# Scope required to request AAD tokens for Azure AI services.
_AAD_SCOPE = "https://cognitiveservices.azure.com/.default"


class Settings:
    """Foundry connection settings, read from the environment."""

    def __init__(self) -> None:
        # v1 endpoint, e.g. https://<resource>.services.ai.azure.com/openai/v1
        self.endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        self.deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
        self.api_key = os.environ.get("AZURE_OPENAI_API_KEY")

    def require(self) -> None:
        """Fail fast if the required variables are not set."""
        missing = [
            name
            for name, value in {
                "AZURE_OPENAI_ENDPOINT": self.endpoint,
                "AZURE_OPENAI_DEPLOYMENT": self.deployment,
            }.items()
            if not value
        ]
        if missing:
            raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")


settings = Settings()


def _bearer_secret() -> str:
    """Return the secret to send as `Authorization: Bearer ...`.

    Prefer the API key; otherwise fetch an Azure AD token (keyless).
    """
    if settings.api_key:
        return settings.api_key
    # DefaultAzureCredential uses `az login`, managed identity, env vars, etc.
    token = DefaultAzureCredential().get_token(_AAD_SCOPE)
    return token.token


def build_chat_model() -> ChatOpenAI:
    """Return a configured chat model. No secrets are hard-coded."""
    return ChatOpenAI(
        base_url=settings.endpoint,   # Foundry v1, OpenAI-compatible route.
        model=settings.deployment,    # deployment name doubles as model name.
        api_key=_bearer_secret(),     # sent as Authorization: Bearer <secret>.
        temperature=0,                # deterministic-ish, easier to reason about.
    )


def ask(model: ChatOpenAI, message: str) -> str:
    """Send one user message, return the model's text reply."""
    # Direct call: one human message in, one AI message out.
    result = model.invoke([HumanMessage(content=message)])
    return result.content


# Sentinel the model must emit when the context does not support an answer.
ABSTAIN_TOKEN = "INSUFFICIENT_EVIDENCE"

# The grounding contract: answer only from context, cite, or abstain.
_RAG_SYSTEM_PROMPT = (
    "You are a study assistant that answers ONLY from the provided context.\n"
    "Rules:\n"
    "1. Use only facts found in the context blocks below. Do not use outside knowledge.\n"
    "2. Cite the chunks you used inline with their bracket numbers, e.g. [1], [2].\n"
    f"3. If the context does not contain enough information, reply with exactly "
    f"'{ABSTAIN_TOKEN}' and nothing else.\n"
    "Be concise."
)


def answer_with_rag(model: ChatOpenAI, question: str, top_k: int = 4):
    """Explicit two-step RAG: retrieve, then generate a grounded answer.

    Returns (answer_text, abstained_bool, retrieved_chunks). The chunks are
    returned so the caller can print/return the exact evidence used.
    """
    # Imported lazily to avoid a circular import (retrieval -> storage -> agent).
    from .retrieval import format_context, retrieve

    chunks = retrieve(question, top_k=top_k)

    # No hits at all -> abstain without even calling the model.
    if not chunks:
        return (
            "I don't have enough information in the notes to answer that.",
            True,
            chunks,
        )

    context = format_context(chunks)
    user_content = f"Context:\n{context}\n\nQuestion: {question}"

    result = model.invoke(
        [
            SystemMessage(content=_RAG_SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ]
    )
    answer = result.content.strip()

    # Detect abstention and normalise it into a friendly message.
    if ABSTAIN_TOKEN in answer:
        return (
            "I don't have enough information in the notes to answer that.",
            True,
            chunks,
        )

    return answer, False, chunks


# ---------------------------------------------------------------------------
# Phase 4: tool-calling agent (create_agent) over notes + tasks
# ---------------------------------------------------------------------------
# Tool schemas describe each tool. This prompt only adds global routing,
# grounding, and error-handling rules that span all tools.
_AGENT_SYSTEM_PROMPT = (
    "You are a study assistant with access to tools. Decide for each turn "
    "whether a tool is needed; do not call tools you do not need.\n\n"
    "Use note search only when the answer needs factual knowledge from the "
    "user's notes. Do not search for greetings, chit-chat, or task management.\n\n"
    "Grounding rules when you use search_notes:\n"
    "1. Answer only from the returned snippets; do not use outside knowledge.\n"
    "2. Cite the snippets you used by their source and chunk_id.\n"
    "3. If the snippets do not contain enough information, say you don't have "
    "enough information in the notes rather than guessing.\n\n"
    "If a tool returns {\"ok\": false, ...}, tell the user the action failed "
    "and briefly why. Never pretend a failed tool succeeded. Be concise."
)


def build_agent():
    """Construct the tool-calling agent with create_agent.

    Binds the chat model to the notes + task tools and installs the fused
    system prompt. Returns a compiled agent graph you invoke with a message
    list, e.g. agent.invoke({"messages": [HumanMessage(...)]}).
    """
    # Lazy import so importing agent.py stays cheap and side-effect free.
    from langchain.agents import create_agent

    from .tools import ALL_TOOLS

    return create_agent(
        build_chat_model(),
        ALL_TOOLS,
        system_prompt=_AGENT_SYSTEM_PROMPT,
    )


def run_agent(agent, message: str) -> tuple[str, list[dict]]:
    """Run one user turn through the agent.

    Returns (reply_text, tool_calls) where tool_calls is a small list of
    {"name", "args"} dicts describing which tools the agent invoked. The
    model-tool loop itself is handled by create_agent; we only read the
    resulting message history.
    """
    result = agent.invoke({"messages": [HumanMessage(content=message)]})
    messages = result["messages"]

    # Collect any tool invocations across the turn for transparency.
    tool_calls: list[dict] = []
    for msg in messages:
        for call in getattr(msg, "tool_calls", None) or []:
            tool_calls.append({"name": call["name"], "args": call.get("args", {})})

    # The final message is the agent's text answer to the user.
    reply = messages[-1].content if messages else ""
    return reply, tool_calls
