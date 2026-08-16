"""Run one agent turn with Foundry / Application Insights tracing.

Uses enable_auto_tracing() — every LangChain/LangGraph invoke in this process
automatically emits GenAI-convention spans to Azure Monitor. LangSmith is
disabled, so no duplicate traces.

Config (set one telemetry target in .env):
  AZURE_AI_PROJECT_ENDPOINT          preferred; ties traces to Foundry project
  APPLICATIONINSIGHTS_CONNECTION_STRING   direct App Insights fallback

Usage:
    uv run python -m langchain_assistant.scripts.run_with_otel "How long is a study session?"

Spans appear in Foundry Tracing tab / App Insights after 1-2 minutes.
"""

import sys
import uuid

from dotenv import load_dotenv


def main() -> None:
    load_dotenv()

    # Enable auto-tracing BEFORE building the agent. After this call, every
    # LangChain invoke is automatically instrumented — no callbacks needed.
    from ..app.observability import enable_foundry_tracing

    enable_foundry_tracing()
    print("Foundry tracing enabled.")

    from ..app.agent import build_agent, run_agent
    from ..app.storage import build_checkpointer, build_store

    message = sys.argv[1] if len(sys.argv) > 1 else "How long is a standard study session?"

    print(f"Building agent...")
    agent = build_agent(checkpointer=build_checkpointer(), store=build_store())

    print(f"Running: {message}")
    reply, tool_calls, pending = run_agent(
        agent,
        message,
        thread_id=f"otel-{uuid.uuid4()}",
        user_id="otel-user",
    )

    print(f"reply: {reply}")
    print(f"tool_calls: {tool_calls}")
    print(f"pending: {pending}")

    # Flush spans before the process exits.
    from opentelemetry import trace

    provider = trace.get_tracer_provider()
    if hasattr(provider, "force_flush"):
        provider.force_flush()
    print("Flushed traces to Azure Monitor. Allow 1-2 min to appear in Foundry.")


if __name__ == "__main__":
    main()
