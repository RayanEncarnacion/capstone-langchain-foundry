"""Run one agent turn with OpenTelemetry export to Application Insights / Foundry.

This is the *separate process* from Phase 7's optional comparison: it inspects
the same LangChain app through Azure Monitor instead of LangSmith. Because
enable_azure_monitor_tracing() disables LangSmith here, no duplicate traces are
sent to both systems.

Usage:
    uv run python -m langchain_assistant.scripts.run_with_otel "How long is a study session?"

Then open Application Insights (Transaction search / End-to-end transaction) or
the Foundry project's Tracing tab to see the spans. Spans can take 1-2 minutes
to appear.
"""

import sys
import uuid

from dotenv import load_dotenv


def main() -> None:
    load_dotenv()

    # Enable Azure Monitor export BEFORE building the agent so instrumentation
    # wraps the LangChain callbacks from the first run.
    from ..app.observability import enable_azure_monitor_tracing

    enable_azure_monitor_tracing()

    from ..app.agent import build_agent, run_agent
    from ..app.storage import build_checkpointer, build_store

    message = sys.argv[1] if len(sys.argv) > 1 else "How long is a standard study session?"

    agent = build_agent(checkpointer=build_checkpointer(), store=build_store())
    reply, tool_calls, pending = run_agent(
        agent,
        message,
        thread_id=f"otel-{uuid.uuid4()}",
        user_id="otel-user",
    )

    print(f"reply: {reply}")
    print(f"tool_calls: {tool_calls}")
    print(f"pending: {pending}")

    # Flush spans before the process exits (exporter batches in the background).
    from opentelemetry import trace

    provider = trace.get_tracer_provider()
    if hasattr(provider, "force_flush"):
        provider.force_flush()
    print("Flushed traces to Azure Monitor. Allow 1-2 min to appear.")


if __name__ == "__main__":
    main()
