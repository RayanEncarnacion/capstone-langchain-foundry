"""Microsoft Foundry / Application Insights tracing for the LangChain agent.

Uses langchain-azure-ai's enable_auto_tracing() to globally patch all LangChain
callback managers. Every agent.invoke() after this call emits GenAI-convention
OTel spans to Application Insights. These surface in the Foundry Tracing tab
when the App Insights resource is linked to the Foundry project.

Normal development still uses LangSmith. To avoid double-sending traces, this
module disables LangSmith in the current process.
"""

import os

DEFAULT_AGENT_ID = os.environ.get("FOUNDRY_AGENT_ID", "study-assistant")


def _disable_langsmith() -> None:
    """Turn LangSmith off in this process so traces aren't double-sent."""
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ.pop("LANGCHAIN_API_KEY", None)


def enable_foundry_tracing(disable_langsmith: bool = True) -> None:
    """Globally enable Foundry / App Insights tracing for all LangChain runs.

    Must be called BEFORE building any agent or chat model. After this call,
    every LangChain invoke (model, tool, graph node) automatically emits spans.

    Resolution order for the telemetry target:
      1. AZURE_AI_PROJECT_ENDPOINT -> resolves linked App Insights from the
         Foundry project (keyless, DefaultAzureCredential).
      2. APPLICATIONINSIGHTS_CONNECTION_STRING -> target App Insights directly.
    """
    from langchain_azure_ai.callbacks.tracers import enable_auto_tracing

    project_endpoint = os.environ.get("AZURE_AI_PROJECT_ENDPOINT") or os.environ.get(
        "FOUNDRY_PROJECT_ENDPOINT"
    )
    connection_string = os.environ.get("APPLICATIONINSIGHTS_CONNECTION_STRING")

    if not project_endpoint and not connection_string:
        raise RuntimeError(
            "Set AZURE_AI_PROJECT_ENDPOINT (preferred) or "
            "APPLICATIONINSIGHTS_CONNECTION_STRING."
        )

    if disable_langsmith:
        _disable_langsmith()

    kwargs = {
        "enable_content_recording": True,
        "provider_name": "azure.ai.openai",
        "trace_all_langgraph_nodes": True,
        "auto_configure_azure_monitor": True,
    }

    if project_endpoint:
        from azure.identity import DefaultAzureCredential

        kwargs["project_endpoint"] = project_endpoint
        kwargs["credential"] = DefaultAzureCredential()
    else:
        kwargs["connection_string"] = connection_string

    enable_auto_tracing(**kwargs)
