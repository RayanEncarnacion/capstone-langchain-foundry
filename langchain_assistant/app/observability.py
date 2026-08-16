"""Optional Microsoft OpenTelemetry export to Application Insights / Foundry.

This is the *comparison* observability path from Phase 7. Normal development
uses LangSmith (LANGCHAIN_TRACING_V2=true). To inspect the same LangChain app
through Application Insights or Foundry observability instead, run a separate
process that calls enable_azure_monitor_tracing() at startup.

Export uses azure-monitor-opentelemetry (Azure Monitor ingestion endpoint), so
the App Insights resource does NOT need the generic OTLP endpoint enabled. The
connection string carries the ingestion endpoint it needs.

To AVOID duplicate traces we hard-disable LangSmith in this process (see
enable_azure_monitor_tracing): a given run should stream to one backend only.
"""

import os


def enable_azure_monitor_tracing(disable_langsmith: bool = True) -> None:
    """Instrument LangChain and export spans to Application Insights.

    Args:
        disable_langsmith: when True (default), turn off LangSmith tracing in
            this process so runs are not double-sent to both backends.

    Requires APPLICATIONINSIGHTS_CONNECTION_STRING in the environment.
    """
    connection_string = os.environ.get("APPLICATIONINSIGHTS_CONNECTION_STRING")
    if not connection_string:
        raise RuntimeError(
            "Set APPLICATIONINSIGHTS_CONNECTION_STRING to export to Azure Monitor."
        )

    if disable_langsmith:
        # Belt and suspenders: unset both the flag and the key so the LangSmith
        # callback never attaches in this process.
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        os.environ.pop("LANGCHAIN_API_KEY", None)

    # Configure the Azure Monitor OpenTelemetry pipeline (traces -> App Insights).
    from azure.monitor.opentelemetry import configure_azure_monitor

    configure_azure_monitor(connection_string=connection_string)

    # Emit OpenTelemetry spans for every LangChain / LangGraph run.
    from openinference.instrumentation.langchain import LangChainInstrumentor

    LangChainInstrumentor().instrument()
