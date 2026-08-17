## Phase 1 — Foundry baseline

Build
- Create or select a project in the new Microsoft Foundry UI.
- Deploy a small model that supports tools and structured output.
- Connect an Application Insights resource.
- Call the project endpoint from a local FastAPI endpoint.

Tools
- Foundry portal, FoundryChatClient, Agent Framework, Azure Identity, FastAPI.

Finish when
- The local API returns a model response.
- The call appears in the Foundry trace view.
- Authentication uses your Entra development identity.

## Phase 2 — Agent, session and response schema

Build
- Create an ephemeral Agent Framework Agent.
- Keep instructions in the Python project.
- Reuse the LangChain implementation's Pydantic response schema.
- Create and reuse an AgentSession.

Finish when
- A follow-up works only when the same session is supplied.
- The response shape matches the LangChain API.
- The agent definition is versioned with the application code.

## Phase 3 — Managed RAG with Foundry IQ

Build
- Create a Foundry IQ knowledge base over the note corpus.
- Use Azure AI Search as the managed retrieval layer.
- Connect the knowledge base through its MCP endpoint.
- Allow only the knowledge_base_retrieve tool.
- Require source-backed answers and abstention.

Tools
- Foundry IQ, Azure AI Search, MCPStreamableHTTPTool, project identity and RBAC.

Finish when
- The agent calls the knowledge-base tool for grounded questions.
- Answers contain useful citations.
- Out-of-scope questions produce an abstention.