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

### Phase 2 — Agent, session and response schema

Build
- Create an ephemeral Agent Framework Agent.
- Keep instructions in the Python project.
- Reuse the LangChain implementation's Pydantic response schema.
- Create and reuse an AgentSession.

Finish when
- A follow-up works only when the same session is supplied.
- The response shape matches the LangChain API.
- The agent definition is versioned with the application code.