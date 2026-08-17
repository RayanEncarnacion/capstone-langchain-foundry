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

## Phase 5 — Local Cosmos DB tools

Build
- Add list_tasks, create_task and complete_task.
- Describe tool parameters with Python annotations and Pydantic fields.
- Inject authenticated user identity through runtime context.
- Hide identity and partition information from the model-visible schema.

Tools
- Agent Framework @tool, invocation context, Azure Cosmos DB SDK.

Finish when
- The agent selects the right tool.
- The model cannot select another user's partition.
- Invalid arguments are rejected inside the tool.

## Phase 6 — Session and long-term memory

Build
- Use AgentSession for current conversation state.
- Add one cross-session preference using a Foundry Memory Store.
- Scope memory to a stable authenticated user identifier.
- If managed memory is unavailable, implement the same behavior with a Cosmos-backed ContextProvider.

Status: Foundry managed memory is currently preview.

Finish when
- Conversation history remains session-specific.
- An explicit preference survives a new session.
- The user can inspect and delete remembered preferences.