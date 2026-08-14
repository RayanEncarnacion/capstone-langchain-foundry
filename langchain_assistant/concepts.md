## LangChain and LangGraph Project

In this version, LangChain supplies the model interface, tools, structured output, middleware and agent API. LangGraph supplies the stateful runtime underneath the agent. Azure supplies the model, object storage, search, database and identity services.

The model can be hosted in Microsoft Foundry, but orchestration must remain independent: do not import Microsoft Agent Framework or the Foundry agent SDK in this implementation.

### Concepts covered

| Concept | How it is used | Importance | Recommendation |
|---|---|---|---|
| Models and messages | Call an Azure-hosted model through `langchain-openai`. | Essential | Make a direct model call before adding tools or memory. |
| Prompt design | Define role, citation rules, tool boundaries and abstention behavior. | Essential | Use one concise system prompt stored in source control. |
| Structured output | Return a validated Pydantic `AssistantResponse`. | High | Prefer schema enforcement over parsing free-form text. |
| RAG ingestion | Load, split, embed and index the note files. | Essential | Preserve source, title, chunk ID and document version. |
| Retrieval | Compare keyword, vector, hybrid and semantic-reranked results. | Essential | Use hybrid plus semantic reranking as the final baseline. |
| Tool calling | Expose search and task operations with `@tool`. | Essential | Use narrow input schemas and clear docstrings. |
| Agent orchestration | Use `create_agent` for the model-tool loop. | Essential | Do not hand-build a graph until the ordinary loop is insufficient. |
| Short-term memory | Save state under a `thread_id` with a checkpointer. | Essential | Conversation history belongs to a thread, not long-term memory. |
| Long-term memory | Save explicit user preferences in a namespaced Store. | High | Remember only useful, explicit facts rather than full chats. |
| Human approval | Pause before creating or completing a task. | High | Use HITL middleware; prompt instructions alone are not enforcement. |
| Guardrails | Validate input, tool arguments, PII and execution limits. | High | Use deterministic checks before model-based safety checks. |
| Observability | Inspect model, retrieval, tool and middleware runs in LangSmith. | Essential | Create a test dataset before extensive prompt tuning. |