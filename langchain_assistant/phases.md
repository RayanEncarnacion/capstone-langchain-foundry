## Phase 1 — Model and API baseline

Build
- Create a local FastAPI application with POST /chat.
- Call the chat model directly without RAG, tools or memory.
- Validate the result with the shared Pydantic response schema.

Tools
- FastAPI, Uvicorn, Pydantic, langchain-openai, Azure Identity.

Finish when
- The endpoint returns typed JSON.
- Invalid output creates a visible validation error.
- No key or credential is hard-coded.

## Phase 2 — Explicit two-step RAG

Build
- Put the small note corpus in Azure Blob Storage.
- Load the notes into LangChain Document objects.
- Split them into chunks and add useful metadata.
- Generate embeddings and index the chunks in Azure AI Search.
- Always retrieve before generation and include citations.

Finish when
- An answer includes the source and chunk ID.
- An unsupported question produces an abstention.
- You can print the exact chunks given to the model.

## Phase 3 — Retrieval experiments

Build
- Choose five questions with different wording from the source text.
- Run keyword-only, vector-only, hybrid, and hybrid plus semantic queries.
- Record the top three chunks and scores for each approach.

Tools
- Azure AI Search Search Explorer and the Python Search SDK.

Finish when
- You can explain why keyword and vector results differ.
- You understand reciprocal rank fusion at a practical level.
- You select a retrieval baseline using evidence rather than intuition.

## Phase 4 — Agent and tools

Build
- Wrap retrieval as search_notes.
- Add list_tasks and create_task.
- Build the agent with create_agent.
- Keep tool results small and structured.

Tools
- LangChain @tool, create_agent, Pydantic tool arguments, Azure Cosmos DB SDK.

Finish when
- The model searches only when knowledge is needed.
- Task requests select the appropriate task tool.
- Tool errors are returned safely rather than hidden.

## Phase 5 — Memory and persistence

Build
- Persist agent state under a thread ID.
- Add a long-term Store namespaced by authenticated user ID.
- Remember one explicit preference such as preferred session duration.
- Keep tasks in a separate application-data container.

Tools
- CosmosDBSaver, CosmosDBStore, langchain-azure-cosmosdb.

Finish when
- A follow-up works in the same thread.
- A preference survives a new thread.
- A different user cannot retrieve that preference.