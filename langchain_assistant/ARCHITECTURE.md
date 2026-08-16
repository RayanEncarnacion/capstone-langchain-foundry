# LangChain Assistant Architecture

This folder contains the current local study assistant implementation. It uses
LangChain and LangGraph for orchestration, while Azure services provide model
access, retrieval storage, task storage, identity-adjacent auth, content safety,
and optional Foundry/Application Insights tracing.

The next implementation should rebuild the same assistant behavior with the new
Microsoft Foundry project endpoint and Microsoft Agent Framework. Keep the app
local, but move project-scoped model access, managed knowledge, observability,
evaluation, and governance into Foundry.

## Current-Stack Warning

This code is intentionally a LangChain implementation. It calls a Foundry-hosted
deployment through the OpenAI-compatible `/openai/v1` route using
`langchain_openai.ChatOpenAI`.

For the Foundry rebuild, do not copy this model wiring as-is. Use the new
Foundry UI, the Foundry project endpoint, the project-scoped Responses API, and
`FoundryChatClient`. Use Microsoft Agent Framework for orchestration instead of
LangChain `create_agent` and LangGraph middleware.

## Runtime Shape

```text
Client or script
    |
    | POST /chat or POST /approve
    v
Local FastAPI app
    |
    | auth, content safety, response schema validation
    v
LangChain agent graph
    |
    | chooses model-only answer or tool call
    v
+-------------------------+--------------------------+
| Chat model              | Tools                    |
| Foundry OpenAI v1 route | search_notes             |
| via ChatOpenAI          | list_tasks               |
|                         | create_task              |
|                         | set_preference           |
|                         | get_preference           |
+-------------------------+--------------------------+
    |                         |
    |                         +--> Azure AI Search for note chunks
    |                         +--> Azure Cosmos DB for tasks
    |                         +--> CosmosDBStore for preferences
    |
    +--> CosmosDBSaver for per-thread graph checkpoints
    +--> LangSmith tracing by default
    +--> optional Foundry/App Insights tracing runner
```

## Entry Points

`app/api.py` is the local service boundary. It exposes:

- `GET /health` for unauthenticated liveness.
- `POST /chat` for one agent turn.
- `POST /approve` to resume a run paused by human approval middleware.

The API owns trust-boundary work:

- Derives `user_id` from `Authorization: Bearer <token>` through `app/auth.py`.
- Rejects anonymous or unknown tokens with HTTP 401.
- Screens inbound messages and outbound replies through `app/content_safety.py`.
- Validates request and response JSON with `app/schemas.py`.
- Creates a new `thread_id` when the caller does not provide one.
- Converts a paused write-tool interrupt into `approval_required=true`.

`scripts/call_api.py` is a tiny local client for exercising `POST /chat`.

## Agent Orchestration

`app/agent.py` builds the LangChain agent.

Current responsibilities:

- Reads Foundry OpenAI-compatible endpoint settings from environment:
  `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT`, and optional
  `AZURE_OPENAI_API_KEY`.
- Uses `DefaultAzureCredential` when no API key is present.
- Builds `ChatOpenAI` with deterministic temperature, timeout, and bounded
  retries.
- Defines the system prompt for tool routing, grounding, citation behavior,
  prompt-injection handling, tool error handling, and durable preference memory.
- Binds all tools from `app/tools.py`.
- Installs LangChain middleware:
  - `ToolCallLimitMiddleware` caps tool calls per run.
  - `ModelCallLimitMiddleware` caps model calls per run.
  - `HumanInTheLoopMiddleware` pauses before `create_task`.
- Provides `run_agent()` for normal turns and `resume_agent()` for approval
  decisions.
- Summarizes raw LangGraph output into API-friendly `reply`, `tool_calls`, and
  `pending` values.

Important invariant: `thread_id` and `user_id` are passed through the run config,
not through model-visible user input. The model cannot choose another user.

## Tool Contract

`app/tools.py` is the model's only route to external state.

Current tools:

- `search_notes(query, top_k)` returns small, citable snippets from Azure AI
  Search.
- `list_tasks(status)` reads the authenticated user's tasks from Cosmos DB.
- `create_task(title, due_date)` writes a pending task and is gated by human
  approval.
- `set_preference(key, value)` stores a durable user preference.
- `get_preference(key)` reads a durable user preference.

Tool design rules already encoded in the implementation:

- Narrow Pydantic argument schemas.
- Small structured dict outputs.
- Caught exceptions returned as `{"ok": false, "error": ...}`.
- User scoping from `config["configurable"]["user_id"]`.
- Retrieved note snippets wrapped as untrusted content.

For the Foundry rebuild, preserve tool names and behavior where practical. That
keeps the assistant contract stable while swapping orchestration from LangChain
tools to Microsoft Agent Framework tools.

## Retrieval And Knowledge

Current knowledge pipeline is custom RAG:

- Source notes live in `seed_data/*.md`.
- `scripts/ingest.py` can upload those notes into Azure Blob Storage.
- The ingest pipeline loads markdown from Blob, splits documents into chunks,
  stamps citation metadata, embeds chunks, and indexes them in Azure AI Search.
- `app/retrieval.py` embeds each question and performs hybrid search
  (`search_text` plus vector query) over the Search index.
- Retrieved chunks carry `source`, `title`, `chunk_id`, and score metadata.
- The agent prompt requires answers to cite source and `chunk_id`, or abstain
  when snippets do not support an answer.

`scripts/compare_retrieval.py` compares keyword, vector, hybrid, and
hybrid-plus-semantic retrieval modes for the phase 3 baseline.

For the Foundry rebuild, this is the main area to replace with managed
knowledge. Keep the corpus semantics, citation requirement, abstention behavior,
and retrieval evaluation cases. Replace direct Blob/Search/embedding wiring when
Foundry managed knowledge can own ingestion, retrieval, citations,
observability, and governance.

## Persistence And Memory

`app/storage.py` centralizes Azure clients and environment-driven settings.

Current stores:

- Azure Blob Storage for raw note corpus.
- Azure AI Search for chunk index and hybrid retrieval.
- Foundry OpenAI-compatible embeddings deployment for vectors.
- Azure Cosmos DB `tasks` container partitioned by `user_id`.
- `CosmosDBSaverSync` for per-thread agent checkpoints.
- `CosmosDBStore` for cross-thread, per-user long-term preferences.

Short-term memory is conversation state under `thread_id`. Long-term memory is
only explicit preferences, namespaced by authenticated `user_id`.

For the Foundry rebuild, decide which persistence remains app-owned:

- App task records likely stay in local/app data storage unless Foundry tool
  state becomes a better fit.
- Conversation state may move to Microsoft Agent Framework runtime mechanisms.
- Knowledge index state should move to Foundry managed knowledge if possible.

## Auth, Approval, And Safety

Current auth is a local bearer-token simulation in `app/auth.py`:

- `API_KEYS` maps opaque bearer tokens to user ids.
- The shape matches Entra bearer auth, but no real JWT validation is implemented.
- Swapping to real Entra auth means replacing `_resolve_user_id()` with issuer,
  audience, signature, and claim validation.

Current approval and guardrails:

- `create_task` is the only write tool requiring approval.
- Approval resumes the same interrupted thread through `/approve`.
- Tool and model calls are capped per run.
- Input and output text are screened with Azure AI Content Safety when
  configured.
- Content Safety fails open when unconfigured or temporarily unavailable.
- Retrieved notes are treated as untrusted data in both prompt rules and tool
  output formatting.

For the Foundry rebuild, preserve approval semantics and user scoping. Move
governance that Foundry can own into the project configuration, but keep local
API enforcement for request auth and task ownership.

## Observability And Evaluation

Current observability is split:

- Normal development uses LangSmith tracing through environment variables.
- `app/observability.py` and `scripts/run_with_otel.py` can run a separate
  process that emits OpenTelemetry spans to Foundry/Application Insights.
- The optional Foundry path disables LangSmith in that process to avoid duplicate
  traces.

Current evaluation:

- `DATASET.md` documents the shared test cases.
- `eval/examples.json` is the machine-readable single-turn dataset.
- `eval/dataset.py` syncs examples to LangSmith.
- `eval/target.py` invokes the compiled agent directly and captures answer,
  tools called, tool outputs, retrieved chunk ids, abstention, and approval
  state.
- `eval/evaluators.py` defines deterministic checks for tool selection,
  abstention, and citations, plus optional LLM judges for groundedness and
  relevance.
- `eval/run_eval.py` runs LangSmith experiments.

For the Foundry rebuild, migrate evaluation and tracing into Foundry project
observability/evaluation. Keep the same dataset intent and evaluator categories
so results remain comparable across stacks.

## Environment Boundary

Current `.env.langchain.example` configures:

- Foundry OpenAI-compatible chat and embedding deployments.
- Blob Storage, Azure AI Search, and Cosmos DB.
- Dev bearer-token auth.
- Azure AI Content Safety.
- LangSmith tracing.
- Optional Foundry/Application Insights tracing target.

`.env.foundry.example` is currently empty. For the Foundry implementation, it
should become the project-scoped local config file. Expected direction:

- Foundry project endpoint, not only the OpenAI-compatible `/openai/v1` endpoint.
- Model deployment name used through project-scoped Responses API /
  `FoundryChatClient`.
- Managed knowledge identifiers or connection names.
- Foundry observability/evaluation settings.
- Local app-only settings such as API auth tokens and task-store connection.

Do not put real secrets in either example file.

## Current Module Map

| Path | Responsibility |
| --- | --- |
| `app/api.py` | FastAPI app, auth dependency usage, content safety boundary, approval response shaping |
| `app/agent.py` | LangChain model wiring, system prompt, middleware stack, run/resume helpers |
| `app/tools.py` | Model-callable tools for notes, tasks, and preferences |
| `app/retrieval.py` | Runtime hybrid retrieval over Azure AI Search |
| `app/storage.py` | Azure Blob, Search, Cosmos, embedding, checkpoint, and store clients |
| `app/auth.py` | Dev bearer token to user id mapping |
| `app/content_safety.py` | Azure AI Content Safety wrapper |
| `app/observability.py` | Optional Foundry/Application Insights tracing setup |
| `app/schemas.py` | API request/response and data schemas |
| `scripts/ingest.py` | Corpus upload, chunking, embedding, and index creation |
| `scripts/compare_retrieval.py` | Retrieval strategy comparison |
| `scripts/run_with_otel.py` | One-turn run with Foundry/Application Insights tracing |
| `scripts/call_api.py` | Local API smoke-test client |
| `eval/*` | LangSmith dataset sync, target function, and evaluators |
| `seed_data/*` | Synthetic markdown note corpus |
| `tests/test_api_e2e.py` (repo root) | End-to-end API tests: RAG search path, note-scoped abstention, and 401 auth boundary (run against live services) |

## Foundry Rebuild Reuse Plan

Keep:

- Local FastAPI service shape: `/chat`, `/approve`, `/health`.
- Request/response schemas unless Microsoft Agent Framework requires a small
  adapter.
- User identity invariant: derive user id from auth, never request body or model.
- Tool behavior and small structured outputs.
- Human approval before task writes.
- Citation and abstention behavior.
- Prompt-injection stance: retrieved knowledge is data, not instructions.
- Dataset categories: grounded answer, abstention, tool choice, approval,
  memory, and security.

Replace:

- `ChatOpenAI` model construction with project-scoped Responses API access via
  `FoundryChatClient`.
- LangChain `create_agent` and LangGraph middleware with Microsoft Agent
  Framework orchestration.
- LangChain `@tool` bindings with Microsoft Agent Framework tool definitions.
- Custom RAG plumbing where Foundry managed knowledge can provide ingestion,
  retrieval, citations, observability, and governance.
- LangSmith-first tracing/evaluation with Foundry project observability and
  evaluation.

Revisit:

- Whether Cosmos-backed checkpoints are still needed once Microsoft Agent
  Framework owns conversation state.
- Whether preferences remain in CosmosDBStore or move to an app-owned simpler
  store.
- Whether Content Safety remains local, moves into Foundry project governance,
  or uses both layers.
- Whether local eval scripts become thin Foundry eval launchers.

## Minimal Foundry Target Architecture

```text
Client or script
    |
    v
Local FastAPI app
    |
    | auth, local safety checks, approval workflow, API schema
    v
Microsoft Agent Framework agent
    |
    +--> FoundryChatClient
    |       |
    |       +--> Foundry project endpoint
    |       +--> project-scoped Responses API
    |
    +--> Foundry managed knowledge
    |
    +--> app tools
    |       +--> task store
    |       +--> preference store
    |
    +--> Foundry tracing, evaluation, governance
```

This target keeps the assistant local and familiar to callers, but changes the
control plane: Foundry becomes the project-scoped home for model access,
knowledge, telemetry, evaluation, and governance.
