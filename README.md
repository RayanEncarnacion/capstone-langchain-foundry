## Grounded Learning & Action Assistant

Create a local API assistant that works over a very small collection of learning notes. Use 6–10 short synthetic Markdown files about any neutral subject. Generate the files quickly; writing source material is not part of the exercise.

### What the assistant does

- Answers questions from the notes.
- Cites the source chunks used.
- Abstains when evidence is insufficient.
- Understands follow-up questions.
- Remembers a few explicit user preferences.
- Lists, creates and completes study tasks.
- Requires approval before changing data.

### Architecture at a glance

```
Authenticated CLI request
        │
        ▼
Local FastAPI service
        │
        ▼
Agent ───────► Chat model
  │
  ├──────────► Knowledge retrieval ───► Azure AI Search
  │                                      ▲
  │                                      │
  │                                Azure Blob notes
  │
  ├──────────► Task tools ───────────► Azure Cosmos DB
  │
  ├──────────► Conversation state
  │
  ├──────────► Long-term user memory
  │
  └──────────► Tracing and evaluation
```

In this version, LangChain supplies the model interface, tools, structured output, middleware and agent API. LangGraph supplies the stateful runtime underneath the agent. Azure supplies the model, object storage, search, database and identity services.