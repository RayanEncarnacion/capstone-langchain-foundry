"""Phase 7 — LangSmith tracing and evaluation.

Package contents:
  - dataset.py     : the shared single-turn example set + LangSmith sync.
  - target.py      : runs the agent on one example and captures what the
                     evaluators need (answer, tools, tool outputs, abstention).
  - evaluators.py  : deterministic (citations, abstention, tool selection) and
                     LLM-judge (groundedness, relevance) evaluators.
  - run_eval.py    : entry point that syncs the dataset and runs evaluate().
"""
