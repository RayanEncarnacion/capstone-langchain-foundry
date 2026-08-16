"""Evaluators for Phase 7.

Two families:

Deterministic (no LLM, cheap, reproducible):
  - tool_selection : did the expected tool fire?
  - abstention     : did the agent abstain exactly when it should?
  - citations      : does a grounded answer cite a chunk it actually retrieved?

LLM judges (run after the deterministic checks work):
  - groundedness   : is the answer supported by the retrieved tool output?
  - relevance      : does the answer actually address the question?

Each function follows the LangSmith evaluator signature (keyword params chosen
from inputs / outputs / reference_outputs) and returns a feedback dict.
"""

import json

from ..app.agent import build_chat_model

# ---------------------------------------------------------------------------
# Deterministic evaluators
# ---------------------------------------------------------------------------


def tool_selection(outputs: dict, reference_outputs: dict) -> dict:
    """Score 1 when the expected tool fired (or no tool when none expected)."""
    expected = (reference_outputs or {}).get("expected_tool")
    called = (outputs or {}).get("tools_called", []) or []

    if expected in (None, "", "none"):
        score = len(called) == 0
        comment = f"expected no tool; called={called}"
    else:
        score = expected in called
        comment = f"expected={expected}; called={called}"

    return {"key": "tool_selection", "score": bool(score), "comment": comment}


def abstention(outputs: dict, reference_outputs: dict) -> dict:
    """Score 1 when the abstention behavior matches the expectation."""
    expected = bool((reference_outputs or {}).get("expected_abstain"))
    got = bool((outputs or {}).get("abstained"))
    return {
        "key": "abstention",
        "score": expected == got,
        "comment": f"expected_abstain={expected}; abstained={got}",
    }


def citations(outputs: dict, reference_outputs: dict) -> dict:
    """Score 1 when a grounded answer cites a chunk it actually retrieved.

    Only meaningful for search-backed, non-abstention answers; returns a null
    score (n/a) otherwise so it doesn't drag the aggregate down.
    """
    ref = reference_outputs or {}
    out = outputs or {}

    if ref.get("expected_abstain") or ref.get("expected_tool") != "search_notes":
        return {"key": "citations", "score": None, "comment": "n/a for this case"}
    if out.get("abstained"):
        return {"key": "citations", "score": None, "comment": "n/a (abstained)"}

    answer = out.get("answer", "") or ""
    retrieved_ids = out.get("retrieved_ids", []) or []
    if not retrieved_ids:
        return {
            "key": "citations",
            "score": False,
            "comment": "no chunks retrieved to cite",
        }

    hit = [cid for cid in retrieved_ids if cid and cid in answer]
    return {
        "key": "citations",
        "score": bool(hit),
        "comment": f"cited={hit or 'none'}; retrieved={retrieved_ids}",
    }


# ---------------------------------------------------------------------------
# LLM judges
# ---------------------------------------------------------------------------
_judge = None


def _get_judge():
    global _judge
    if _judge is None:
        _judge = build_chat_model()
    return _judge


def _judge_json(system: str, user: str) -> dict:
    """Ask the judge for a strict {score, reason} JSON verdict."""
    from langchain_core.messages import HumanMessage, SystemMessage

    reply = _get_judge().invoke(
        [SystemMessage(content=system), HumanMessage(content=user)]
    ).content.strip()

    # Be tolerant of code fences / stray prose around the JSON object.
    start, end = reply.find("{"), reply.rfind("}")
    if start != -1 and end != -1:
        try:
            return json.loads(reply[start : end + 1])
        except json.JSONDecodeError:
            pass
    return {"score": 0, "reason": f"unparseable judge reply: {reply[:200]}"}


_GROUNDED_SYSTEM = (
    "You are a strict evaluator. Decide whether the ANSWER is fully supported "
    "by the CONTEXT (retrieved note snippets). Reply with JSON only: "
    '{"score": <0 or 1>, "reason": "<short>"}. Score 1 only if every factual '
    "claim in the answer is backed by the context. If the answer abstains or "
    "says it lacks information, score 1 (correctly grounded)."
)

_RELEVANCE_SYSTEM = (
    "You are a strict evaluator. Decide whether the ANSWER actually addresses "
    "the QUESTION. Reply with JSON only: "
    '{"score": <0 or 1>, "reason": "<short>"}. A correct abstention for an '
    "unanswerable question is relevant (score 1)."
)


def groundedness(inputs: dict, outputs: dict) -> dict:
    """LLM judge: is the answer supported by the retrieved context?"""
    out = outputs or {}
    context_parts = [
        str(o.get("content", "")) for o in out.get("tool_outputs", []) or []
    ]
    context = "\n\n".join(context_parts) or "(no tool output)"
    user = (
        f"CONTEXT:\n{context}\n\n"
        f"ANSWER:\n{out.get('answer', '')}"
    )
    verdict = _judge_json(_GROUNDED_SYSTEM, user)
    return {
        "key": "groundedness",
        "score": float(verdict.get("score", 0)),
        "comment": verdict.get("reason", ""),
    }


def relevance(inputs: dict, outputs: dict) -> dict:
    """LLM judge: does the answer address the question?"""
    user = (
        f"QUESTION:\n{(inputs or {}).get('question', '')}\n\n"
        f"ANSWER:\n{(outputs or {}).get('answer', '')}"
    )
    verdict = _judge_json(_RELEVANCE_SYSTEM, user)
    return {
        "key": "relevance",
        "score": float(verdict.get("score", 0)),
        "comment": verdict.get("reason", ""),
    }


# Grouped for convenience in run_eval.py.
DETERMINISTIC_EVALUATORS = [tool_selection, abstention, citations]
LLM_EVALUATORS = [groundedness, relevance]
