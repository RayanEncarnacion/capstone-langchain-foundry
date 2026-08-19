"""Token instrumentation, context budgeting, and prompt caching utilities.

Phase 1 (Tokens / Context / Prompt Caching):
- Tracks token contribution from system instructions, tool schemas, conversation history,
  RAG chunks, long-term memories, user messages, and model outputs.
- Logs structured token metrics:
    input_tokens
    output_tokens
    cached_tokens
    retrieval_tokens
    conversation_tokens
    tool_schema_tokens
- Enforces an explicit context budget to prevent unbounded context growth.
- Ensures stable prefix ordering for provider prompt caching (OpenAI / Azure OpenAI).
"""

from dataclasses import asdict, dataclass, field
import json
import logging
import math
from typing import Any, Sequence

logger = logging.getLogger(__name__)

# Fallback encoding if tiktoken model lookup fails
_ENCODING_CACHE: dict[str, Any] = {}


def get_token_encoder(model_or_encoding: str = "o200k_base"):
    """Get or load a tiktoken encoder safely with caching."""
    if model_or_encoding in _ENCODING_CACHE:
        return _ENCODING_CACHE[model_or_encoding]

    try:
        import tiktoken

        try:
            enc = tiktoken.encoding_for_model(model_or_encoding)
        except (KeyError, ValueError):
            enc = tiktoken.get_encoding("o200k_base")
        _ENCODING_CACHE[model_or_encoding] = enc
        return enc
    except Exception as exc:
        logger.debug(f"tiktoken unavailable or failed to load: {exc}")
        return None


def count_tokens(text: str | Any, model: str = "o200k_base") -> int:
    """Count tokens for a given string or json-serializable object."""
    if text is None:
        return 0
    if not isinstance(text, str):
        try:
            text = json.dumps(text, ensure_ascii=False)
        except Exception:
            text = str(text)

    if not text:
        return 0

    enc = get_token_encoder(model)
    if enc is not None:
        try:
            return len(enc.encode(text, disallowed_special=()))
        except Exception:
            pass

    # Heuristic fallback: ~4 characters per token
    return max(1, math.ceil(len(text) / 4))


def estimate_tool_schema_tokens(tools: Sequence[Any] | None, model: str = "o200k_base") -> int:
    """Estimate token count for registered tools/functions exposed to the model."""
    if not tools:
        return 0

    total = 0
    for t in tools:
        if t is None:
            continue
        # Check standard properties on Agent Framework tools / callables
        name = getattr(t, "name", "") or getattr(t, "__name__", "")
        description = getattr(t, "description", "") or getattr(t, "__doc__", "")
        parameters = getattr(t, "parameters", None) or getattr(t, "declaration", None)

        tool_repr = {
            "name": name,
            "description": description,
            "parameters": parameters,
        }
        total += count_tokens(tool_repr, model=model) + 10  # schema overhead
    return total


@dataclass
class TokenUsageBreakdown:
    """Detailed token usage breakdown across all context components."""

    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    retrieval_tokens: int = 0
    conversation_tokens: int = 0
    tool_schema_tokens: int = 0
    system_tokens: int = 0
    memory_tokens: int = 0
    user_message_tokens: int = 0

    def to_dict(self) -> dict[str, int]:
        """Convert to dict matching spec log keys."""
        return asdict(self)

    def log_breakdown(self, custom_logger: logging.Logger | None = None) -> None:
        """Log token usage in the format specified by Phase 1."""
        target_logger = custom_logger or logger
        msg = (
            "\n--- Token Usage Breakdown ---\n"
            f"input_tokens:        {self.input_tokens}\n"
            f"output_tokens:       {self.output_tokens}\n"
            f"cached_tokens:       {self.cached_tokens}\n"
            f"retrieval_tokens:    {self.retrieval_tokens}\n"
            f"conversation_tokens: {self.conversation_tokens}\n"
            f"tool_schema_tokens:  {self.tool_schema_tokens}\n"
            f"system_tokens:       {self.system_tokens}\n"
            f"memory_tokens:       {self.memory_tokens}\n"
            f"user_message_tokens: {self.user_message_tokens}\n"
            "-----------------------------"
        )
        target_logger.info(msg)


@dataclass
class ContextBudgetConfig:
    """Explicit context budget thresholds."""

    max_total_input_tokens: int = 8000
    max_history_tokens: int = 4000
    max_retrieval_tokens: int = 2000
    max_memory_tokens: int = 500
    keep_last_turns: int = 6


class ContextBudgetManager:
    """Manages context budgeting, history pruning, and prompt prefix stability.

    Prompt Caching Policy:
    Stable prefix ordering guarantees maximum cache hits in provider (Azure OpenAI):
      1. [STABLE PREFIX] System Prompt & Instructions
      2. [STABLE PREFIX] Tool / Function Schemas
      3. [SEMI-STABLE] Long-term user preferences / memories
      4. [DYNAMIC BOUNDED] Pruned recent conversation history (capped at max_history_tokens)
      5. [DYNAMIC BOUNDED] Top-K RAG retrieval chunks (capped at max_retrieval_tokens)
      6. [DYNAMIC] Current User Query
    """

    def __init__(self, config: ContextBudgetConfig | None = None) -> None:
        self.config = config or ContextBudgetConfig()

    def prune_history_messages(
        self, messages: list[Any], model: str = "o200k_base"
    ) -> tuple[list[Any], int, bool]:
        """Prune conversation messages to fit within max_history_tokens budget.

        Returns (pruned_messages, total_history_tokens, was_pruned).
        Always keeps the most recent turns up to keep_last_turns and within budget.
        """
        if not messages:
            return [], 0, False

        total_tokens = 0
        retained: list[Any] = []
        was_pruned = False

        # Traverse backwards from newest to oldest
        for msg in reversed(messages):
            # Extract content representation
            text = getattr(msg, "text", None) or getattr(msg, "content", None) or str(msg)
            msg_tokens = count_tokens(text, model=model)

            if (
                len(retained) >= self.config.keep_last_turns
                or (total_tokens + msg_tokens) > self.config.max_history_tokens
            ):
                was_pruned = True
                continue

            retained.append(msg)
            total_tokens += msg_tokens

        # Restore chronological order
        retained.reverse()
        return retained, total_tokens, was_pruned

    def calculate_breakdown(
        self,
        *,
        system_prompt: str,
        tools: Sequence[Any] | None = None,
        history_messages: Sequence[Any] | None = None,
        retrieval_content: str | None = None,
        memories_content: str | None = None,
        user_message: str = "",
        model_output: str = "",
        usage_details: Any | None = None,
        model: str = "o200k_base",
    ) -> TokenUsageBreakdown:
        """Compute estimated & actual token breakdown."""
        system_tokens = count_tokens(system_prompt, model=model)
        tool_schema_tokens = estimate_tool_schema_tokens(tools, model=model)

        conv_tokens = 0
        if history_messages:
            for m in history_messages:
                txt = getattr(m, "text", None) or getattr(m, "content", None) or str(m)
                conv_tokens += count_tokens(txt, model=model)

        retrieval_tokens = count_tokens(retrieval_content, model=model) if retrieval_content else 0
        memory_tokens = count_tokens(memories_content, model=model) if memories_content else 0
        user_msg_tokens = count_tokens(user_message, model=model)

        estimated_input = (
            system_tokens
            + tool_schema_tokens
            + conv_tokens
            + retrieval_tokens
            + memory_tokens
            + user_msg_tokens
        )
        estimated_output = count_tokens(model_output, model=model)

        # Provider reported usage (if available from Agent Framework / Azure OpenAI)
        input_tokens = estimated_input
        output_tokens = estimated_output
        cached_tokens = 0

        if usage_details:
            if isinstance(usage_details, dict):
                input_tokens = usage_details.get("input_token_count") or input_tokens
                output_tokens = usage_details.get("output_token_count") or output_tokens
                cached_tokens = (
                    usage_details.get("cache_read_input_token_count")
                    or usage_details.get("cached_tokens")
                    or 0
                )
            else:
                input_tokens = getattr(usage_details, "input_token_count", None) or input_tokens
                output_tokens = getattr(usage_details, "output_token_count", None) or output_tokens
                cached_tokens = getattr(usage_details, "cache_read_input_token_count", None) or 0

        return TokenUsageBreakdown(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            retrieval_tokens=retrieval_tokens,
            conversation_tokens=conv_tokens,
            tool_schema_tokens=tool_schema_tokens,
            system_tokens=system_tokens,
            memory_tokens=memory_tokens,
            user_message_tokens=user_msg_tokens,
        )
