"""Unit and integration tests for Phase 1: Tokens / Context / Prompt Caching."""

import logging
from unittest.mock import MagicMock
import pytest

from foundry_assistant.app.schemas import ChatResponse, TokenUsage, ToolCall
from foundry_assistant.app.tokens import (
    ContextBudgetConfig,
    ContextBudgetManager,
    TokenUsageBreakdown,
    count_tokens,
    estimate_tool_schema_tokens,
)


def test_count_tokens_basic():
    """Verify count_tokens calculates non-zero positive token counts."""
    text = "You are a helpful AI assistant for the Northstar learning program."
    tokens = count_tokens(text)
    assert tokens > 0
    assert tokens < 50

    empty_tokens = count_tokens("")
    assert empty_tokens == 0

    none_tokens = count_tokens(None)
    assert none_tokens == 0


def test_estimate_tool_schema_tokens():
    """Verify tool schema tokens are estimated accurately."""
    mock_tool = MagicMock()
    mock_tool.name = "knowledge_base_retrieve"
    mock_tool.description = "Retrieve grounded passages from the study note index."
    mock_tool.parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"}
        },
        "required": ["query"],
    }

    tokens = estimate_tool_schema_tokens([mock_tool])
    assert tokens > 15


def test_token_usage_breakdown_logging(caplog):
    """Verify token breakdown records all required Phase 1 fields and formats log output."""
    breakdown = TokenUsageBreakdown(
        input_tokens=1500,
        output_tokens=120,
        cached_tokens=1024,
        retrieval_tokens=300,
        conversation_tokens=450,
        tool_schema_tokens=250,
        system_tokens=200,
        memory_tokens=50,
        user_message_tokens=130,
    )

    data = breakdown.to_dict()
    assert data["input_tokens"] == 1500
    assert data["output_tokens"] == 120
    assert data["cached_tokens"] == 1024
    assert data["retrieval_tokens"] == 300
    assert data["conversation_tokens"] == 450
    assert data["tool_schema_tokens"] == 250

    with caplog.at_level(logging.INFO):
        test_logger = logging.getLogger("test_breakdown")
        breakdown.log_breakdown(custom_logger=test_logger)

    assert "input_tokens:        1500" in caplog.text
    assert "cached_tokens:       1024" in caplog.text
    assert "retrieval_tokens:    300" in caplog.text


def test_context_budget_history_pruning():
    """Verify ContextBudgetManager prunes older messages when budget is exceeded."""
    config = ContextBudgetConfig(
        max_history_tokens=100,
        keep_last_turns=4,
    )
    mgr = ContextBudgetManager(config=config)

    # 10 synthetic messages with ~30 tokens each
    messages = [
        MagicMock(text=f"Turn {i}: This is a sample conversation message with several words.")
        for i in range(10)
    ]

    pruned, total_tokens, was_pruned = mgr.prune_history_messages(messages)
    assert was_pruned is True
    assert len(pruned) <= 4
    assert total_tokens <= 100


def test_calculate_breakdown_with_provider_usage():
    """Verify breakdown extraction reconciles provider usage with component estimates."""
    mgr = ContextBudgetManager()

    mock_usage = {
        "input_token_count": 2048,
        "output_token_count": 95,
        "cache_read_input_token_count": 1024,
    }

    breakdown = mgr.calculate_breakdown(
        system_prompt="You are a study assistant.",
        tools=[],
        history_messages=[],
        retrieval_content="Document passage excerpt.",
        user_message="Summarize lesson 1.",
        model_output="Here is the summary.",
        usage_details=mock_usage,
    )

    assert breakdown.input_tokens == 2048
    assert breakdown.output_tokens == 95
    assert breakdown.cached_tokens == 1024
    assert breakdown.system_tokens > 0
    assert breakdown.retrieval_tokens > 0
    assert breakdown.user_message_tokens > 0


def test_chat_response_schema_with_usage():
    """Verify ChatResponse model serializes optional TokenUsage properly."""
    usage = TokenUsage(
        input_tokens=1200,
        output_tokens=80,
        cached_tokens=1024,
        retrieval_tokens=200,
        conversation_tokens=150,
        tool_schema_tokens=100,
    )
    resp = ChatResponse(
        session_id="session-123",
        message="Hello",
        tool_calls=[ToolCall(name="test_tool", args={"q": "1"})],
        usage=usage,
    )

    dumped = resp.model_dump()
    assert dumped["session_id"] == "session-123"
    assert dumped["usage"]["cached_tokens"] == 1024
    assert dumped["usage"]["input_tokens"] == 1200
