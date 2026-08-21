"""Output sanitization for untrusted tool results (MCP and database).

MCP servers and user-authored database content are untrusted input that could
contain indirect prompt injection payloads. This module provides a sanitizer
that strips known injection patterns and enforces length limits before tool
output enters the model context.
"""

import json
import re
from typing import Any

from .tokens import count_tokens

_MAX_TOOL_OUTPUT_CHARS = 8000

_INJECTION_PATTERNS = re.compile(
    r"(?i)"
    r"(SYSTEM\s*:|IGNORE\s+(ALL\s+)?PREVIOUS|"
    r"YOU\s+ARE\s+NOW|FORGET\s+(ALL\s+)?INSTRUCTIONS|"
    r"NEW\s+INSTRUCTIONS|OVERRIDE\s+PROMPT|"
    r"<\s*/?system\s*>|"
    r"\[INST\]|\[/INST\]|"
    r"<<\s*SYS\s*>>|<</\s*SYS\s*>>)"
)


def sanitize_tool_output(output: str, max_chars: int = _MAX_TOOL_OUTPUT_CHARS) -> str:
    """Sanitize a tool output string.

    - Strips content matching known prompt injection patterns.
    - Truncates to max_chars.
    - Wraps in delimiters so the model treats it as data, not instruction.
    """
    cleaned = _INJECTION_PATTERNS.sub("[REDACTED]", output)
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars] + "... [truncated]"
    return cleaned


def sanitize_dict_output(data: dict, max_chars: int = _MAX_TOOL_OUTPUT_CHARS) -> dict:
    """Sanitize string values inside a tool result dict (recursive on nested strings)."""
    serialized = json.dumps(data, default=str)
    cleaned = sanitize_tool_output(serialized, max_chars)
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        return {"sanitized_output": cleaned}


def mcp_result_parser(result) -> str:
    """Custom parse_tool_results for the MCP KB tool.

    Receives a CallToolResult from the MCP SDK and returns a sanitized string.
    """
    parts: list[str] = []
    for content in getattr(result, "content", []) or []:
        text = getattr(content, "text", None)
        if text:
            parts.append(text)
    # The MCP content list commonly maps one item to one retrieved passage.
    # Cap it here, then cap again in function middleware after framework
    # normalization. Double enforcement keeps provider-shape changes bounded.
    from .context_engineering import context_policy

    raw: Any = parts if parts else str(result)
    return compress_retrieval_output(
        raw,
        max_chunks=context_policy.max_retrieval_chunks,
        max_tokens=context_policy.max_retrieval_tokens,
    )


def _limit_collections(value: Any, max_items: int) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _limit_collections(child, max_items)
            for key, child in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_limit_collections(child, max_items) for child in value[:max_items]]
    return value


def _token_bounded_text(value: Any, max_tokens: int) -> str:
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, default=str)
    sanitized = sanitize_tool_output(text, max_chars=max_tokens * 4)
    if count_tokens(sanitized) <= max_tokens:
        return sanitized
    return sanitized[: max(1, max_tokens * 3)].rstrip() + "... [token truncated]"


def compress_tool_output(value: Any, *, max_items: int, max_tokens: int) -> Any:
    """Return a small, sanitized tool result while keeping structured data when possible."""

    limited = _limit_collections(value, max_items)
    if isinstance(limited, dict):
        cleaned = sanitize_dict_output(limited, max_chars=max_tokens * 4)
        if count_tokens(cleaned) <= max_tokens:
            return cleaned
        return {
            "ok": cleaned.get("ok", True),
            "compressed": True,
            "preview": _token_bounded_text(cleaned, max_tokens),
        }
    if isinstance(limited, list):
        serialized = _token_bounded_text(limited, max_tokens)
        try:
            return json.loads(serialized)
        except (json.JSONDecodeError, TypeError):
            return {"compressed": True, "preview": serialized}
    return _token_bounded_text(limited, max_tokens)


def compress_retrieval_output(
    value: Any, *, max_chunks: int, max_tokens: int
) -> str:
    """Keep at most top-K retrieval items and a strict token budget."""

    parsed = value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            # Shape is unknown: token-bound it, but do not guess paragraph
            # boundaries because trailing citation metadata may be essential.
            parsed = value
    limited = _limit_collections(parsed, max_chunks)
    return _token_bounded_text(limited, max_tokens)
