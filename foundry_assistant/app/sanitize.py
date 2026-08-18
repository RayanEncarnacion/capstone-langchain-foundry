"""Output sanitization for untrusted tool results (MCP and database).

MCP servers and user-authored database content are untrusted input that could
contain indirect prompt injection payloads. This module provides a sanitizer
that strips known injection patterns and enforces length limits before tool
output enters the model context.
"""

import json
import re

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
    raw = "\n".join(parts) if parts else str(result)
    return sanitize_tool_output(raw)
