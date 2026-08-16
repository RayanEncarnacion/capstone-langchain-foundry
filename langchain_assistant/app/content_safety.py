"""PII / harmful-content guardrail via Azure AI Content Safety.

We screen text that crosses the trust boundary: the user's inbound message
and the agent's outbound reply. If Content Safety flags a category at or
above the configured severity, we block the turn instead of returning the
text.

Config (from the environment):
  CONTENT_SAFETY_ENDPOINT  - https://<resource>.cognitiveservices.azure.com/
  CONTENT_SAFETY_KEY       - key auth; omit to use DefaultAzureCredential.

If no endpoint is configured we fail OPEN (allow) but log a warning, so the
rest of Phase 6 is not blocked on provisioning. Everything is read from the
environment; nothing is hard-coded.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Load .env so settings resolve regardless of import order.
load_dotenv()

# Severity threshold (Azure returns 0..7 in steps of 2). >=4 = medium+.
_SEVERITY_THRESHOLD = 4


@dataclass
class SafetyVerdict:
    """Outcome of a Content Safety screen."""

    allowed: bool
    categories: list[str]  # categories that breached the threshold.
    reason: str = ""


class _Settings:
    """Content Safety connection settings, read from the environment."""

    def __init__(self) -> None:
        self.endpoint = os.environ.get("CONTENT_SAFETY_ENDPOINT")
        self.key = os.environ.get("CONTENT_SAFETY_KEY")


_settings = _Settings()
_client = None  # built lazily and cached.


def _get_client():
    """Return a cached ContentSafetyClient, or None if unconfigured."""
    global _client
    if _client is not None:
        return _client
    if not _settings.endpoint:
        return None

    from azure.ai.contentsafety import ContentSafetyClient
    from azure.core.credentials import AzureKeyCredential

    if _settings.key:
        credential = AzureKeyCredential(_settings.key)
    else:
        from azure.identity import DefaultAzureCredential

        credential = DefaultAzureCredential()

    _client = ContentSafetyClient(endpoint=_settings.endpoint, credential=credential)
    return _client


def screen_text(text: str) -> SafetyVerdict:
    """Screen `text` through Azure AI Content Safety.

    Returns a SafetyVerdict. Fails open (allowed=True) when Content Safety is
    not configured or the call errors, so a screening outage does not take the
    whole API down. Blocking decisions are only made on a real positive result.
    """
    if not text or not text.strip():
        return SafetyVerdict(allowed=True, categories=[])

    client = _get_client()
    if client is None:
        print("content_safety: CONTENT_SAFETY_ENDPOINT unset; skipping screen.")
        return SafetyVerdict(allowed=True, categories=[], reason="not_configured")

    try:
        from azure.ai.contentsafety.models import AnalyzeTextOptions

        response = client.analyze_text(AnalyzeTextOptions(text=text))
        flagged = [
            item.category
            for item in (response.categories_analysis or [])
            if (item.severity or 0) >= _SEVERITY_THRESHOLD
        ]
        if flagged:
            return SafetyVerdict(
                allowed=False,
                categories=[str(c) for c in flagged],
                reason="content_safety_flagged",
            )
        return SafetyVerdict(allowed=True, categories=[])
    except Exception as exc:  # never let a screening error crash the request.
        print(f"content_safety: screen failed, allowing by default: {exc}")
        return SafetyVerdict(allowed=True, categories=[], reason=f"error:{exc}")
