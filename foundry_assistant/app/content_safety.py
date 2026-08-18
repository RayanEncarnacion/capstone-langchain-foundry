"""Content safety guardrail via Azure AI Content Safety.

Screens text crossing the trust boundary (inbound user messages and outbound
model replies). Blocks turns where any category meets the severity threshold.

Config (environment):
    CONTENT_SAFETY_ENDPOINT  — https://<resource>.cognitiveservices.azure.com/
    CONTENT_SAFETY_KEY       — key auth; omit to use DefaultAzureCredential.

Fails OPEN when unconfigured so the rest of the app is not blocked on
provisioning.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

_SEVERITY_THRESHOLD = 4


@dataclass
class SafetyVerdict:
    """Outcome of a Content Safety screen."""

    allowed: bool
    categories: list[str]
    reason: str = ""


class _Settings:
    def __init__(self) -> None:
        self.endpoint = os.environ.get("CONTENT_SAFETY_ENDPOINT")
        self.key = os.environ.get("CONTENT_SAFETY_KEY")


_settings = _Settings()
_client = None


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
    """Screen text through Azure AI Content Safety.

    Fails open when Content Safety is not configured or the call errors.
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
    except Exception as exc:
        print(f"content_safety: screen failed, allowing by default: {exc}")
        return SafetyVerdict(allowed=True, categories=[], reason=f"error:{exc}")
