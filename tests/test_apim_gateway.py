"""Focused unit tests for APIM model routing and boundary behavior."""

from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from foundry_assistant.app import agent as agent_module
from foundry_assistant.app.gateway import (
    CACHE_ELIGIBILITY_HEADER,
    CACHE_PARTITION_HEADER,
    CORRELATION_HEADER,
    correlation_id,
    model_request_headers,
    rate_limit_retry_after,
)


_APIM_ENV = {
    "FOUNDRY_PROJECT_ENDPOINT": "https://example.services.ai.azure.com/api/projects/p",
    "FOUNDRY_MODEL": "gpt-5-mini",
    "KB_MCP_URL": "https://search.example/knowledgebases/k/mcp",
    "APIM_GATEWAY_REQUIRED": "true",
    "APIM_OPENAI_BASE_URL": "https://gateway.example/foundry/openai/v1",
    "APIM_SUBSCRIPTION_HEADER": "Ocp-Apim-Subscription-Key",
    "APIM_SUBSCRIPTION_KEY": "unit-test-subscription-key",
    "APIM_CACHE_PARTITION_SECRET": "unit-test-cache-secret-with-32-characters",
}


def _settings(monkeypatch, **overrides):
    for name in (
        "APIM_GATEWAY_REQUIRED",
        "APIM_OPENAI_BASE_URL",
        "APIM_SUBSCRIPTION_HEADER",
        "APIM_SUBSCRIPTION_KEY",
        "APIM_CACHE_PARTITION_SECRET",
    ):
        monkeypatch.delenv(name, raising=False)
    values = {**_APIM_ENV, **overrides}
    for name, value in values.items():
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    return agent_module.Settings()


def test_partial_gateway_configuration_fails_without_leaking_key(monkeypatch):
    secret = "do-not-render-this-secret"
    settings = _settings(
        monkeypatch,
        APIM_SUBSCRIPTION_KEY=secret,
        APIM_CACHE_PARTITION_SECRET=None,
    )

    with pytest.raises(RuntimeError) as caught:
        settings.require()

    assert "APIM_CACHE_PARTITION_SECRET" in str(caught.value)
    assert secret not in str(caught.value)


def test_apim_client_uses_gateway_and_configured_auth_header(monkeypatch):
    settings = _settings(monkeypatch)
    settings.require()
    captured = {}

    class FakeOpenAIChatClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import agent_framework.openai

    monkeypatch.setattr(
        agent_framework.openai, "OpenAIChatClient", FakeOpenAIChatClient
    )
    client = agent_module.build_chat_client(settings)

    assert isinstance(client, FakeOpenAIChatClient)
    assert captured["base_url"] == _APIM_ENV["APIM_OPENAI_BASE_URL"]
    assert captured["model"] == "gpt-5-mini"
    assert captured["default_headers"] == {
        "Ocp-Apim-Subscription-Key": _APIM_ENV["APIM_SUBSCRIPTION_KEY"]
    }
    assert captured["api_key"] != _APIM_ENV["APIM_SUBSCRIPTION_KEY"]


def test_cache_partition_is_stable_only_for_opted_in_user_session():
    common = {
        "cache_secret": "s" * 32,
        "user_id": "raw-user-oid",
        "session_id": "raw-session-id",
    }
    first = model_request_headers(
        **common, request_id="request-a", cache_eligible=True
    )
    second = model_request_headers(
        **common, request_id="request-b", cache_eligible=True
    )
    bypass = model_request_headers(
        **common, request_id="request-c", cache_eligible=False
    )
    other_user = model_request_headers(
        **{**common, "user_id": "another-user"},
        request_id="request-d",
        cache_eligible=True,
    )

    assert first[CACHE_PARTITION_HEADER] == second[CACHE_PARTITION_HEADER]
    assert bypass[CACHE_PARTITION_HEADER] != first[CACHE_PARTITION_HEADER]
    assert other_user[CACHE_PARTITION_HEADER] != first[CACHE_PARTITION_HEADER]
    assert first[CACHE_ELIGIBILITY_HEADER] == "true"
    assert bypass[CACHE_ELIGIBILITY_HEADER] == "false"
    assert "raw-user-oid" not in first[CACHE_PARTITION_HEADER]
    assert "raw-session-id" not in first[CACHE_PARTITION_HEADER]


def test_correlation_rejects_free_form_or_header_injection():
    assert correlation_id("safe-id_123") == "safe-id_123"
    generated = correlation_id("person@example.com\r\nX-Evil: yes")
    assert generated != "person@example.com\r\nX-Evil: yes"
    assert len(generated) == 32


def test_fastapi_response_propagates_safe_correlation_id():
    from foundry_assistant.app.api import app

    client = TestClient(app)
    response = client.get("/health", headers={CORRELATION_HEADER: "trace-test-123"})

    assert response.status_code == 200
    assert response.headers[CORRELATION_HEADER] == "trace-test-123"


def test_nested_429_preserves_only_safe_retry_after():
    response = SimpleNamespace(status_code=429, headers={"retry-after-ms": "1501"})
    inner = SimpleRateLimitError(response)
    outer = RuntimeError("framework wrapper", inner)

    assert rate_limit_retry_after(outer) == (True, "2")


def test_fastapi_boundary_maps_429_and_does_not_leak_other_errors():
    from foundry_assistant.app.api import _raise_model_error

    limited = SimpleRateLimitError(
        SimpleNamespace(status_code=429, headers={"retry-after": "7"})
    )
    with pytest.raises(HTTPException) as caught:
        _raise_model_error(limited, "corr-1", "Agent call")
    assert caught.value.status_code == 429
    assert caught.value.headers == {"Retry-After": "7"}

    secret = "subscription-secret-that-must-not-leak"
    with pytest.raises(HTTPException) as caught:
        _raise_model_error(RuntimeError(secret), "corr-2", "Agent call")
    assert caught.value.status_code == 502
    assert secret not in caught.value.detail
    assert "corr-2" in caught.value.detail


def test_unexpected_value_error_is_sanitized_at_chat_boundary(monkeypatch):
    """Only explicit AgentSessionError instances may be reflected as 404s."""

    import asyncio

    from foundry_assistant.app import api

    secret = "sdk-value-error-secret"

    async def failing_run(*args, **kwargs):
        raise ValueError(secret)

    monkeypatch.setattr(api, "run_agent", failing_run)
    monkeypatch.setattr(api, "_guard_input", lambda _message: None)
    request = SimpleNamespace(
        state=SimpleNamespace(correlation_id="corr-value-error")
    )

    with pytest.raises(HTTPException) as caught:
        asyncio.run(
            api.chat(
                request=request,
                payload=api.ChatRequest(message="hello"),
                user_id="test-user",
            )
        )
    assert caught.value.status_code == 502
    assert secret not in caught.value.detail


class SimpleRateLimitError(Exception):
    def __init__(self, response):
        super().__init__("rate limited")
        self.response = response
        self.status_code = response.status_code
