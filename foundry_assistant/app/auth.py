"""Bearer-token auth: token -> user id (same contract as langchain_assistant)."""

import json
import os

from dotenv import load_dotenv
from fastapi import Header, HTTPException, status

load_dotenv()

_BEARER_CHALLENGE = {"WWW-Authenticate": "Bearer"}


def _load_api_keys() -> dict[str, str]:
    raw = os.environ.get("API_KEYS", "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"API_KEYS is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("API_KEYS must be a JSON object of token -> user_id.")
    return {str(k): str(v) for k, v in parsed.items()}


def _resolve_user_id(token: str) -> str | None:
    return _load_api_keys().get(token)


def get_current_user(authorization: str | None = Header(default=None)) -> str:
    """FastAPI dependency: bearer token -> authenticated user id."""
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header.",
            headers=_BEARER_CHALLENGE,
        )

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header must be 'Bearer <token>'.",
            headers=_BEARER_CHALLENGE,
        )

    user_id = _resolve_user_id(token.strip())
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or unknown access token.",
            headers=_BEARER_CHALLENGE,
        )
    return user_id
