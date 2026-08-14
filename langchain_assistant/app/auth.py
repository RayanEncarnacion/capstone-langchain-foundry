"""Lightweight API-key auth: maps bearer tokens to user IDs.

Keys live in the API_KEYS env var as a JSON object:
  API_KEYS={"sk-rayan-abc123": "u1", "sk-alice-xyz789": "u2"}

The dependency ``get_current_user`` extracts the bearer token from the
Authorization header, looks it up, and returns the matching user_id.
Swap this module for JWT / Azure AD later without touching the rest of
the app — the contract is just ``Depends(get_current_user) -> str``.
"""

import json
import os

from fastapi import Header, HTTPException


def _load_api_keys() -> dict[str, str]:
    raw = os.environ.get("API_KEYS", "{}")
    try:
        keys = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"API_KEYS env var is not valid JSON: {exc}") from exc
    if not isinstance(keys, dict):
        raise RuntimeError("API_KEYS must be a JSON object mapping tokens to user IDs")
    return keys


API_KEYS: dict[str, str] = {}


def init_api_keys() -> None:
    """Load keys from the environment. Call once at startup."""
    global API_KEYS  # noqa: PLW0603
    API_KEYS = _load_api_keys()
    if not API_KEYS:
        raise RuntimeError(
            "API_KEYS is empty — set it to a JSON object like "
            '{\"sk-mykey\": \"user-id\"}'
        )


def get_current_user(authorization: str = Header(...)) -> str:
    """FastAPI dependency: validate bearer token, return user_id."""
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(status_code=401, detail="Expected 'Authorization: Bearer <key>'")
    user_id = API_KEYS.get(token)
    if user_id is None:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return user_id
