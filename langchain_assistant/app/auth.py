"""Request authentication: bearer token -> authenticated user id.

Phase 6 protects the API so an anonymous caller is rejected and the user
identity is derived from the validated token, NOT from anything the client
(or the model) can put in the request body.

Dev/simulation mode (no Entra app registration required):
  The environment holds a JSON map of opaque bearer tokens to user ids, e.g.
      API_KEYS={"sk-rayan-abc123": "u1", "sk-alice-xyz789": "u2"}
  A request must send `Authorization: Bearer <token>`; we look the token up
  and return the mapped user id. This mirrors the exact shape of real Entra
  JWT auth (Bearer token in, validated identity out) so swapping to Entra
  later only means replacing `_resolve_user_id` with JWT claim validation.
"""

import json
import os

from dotenv import load_dotenv
from fastapi import Header, HTTPException, status

# Load .env so API_KEYS resolves regardless of import order.
load_dotenv()

# WWW-Authenticate header value returned with every 401 (RFC 6750).
_BEARER_CHALLENGE = {"WWW-Authenticate": "Bearer"}


def _load_api_keys() -> dict[str, str]:
    """Parse the API_KEYS JSON map from the environment.

    Returns an empty map if unset/blank. A malformed value is treated as a
    hard configuration error so it fails loudly at request time.
    """
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
    """Map a validated bearer token to a user id, or None if unknown.

    This is the single seam to swap for real Entra JWT validation: validate
    the signature/issuer/audience and return the `sub`/`oid` claim instead.
    """
    return _load_api_keys().get(token)


def get_current_user(authorization: str | None = Header(default=None)) -> str:
    """FastAPI dependency: authenticate the caller and return their user id.

    Raises 401 when the Authorization header is missing, malformed, or the
    token is not recognised. The returned user id is the ONLY source of
    identity downstream; the request body cannot supply or override it.
    """
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
