"""Request authentication: real Microsoft Entra bearer tokens -> user id.

Phase 5 protects the API with genuine Entra ID auth instead of a hardcoded
token map. The caller obtains a real Microsoft-issued access token (e.g. from
the Azure CLI) and sends it as `Authorization: Bearer <token>`; this module
validates the token's signature against Entra's published JWKS, checks the
issuer/expiry (and optionally the audience), and derives the user identity
from the token's own claims (`oid`).

Because the identity comes from a cryptographically validated token, neither
the client body nor the model can spoof or override which user — and therefore
which Cosmos partition — a request operates on.

How to get a token (Azure CLI, MFA-backed):
    az account get-access-token \
        --resource https://<project-name>.services.ai.azure.com \
        --query accessToken -o tsv

Environment variables (all optional):
    ENTRA_TENANT_ID     — restrict accepted tokens to this tenant (else the
                          token's own `tid` claim is trusted for key lookup).
    ENTRA_API_AUDIENCE  — if set, the token `aud` must match it exactly.
"""

import os
import time

import httpx
from dotenv import load_dotenv
from fastapi import Header, HTTPException, status
from jose import jwt
from jose.exceptions import JWTError

load_dotenv()

# WWW-Authenticate header value returned with every 401 (RFC 6750).
_BEARER_CHALLENGE = {"WWW-Authenticate": "Bearer"}

# Optional hardening knobs (see module docstring).
_EXPECTED_TENANT = os.environ.get("ENTRA_TENANT_ID")
_EXPECTED_AUDIENCE = os.environ.get("ENTRA_API_AUDIENCE")

# Entra signing keys change rarely; cache per tenant to avoid a network hop on
# every request. (fetched_at, keys).
_JWKS_TTL_SECONDS = 3600
_JWKS_CACHE: dict[str, tuple[float, list[dict]]] = {}


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers=_BEARER_CHALLENGE,
    )


def _fetch_jwks(tenant_id: str) -> list[dict]:
    """Return the tenant's JSON Web Key Set, cached for a short TTL."""
    cached = _JWKS_CACHE.get(tenant_id)
    now = time.time()
    if cached and now - cached[0] < _JWKS_TTL_SECONDS:
        return cached[1]

    url = f"https://login.microsoftonline.com/{tenant_id}/discovery/v2.0/keys"
    resp = httpx.get(url, timeout=10)
    resp.raise_for_status()
    keys = resp.json().get("keys", [])
    _JWKS_CACHE[tenant_id] = (now, keys)
    return keys


def _valid_issuers(tenant_id: str) -> set[str]:
    """Accept both v1 (sts.windows.net) and v2 issuer formats for a tenant."""
    return {
        f"https://sts.windows.net/{tenant_id}/",
        f"https://login.microsoftonline.com/{tenant_id}/v2.0",
    }


def _validate_token(token: str) -> dict:
    """Verify signature/issuer/expiry (and audience) and return the claims."""
    try:
        header = jwt.get_unverified_header(token)
        unverified = jwt.get_unverified_claims(token)
    except JWTError as exc:
        raise _unauthorized(f"Malformed bearer token: {exc}") from exc

    tenant_id = _EXPECTED_TENANT or unverified.get("tid")
    if not tenant_id:
        raise _unauthorized("Token is missing a tenant id (tid) claim.")

    kid = header.get("kid")
    keys = _fetch_jwks(tenant_id)
    signing_key = next((k for k in keys if k.get("kid") == kid), None)
    if signing_key is None:
        raise _unauthorized("Token signing key not found in tenant JWKS.")

    # Verify audience only when configured; az CLI tokens are minted for a
    # resource the caller chose, so we don't hardcode one.
    options = {"verify_aud": bool(_EXPECTED_AUDIENCE)}
    try:
        claims = jwt.decode(
            token,
            signing_key,
            algorithms=[header.get("alg", "RS256")],
            audience=_EXPECTED_AUDIENCE,
            options=options,
        )
    except JWTError as exc:
        raise _unauthorized(f"Token validation failed: {exc}") from exc

    if claims.get("iss") not in _valid_issuers(tenant_id):
        raise _unauthorized("Token issuer is not a trusted Entra endpoint.")

    return claims


def _user_id_from_claims(claims: dict) -> str:
    """Derive a stable per-user id from validated Entra claims.

    Prefers the immutable object id (`oid`); falls back to `sub`. This value
    becomes the Cosmos partition key, so it must be stable per user.
    """
    user_id = claims.get("oid") or claims.get("sub")
    if not user_id:
        raise _unauthorized("Token has no usable identity claim (oid/sub).")
    return str(user_id)


def get_current_user(authorization: str | None = Header(default=None)) -> str:
    """FastAPI dependency: validate the Entra token and return the user id.

    Raises 401 when the Authorization header is missing, malformed, or the
    token fails validation. The returned user id is the ONLY source of
    identity downstream; the request body cannot supply or override it.
    """
    if not authorization:
        raise _unauthorized("Missing Authorization header.")

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise _unauthorized("Authorization header must be 'Bearer <token>'.")

    claims = _validate_token(token.strip())
    return _user_id_from_claims(claims)
