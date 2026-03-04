"""JWT token creation, verification, and password hashing.

Uses PyJWT for JWT operations and passlib[bcrypt] for password hashing.
All secrets are read from application settings.
"""

from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any

import bcrypt as _bcrypt_lib
from jose import JWTError, jwt

from quantflow.api.auth.schemas import TokenPayload
from quantflow.config.logging import get_logger
from quantflow.config.settings import settings

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Password hashing (using bcrypt directly — avoids passlib/bcrypt≥4 conflict)
# ---------------------------------------------------------------------------


def hash_password(plain_password: str) -> str:
    """Hash a plain-text password with bcrypt.

    Args:
        plain_password: Raw password from user.

    Returns:
        bcrypt hash string (safe to store in DB).
    """
    return _bcrypt_lib.hashpw(plain_password.encode(), _bcrypt_lib.gensalt()).decode()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain-text password against a stored hash.

    Args:
        plain_password: Raw password from login attempt.
        hashed_password: Stored bcrypt hash.

    Returns:
        True if password matches.
    """
    return _bcrypt_lib.checkpw(plain_password.encode(), hashed_password.encode())


# ---------------------------------------------------------------------------
# JWT operations
# ---------------------------------------------------------------------------


def create_access_token(
    subject: int | str,
    tier: str = "free",
    expires_delta: timedelta | None = None,
) -> tuple[str, int]:
    """Create a signed JWT access token.

    Args:
        subject: User ID (used as JWT ``sub`` claim).
        tier: Subscription tier embedded in the token.
        expires_delta: Custom expiry; defaults to ``settings.jwt_expire_minutes``.

    Returns:
        Tuple of (token_string, expires_in_seconds).
    """
    delta = expires_delta or timedelta(minutes=settings.jwt_expire_minutes)
    expire = datetime.now(tz=timezone.utc) + delta
    payload: dict[str, Any] = {
        "sub": str(subject),
        "tier": tier,
        "exp": int(expire.timestamp()),
        "iat": int(datetime.now(tz=timezone.utc).timestamp()),
    }
    token = jwt.encode(
        payload,
        settings.jwt_secret_key.get_secret_value(),
        algorithm=settings.jwt_algorithm,
    )
    return token, int(delta.total_seconds())


def decode_token(token: str) -> TokenPayload:
    """Decode and validate a JWT access token.

    Args:
        token: Raw Bearer token string.

    Returns:
        :class:`TokenPayload` with ``sub``, ``tier``, ``exp``.

    Raises:
        JWTError: If the token is invalid or expired.
    """
    payload = jwt.decode(
        token,
        settings.jwt_secret_key.get_secret_value(),
        algorithms=[settings.jwt_algorithm],
    )
    return TokenPayload(
        sub=str(payload["sub"]),
        tier=payload.get("tier", "free"),
        exp=int(payload["exp"]),
    )


# ---------------------------------------------------------------------------
# API key utilities
# ---------------------------------------------------------------------------


def generate_api_key() -> tuple[str, str]:
    """Generate a new API key pair.

    Returns:
        Tuple of (plain_key, hashed_key).
        Store the hash in the DB; return the plain key to the user once.
    """
    plain = "qf_" + secrets.token_urlsafe(32)
    hashed = hashlib.sha256(plain.encode()).hexdigest()
    return plain, hashed


def hash_api_key(plain_key: str) -> str:
    """Hash an API key for storage/lookup.

    Args:
        plain_key: Raw API key string.

    Returns:
        SHA-256 hex digest.
    """
    return hashlib.sha256(plain_key.encode()).hexdigest()
