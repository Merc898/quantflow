"""FastAPI dependency injection for authentication and authorisation.

Provides:
- ``get_db`` — async SQLAlchemy session.
- ``get_current_user`` — extract and verify JWT → User ORM object.
- ``get_optional_user`` — same but returns ``None`` if unauthenticated.
- ``require_tier(min_tier)`` — dependency factory for tier enforcement.
- ``check_symbol_limit`` — enforce per-tier symbol quotas.
"""

from __future__ import annotations

from typing import Annotated, AsyncGenerator, Callable

from fastapi import Depends, HTTPException, Request, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from quantflow.api.auth.jwt import decode_token, hash_api_key
from quantflow.config.constants import (
    TIER_FREE,
    TIER_INSTITUTIONAL,
    TIER_PREMIUM,
    TIER_SYMBOL_LIMITS,
)
from quantflow.config.logging import get_logger
from quantflow.db.engine import AsyncSessionLocal
from quantflow.db.models import ApiKey, User

logger = get_logger(__name__)

_bearer = HTTPBearer(auto_error=False)

# Tier ordering for comparison
_TIER_ORDER = {TIER_FREE: 0, TIER_PREMIUM: 1, TIER_INSTITUTIONAL: 2}


# ---------------------------------------------------------------------------
# Database session
# ---------------------------------------------------------------------------


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Provide an async SQLAlchemy session per request.

    Yields:
        :class:`AsyncSession` that is committed on success and rolled back
        on exception.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


DbDep = Annotated[AsyncSession, Depends(get_db)]


# ---------------------------------------------------------------------------
# Current user extraction
# ---------------------------------------------------------------------------


async def _user_from_jwt(token: str, db: AsyncSession) -> User:
    """Verify JWT and load user from database.

    Args:
        token: Raw JWT string.
        db: Active async database session.

    Returns:
        :class:`User` ORM object.

    Raises:
        HTTPException 401: Token invalid, expired, or user not found.
    """
    credentials_error = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = decode_token(token)
    except JWTError:
        raise credentials_error

    result = await db.execute(select(User).where(User.id == int(payload.sub)))
    user = result.scalar_one_or_none()
    if user is None or not user.is_active:
        raise credentials_error
    return user


async def _user_from_api_key(api_key: str, db: AsyncSession) -> User:
    """Verify API key and load user.

    Args:
        api_key: Raw API key string.
        db: Active async database session.

    Returns:
        :class:`User` ORM object.

    Raises:
        HTTPException 401: Key not found or inactive.
    """
    key_hash = hash_api_key(api_key)
    result = await db.execute(
        select(ApiKey, User)
        .join(User, User.id == ApiKey.user_id)
        .where(ApiKey.key_hash == key_hash, ApiKey.is_active.is_(True))
    )
    row = result.first()
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
        )
    _, user = row
    return user


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Security(_bearer)],
    request: Request,
    db: DbDep,
) -> User:
    """Extract the authenticated user from Bearer token or X-API-Key header.

    Priority:
    1. ``Authorization: Bearer <jwt>``
    2. ``X-API-Key: <key>``

    Args:
        credentials: Parsed Bearer credentials (may be None).
        request: FastAPI request (for X-API-Key header fallback).
        db: Database session.

    Returns:
        Authenticated :class:`User`.

    Raises:
        HTTPException 401: No valid credentials provided.
    """
    # 1. JWT Bearer token
    if credentials and credentials.credentials:
        return await _user_from_jwt(credentials.credentials, db)

    # 2. API Key header
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return await _user_from_api_key(api_key, db)

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required.",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def get_optional_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Security(_bearer)],
    request: Request,
    db: DbDep,
) -> User | None:
    """Like :func:`get_current_user` but returns ``None`` if unauthenticated.

    Args:
        credentials: Optional bearer credentials.
        request: FastAPI request.
        db: Database session.

    Returns:
        Authenticated :class:`User` or ``None``.
    """
    try:
        return await get_current_user(credentials, request, db)
    except HTTPException:
        return None


CurrentUser = Annotated[User, Depends(get_current_user)]
OptionalUser = Annotated[User | None, Depends(get_optional_user)]


# ---------------------------------------------------------------------------
# Tier enforcement
# ---------------------------------------------------------------------------


def require_tier(min_tier: str) -> Callable:
    """Dependency factory: enforce a minimum subscription tier.

    Args:
        min_tier: Minimum required tier (``"free"``, ``"premium"``, or
            ``"institutional"``).

    Returns:
        FastAPI dependency function that raises HTTP 403 if the user's tier
        is below ``min_tier``.

    Example::

        @router.get("/premium-only")
        async def endpoint(user: CurrentUser = Depends(require_tier("premium"))):
            ...
    """

    async def _check(user: CurrentUser) -> User:
        user_level = _TIER_ORDER.get(user.subscription_tier, 0)
        required_level = _TIER_ORDER.get(min_tier, 0)
        if user_level < required_level:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    f"This feature requires a {min_tier} subscription. "
                    f"Your current plan: {user.subscription_tier}."
                ),
            )
        return user

    return Depends(_check)


# ---------------------------------------------------------------------------
# Symbol limit enforcement
# ---------------------------------------------------------------------------


def check_symbol_limit(symbols: list[str], tier: str) -> None:
    """Raise HTTP 403 if the symbol count exceeds the tier limit.

    Args:
        symbols: List of requested symbols.
        tier: User's subscription tier.

    Raises:
        HTTPException 403: Symbol count exceeds the tier quota.
    """
    limit = TIER_SYMBOL_LIMITS.get(tier, 5)
    if limit == 0:
        return  # 0 = unlimited
    if len(symbols) > limit:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                f"Your {tier} plan allows up to {limit} symbols. "
                f"Requested {len(symbols)}. Please upgrade."
            ),
        )
