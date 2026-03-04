"""Authentication router: register, login, profile, watchlist, API keys."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select

from quantflow.api.auth.dependencies import CurrentUser, DbDep, require_tier
from quantflow.api.auth.jwt import (
    create_access_token,
    generate_api_key,
    hash_api_key,
    hash_password,
    verify_password,
)
from quantflow.api.auth.schemas import (
    ApiKeyCreate,
    ApiKeyResponse,
    Token,
    UserCreate,
    UserLogin,
    UserResponse,
    WatchlistUpdate,
)
from quantflow.config.constants import TIER_PREMIUM
from quantflow.config.logging import get_logger
from quantflow.db.models import ApiKey, User

logger = get_logger(__name__)

router = APIRouter(tags=["auth"])


# ---------------------------------------------------------------------------
# Registration & login
# ---------------------------------------------------------------------------


@router.post(
    "/register",
    response_model=Token,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user account",
)
async def register(body: UserCreate, db: DbDep) -> Token:
    """Create a new user account.

    Returns a JWT access token on success.
    Raises HTTP 409 if the email is already registered.
    """
    existing = await db.execute(select(User).where(User.email == body.email))
    if existing.scalar_one_or_none() is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists.",
        )

    user = User(
        email=body.email,
        hashed_password=hash_password(body.password),
        subscription_tier="free",
        watchlist=[],
        alert_settings={},
    )
    db.add(user)
    await db.flush()  # get the auto-generated ID

    token, expires_in = create_access_token(user.id, tier=user.subscription_tier)
    logger.info("User registered", user_id=user.id, email=user.email)

    return Token(
        access_token=token,
        expires_in=expires_in,
        user=UserResponse.model_validate(user),
    )


@router.post(
    "/login",
    response_model=Token,
    summary="Login and obtain a JWT token",
)
async def login(body: UserLogin, db: DbDep) -> Token:
    """Authenticate with email and password.

    Returns a JWT access token on success.
    Raises HTTP 401 on invalid credentials.
    """
    result = await db.execute(select(User).where(User.email == body.email))
    user = result.scalar_one_or_none()

    if user is None or not verify_password(body.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated.",
        )

    token, expires_in = create_access_token(user.id, tier=user.subscription_tier)
    logger.info("User logged in", user_id=user.id)

    return Token(
        access_token=token,
        expires_in=expires_in,
        user=UserResponse.model_validate(user),
    )


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------


@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get current user profile",
)
async def get_me(user: CurrentUser) -> UserResponse:
    """Return the profile of the currently authenticated user."""
    return UserResponse.model_validate(user)


@router.put(
    "/me/watchlist",
    response_model=UserResponse,
    summary="Update watchlist",
)
async def update_watchlist(
    body: WatchlistUpdate,
    user: CurrentUser,
    db: DbDep,
) -> UserResponse:
    """Replace the user's watchlist with the provided symbol list.

    Free tier is limited to 5 symbols; premium and institutional are unlimited.
    """
    from quantflow.api.auth.dependencies import check_symbol_limit

    check_symbol_limit(body.symbols, user.subscription_tier)
    user.watchlist = body.symbols
    db.add(user)
    logger.info("Watchlist updated", user_id=user.id, n_symbols=len(body.symbols))
    return UserResponse.model_validate(user)


# ---------------------------------------------------------------------------
# API Keys (premium+)
# ---------------------------------------------------------------------------


@router.post(
    "/api-keys",
    response_model=ApiKeyResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create an API key (Premium+)",
    dependencies=[require_tier(TIER_PREMIUM)],
)
async def create_api_key(
    body: ApiKeyCreate,
    user: CurrentUser,
    db: DbDep,
) -> ApiKeyResponse:
    """Generate a new API key for programmatic access.

    The raw key is returned **only once**; store it securely.
    Subsequent calls return only the key ID and name.
    """
    plain_key, hashed = generate_api_key()
    api_key = ApiKey(
        user_id=user.id,
        key_hash=hashed,
        name=body.name,
        is_active=True,
    )
    db.add(api_key)
    await db.flush()

    logger.info("API key created", user_id=user.id, key_name=body.name)
    resp = ApiKeyResponse.model_validate(api_key)
    resp.key = plain_key  # expose raw key only on creation
    return resp


@router.get(
    "/api-keys",
    response_model=list[ApiKeyResponse],
    summary="List API keys",
    dependencies=[require_tier(TIER_PREMIUM)],
)
async def list_api_keys(user: CurrentUser, db: DbDep) -> list[ApiKeyResponse]:
    """List all active API keys for the authenticated user."""
    result = await db.execute(
        select(ApiKey).where(ApiKey.user_id == user.id, ApiKey.is_active.is_(True))
    )
    keys = result.scalars().all()
    return [ApiKeyResponse.model_validate(k) for k in keys]


@router.delete(
    "/api-keys/{key_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Revoke an API key",
    dependencies=[require_tier(TIER_PREMIUM)],
)
async def revoke_api_key(
    key_id: int,
    user: CurrentUser,
    db: DbDep,
) -> None:
    """Deactivate an API key by ID."""
    result = await db.execute(
        select(ApiKey).where(ApiKey.id == key_id, ApiKey.user_id == user.id)
    )
    api_key = result.scalar_one_or_none()
    if api_key is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="API key not found.")
    api_key.is_active = False
    db.add(api_key)
    logger.info("API key revoked", key_id=key_id, user_id=user.id)
