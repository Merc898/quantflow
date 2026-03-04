"""Authentication: JWT, user management, subscription tiers."""

from quantflow.api.auth.dependencies import (
    CurrentUser,
    DbDep,
    OptionalUser,
    check_symbol_limit,
    get_current_user,
    get_db,
    get_optional_user,
    require_tier,
)
from quantflow.api.auth.jwt import (
    create_access_token,
    decode_token,
    generate_api_key,
    hash_password,
    verify_password,
)
from quantflow.api.auth.schemas import (
    ApiKeyCreate,
    ApiKeyResponse,
    ErrorResponse,
    Token,
    TokenPayload,
    UserCreate,
    UserLogin,
    UserResponse,
    WatchlistUpdate,
)

__all__ = [
    # Schemas
    "ApiKeyCreate",
    "ApiKeyResponse",
    # Dependencies
    "CurrentUser",
    "DbDep",
    "ErrorResponse",
    "OptionalUser",
    "Token",
    "TokenPayload",
    "UserCreate",
    "UserLogin",
    "UserResponse",
    "WatchlistUpdate",
    "check_symbol_limit",
    # JWT
    "create_access_token",
    "decode_token",
    "generate_api_key",
    "get_current_user",
    "get_db",
    "get_optional_user",
    "hash_password",
    "require_tier",
    "verify_password",
]
