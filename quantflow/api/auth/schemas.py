"""Pydantic v2 schemas for authentication and user management."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, EmailStr, Field, field_validator

if TYPE_CHECKING:
    from datetime import datetime

# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------


class UserCreate(BaseModel):
    """User registration request.

    Attributes:
        email: Valid email address (becomes the login identifier).
        password: Plain-text password (min 8 chars); hashed before storage.
    """

    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        """Require at least one digit or special character."""
        has_digit = any(c.isdigit() for c in v)
        has_special = any(c in "!@#$%^&*()_+-=[]{}|;':\",./<>?" for c in v)
        if not (has_digit or has_special):
            raise ValueError("Password must contain at least one digit or special character.")
        return v


class UserLogin(BaseModel):
    """Login request.

    Attributes:
        email: Registered email address.
        password: Plain-text password.
    """

    email: EmailStr
    password: str


class WatchlistUpdate(BaseModel):
    """Watchlist update request.

    Attributes:
        symbols: List of ticker symbols (uppercase, max 50).
    """

    symbols: list[str] = Field(..., max_length=50)

    @field_validator("symbols", mode="before")
    @classmethod
    def uppercase_symbols(cls, v: list[str]) -> list[str]:
        return [s.strip().upper() for s in v]


class ApiKeyCreate(BaseModel):
    """API key creation request.

    Attributes:
        name: Human-readable label for this key.
    """

    name: str = Field(..., min_length=1, max_length=100)


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class UserResponse(BaseModel):
    """User profile response.

    Attributes:
        id: Internal user ID.
        email: Email address.
        subscription_tier: Current subscription level.
        is_verified: Whether email has been verified.
        watchlist: List of tracked symbols.
        created_at: Account creation timestamp (UTC).
    """

    id: int
    email: str
    subscription_tier: Literal["free", "premium", "institutional"]
    is_verified: bool
    watchlist: list[str]
    created_at: datetime

    model_config = {"from_attributes": True}


class Token(BaseModel):
    """JWT access token response.

    Attributes:
        access_token: Signed JWT.
        token_type: Always ``"bearer"``.
        expires_in: Seconds until expiry.
        user: Embedded user profile.
    """

    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse


class TokenPayload(BaseModel):
    """Decoded JWT payload.

    Attributes:
        sub: Subject (user ID as string).
        tier: Subscription tier.
        exp: Expiry timestamp.
    """

    sub: str
    tier: str = "free"
    exp: int


class ApiKeyResponse(BaseModel):
    """API key creation response.

    Attributes:
        id: Key ID.
        name: Key label.
        key: The actual API key (only shown once on creation).
        created_at: Creation timestamp.
    """

    id: int
    name: str
    key: str | None = None  # Only populated on creation
    created_at: datetime

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Error schemas
# ---------------------------------------------------------------------------


class ErrorResponse(BaseModel):
    """Standard error response body.

    Attributes:
        detail: Human-readable error message.
        code: Machine-readable error code.
    """

    detail: str
    code: str = "error"
