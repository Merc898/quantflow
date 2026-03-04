"""Admin-only API endpoints.

All routes require ``institutional`` tier.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import select

from quantflow.api.auth.dependencies import DbDep, require_tier
from quantflow.config.constants import TIER_INSTITUTIONAL
from quantflow.db.models import User

router = APIRouter(tags=["admin"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class AdminUserResponse(BaseModel):
    """Minimal user record returned by the admin list endpoint."""

    id: int
    email: str
    subscription_tier: Literal["free", "premium", "institutional"]
    is_active: bool
    is_verified: bool
    created_at: datetime

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/users",
    response_model=list[AdminUserResponse],
    summary="List all users (admin)",
)
async def list_users(
    db: DbDep,
    _admin: User = Depends(require_tier(TIER_INSTITUTIONAL)),
) -> list[AdminUserResponse]:
    """Return every user with their tier and creation timestamp.

    Requires **institutional** tier.
    """
    result = await db.execute(
        select(User).order_by(User.created_at.desc())
    )
    users = result.scalars().all()
    return [AdminUserResponse.model_validate(u) for u in users]
