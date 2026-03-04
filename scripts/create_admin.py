"""Create (or fetch) the admin user and print a ready-to-use JWT.

Usage::

    python -m scripts.create_admin          # from project root
    python scripts/create_admin.py          # also works

Idempotent: if admin@quantflow.dev already exists the script just
generates a fresh token for the existing account.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Ensure project root is on sys.path so `quantflow` is importable when
# the script is executed directly (python scripts/create_admin.py).
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from sqlalchemy import select

from quantflow.api.auth.jwt import create_access_token, hash_password
from quantflow.config.constants import TIER_INSTITUTIONAL
from quantflow.db.engine import AsyncSessionLocal, engine
from quantflow.db.models import Base, User

ADMIN_EMAIL = "admin@quantflow.dev"
ADMIN_PASSWORD = "Admin123!"


async def main() -> None:
    """Create the admin user if needed and print a JWT token."""

    # Ensure tables exist (safe to call repeatedly).
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with AsyncSessionLocal() as session:
        result = await session.execute(select(User).where(User.email == ADMIN_EMAIL))
        user = result.scalar_one_or_none()

        if user is None:
            user = User(
                email=ADMIN_EMAIL,
                hashed_password=hash_password(ADMIN_PASSWORD),
                subscription_tier=TIER_INSTITUTIONAL,
                is_verified=True,
                is_active=True,
                watchlist=[],
                alert_settings={},
            )
            session.add(user)
            await session.flush()  # populate user.id
            await session.commit()
            await session.refresh(user)
            print(f"Created admin user  id={user.id}  email={ADMIN_EMAIL}")
        else:
            print(f"Admin user already exists  id={user.id}  email={ADMIN_EMAIL}")

        token, expires_in = create_access_token(
            subject=user.id,
            tier=user.subscription_tier,
        )

        print(f"Tier: {user.subscription_tier}")
        print(f"Token expires in: {expires_in // 3600}h")
        print()
        print(token)

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
