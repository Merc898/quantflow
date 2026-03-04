"""Stripe subscription management endpoints.

Handles:
- Checkout session creation (upgrade to Premium / Institutional).
- Stripe webhook processing (subscription lifecycle events).
- Subscription status queries.
- Subscription cancellation.
"""

from __future__ import annotations

import hmac
import hashlib
from typing import Any, Literal

from fastapi import APIRouter, Header, HTTPException, Request, status
from pydantic import BaseModel
from sqlalchemy import select

from quantflow.api.auth.dependencies import CurrentUser, DbDep
from quantflow.config.logging import get_logger
from quantflow.config.settings import settings
from quantflow.db.models import User

logger = get_logger(__name__)

router = APIRouter(tags=["subscription"])

try:
    import stripe as _stripe

    _STRIPE_AVAILABLE = True
except ImportError:
    _STRIPE_AVAILABLE = False

# Stripe price IDs (set via environment in production)
_PRICE_IDS: dict[str, str] = {
    "premium": "price_premium_placeholder",
    "institutional": "price_institutional_placeholder",
}


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class CheckoutRequest(BaseModel):
    """Create a Stripe checkout session.

    Attributes:
        tier: Target subscription tier.
        success_url: Redirect URL after successful payment.
        cancel_url: Redirect URL if user cancels checkout.
    """

    tier: Literal["premium", "institutional"]
    success_url: str = "http://localhost:3000/dashboard?upgraded=true"
    cancel_url: str = "http://localhost:3000/pricing"


class CheckoutResponse(BaseModel):
    """Checkout session response.

    Attributes:
        checkout_url: URL to redirect the user to for payment.
        session_id: Stripe session ID.
    """

    checkout_url: str
    session_id: str


class SubscriptionStatus(BaseModel):
    """Current subscription status.

    Attributes:
        tier: Current tier.
        stripe_customer_id: Stripe customer ID if set.
        stripe_subscription_id: Active subscription ID if set.
        is_active: Whether the subscription is currently active.
    """

    tier: str
    stripe_customer_id: str | None
    stripe_subscription_id: str | None
    is_active: bool


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/status",
    response_model=SubscriptionStatus,
    summary="Get current subscription status",
)
async def subscription_status(user: CurrentUser) -> SubscriptionStatus:
    """Return the current user's subscription tier and Stripe IDs."""
    return SubscriptionStatus(
        tier=user.subscription_tier,
        stripe_customer_id=user.stripe_customer_id,
        stripe_subscription_id=user.stripe_subscription_id,
        is_active=user.is_active,
    )


@router.post(
    "/create-checkout-session",
    response_model=CheckoutResponse,
    summary="Create a Stripe checkout session for subscription upgrade",
)
async def create_checkout(
    body: CheckoutRequest,
    user: CurrentUser,
) -> CheckoutResponse:
    """Create a Stripe checkout session for subscription upgrade.

    Returns a checkout URL to redirect the user to.
    Requires Stripe to be configured (``STRIPE_SECRET_KEY``).
    """
    if not _STRIPE_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Stripe integration is not available.",
        )

    stripe_key = settings.stripe_secret_key
    if stripe_key is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Stripe is not configured. Set STRIPE_SECRET_KEY.",
        )

    _stripe.api_key = stripe_key.get_secret_value()

    price_id = _PRICE_IDS.get(body.tier)
    if not price_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown tier: {body.tier}",
        )

    try:
        session = _stripe.checkout.Session.create(
            mode="subscription",
            payment_method_types=["card"],
            line_items=[{"price": price_id, "quantity": 1}],
            success_url=body.success_url,
            cancel_url=body.cancel_url,
            customer_email=user.email,
            metadata={"user_id": str(user.id), "tier": body.tier},
        )
        return CheckoutResponse(checkout_url=session.url, session_id=session.id)
    except Exception as exc:
        logger.error("Stripe checkout creation failed", error=str(exc), user_id=user.id)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Stripe error: {exc}",
        )


@router.delete(
    "/cancel",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Cancel the current subscription",
)
async def cancel_subscription(
    user: CurrentUser,
    db: DbDep,
) -> None:
    """Cancel the active Stripe subscription at period end.

    Downgrades the user to the free tier immediately in the database.
    """
    if user.stripe_subscription_id and _STRIPE_AVAILABLE:
        stripe_key = settings.stripe_secret_key
        if stripe_key is not None:
            try:
                _stripe.api_key = stripe_key.get_secret_value()
                _stripe.Subscription.modify(
                    user.stripe_subscription_id,
                    cancel_at_period_end=True,
                )
                logger.info("Subscription cancellation scheduled", user_id=user.id)
            except Exception as exc:
                logger.warning("Stripe cancellation failed", error=str(exc))

    user.subscription_tier = "free"
    user.stripe_subscription_id = None
    db.add(user)
    logger.info("User downgraded to free tier", user_id=user.id)


# ---------------------------------------------------------------------------
# Stripe webhook
# ---------------------------------------------------------------------------


@router.post(
    "/stripe/webhook",
    status_code=status.HTTP_200_OK,
    summary="Stripe webhook handler",
    include_in_schema=False,  # Not shown in public docs
)
async def stripe_webhook(
    request: Request,
    db: DbDep,
    stripe_signature: str | None = Header(default=None, alias="stripe-signature"),
) -> dict[str, str]:
    """Process Stripe lifecycle events.

    Verifies webhook signature and handles:
    - ``customer.subscription.created`` → activate Premium/Institutional.
    - ``customer.subscription.deleted`` → downgrade to free.
    - ``invoice.payment_failed`` → log warning (no immediate downgrade).
    """
    payload = await request.body()

    webhook_secret = settings.stripe_webhook_secret
    if webhook_secret and stripe_signature and _STRIPE_AVAILABLE:
        try:
            _stripe.api_key = (settings.stripe_secret_key or "").get_secret_value() if settings.stripe_secret_key else ""
            event = _stripe.Webhook.construct_event(
                payload,
                stripe_signature,
                webhook_secret.get_secret_value(),
            )
        except Exception as exc:
            logger.warning("Invalid Stripe webhook signature", error=str(exc))
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid signature")
    else:
        # In development: parse JSON without verification
        import json
        try:
            event = json.loads(payload)
        except Exception:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid JSON")

    event_type = event.get("type", "")
    data = event.get("data", {}).get("object", {})

    match event_type:
        case "customer.subscription.created" | "customer.subscription.updated":
            await _handle_subscription_active(data, db)
        case "customer.subscription.deleted":
            await _handle_subscription_cancelled(data, db)
        case "invoice.payment_failed":
            logger.warning(
                "Stripe payment failed",
                customer=data.get("customer"),
                amount=data.get("amount_due"),
            )
        case _:
            logger.info("Unhandled Stripe event", event_type=event_type)

    return {"status": "ok"}


async def _handle_subscription_active(data: dict[str, Any], db: DbDep) -> None:
    """Activate or upgrade a subscription.

    Args:
        data: Stripe subscription object.
        db: Database session.
    """
    customer_id = data.get("customer")
    metadata = data.get("metadata", {})
    user_id = metadata.get("user_id")
    tier = metadata.get("tier", "premium")

    if not user_id:
        logger.warning("Stripe webhook: missing user_id in metadata")
        return

    result = await db.execute(select(User).where(User.id == int(user_id)))
    user = result.scalar_one_or_none()
    if user is None:
        logger.warning("Stripe webhook: user not found", user_id=user_id)
        return

    user.subscription_tier = tier
    user.stripe_customer_id = customer_id
    user.stripe_subscription_id = data.get("id")
    db.add(user)
    logger.info("Subscription activated", user_id=user_id, tier=tier)


async def _handle_subscription_cancelled(data: dict[str, Any], db: DbDep) -> None:
    """Downgrade a user to free tier on subscription cancellation.

    Args:
        data: Stripe subscription object.
        db: Database session.
    """
    subscription_id = data.get("id")
    result = await db.execute(
        select(User).where(User.stripe_subscription_id == subscription_id)
    )
    user = result.scalar_one_or_none()
    if user is None:
        logger.warning("Stripe webhook: no user for subscription", sub_id=subscription_id)
        return

    user.subscription_tier = "free"
    user.stripe_subscription_id = None
    db.add(user)
    logger.info("Subscription cancelled → free tier", user_id=user.id)
