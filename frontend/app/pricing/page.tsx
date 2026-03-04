"use client";

import { useState } from "react";
import Link from "next/link";
import { Check, Zap, Star, Building2, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
  CardFooter,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { subscription as subscriptionApi, ApiError } from "@/lib/api";
import { useIsAuthenticated, useUser } from "@/lib/store";
import { cn } from "@/lib/utils";

// ---------------------------------------------------------------------------
// Tier data
// ---------------------------------------------------------------------------

const tiers = [
  {
    id: "free" as const,
    name: "Free",
    price: "€0",
    period: "forever",
    icon: Zap,
    description: "Get started with core quantitative signals.",
    cta: "Get started",
    ctaVariant: "outline" as const,
    featured: false,
    features: [
      "5 symbols in watchlist",
      "Daily signal refresh",
      "10 basic models",
      "30-day signal history",
      "Headlines-only intelligence",
      "Community support",
    ],
    missing: [
      "All 50+ models",
      "SHAP explainability",
      "Portfolio optimizer",
      "AI intelligence reports",
      "API access",
      "Real-time signals",
    ],
  },
  {
    id: "premium" as const,
    name: "Premium",
    price: "€49",
    period: "per month",
    icon: Star,
    description: "Full model suite with AI intelligence and portfolio tools.",
    cta: "Upgrade to Premium",
    ctaVariant: "default" as const,
    featured: true,
    features: [
      "Unlimited symbols",
      "4-hour signal refresh",
      "All 50+ quantitative models",
      "SHAP feature explainability",
      "Full AI intelligence reports",
      "Portfolio optimizer (MVO/HRP/BL)",
      "Signal screener",
      "3-year signal history",
      "API access (1,000 req/day)",
      "CSV/PDF export",
      "Email alerts",
      "Email support",
    ],
    missing: [],
  },
  {
    id: "institutional" as const,
    name: "Institutional",
    price: "€299",
    period: "per month",
    icon: Building2,
    description: "Custom models, real-time data, and dedicated support.",
    cta: "Contact Sales",
    ctaVariant: "outline" as const,
    featured: false,
    features: [
      "Everything in Premium",
      "Real-time signals",
      "Custom model integration",
      "Full signal history",
      "Unlimited API access",
      "Options & derivatives models",
      "Email + Webhook + SMS alerts",
      "Dedicated account manager",
      "SLA: 99.9% uptime",
    ],
    missing: [],
  },
];

// ---------------------------------------------------------------------------
// FAQ
// ---------------------------------------------------------------------------

const faqs = [
  {
    q: "Can I cancel at any time?",
    a: "Yes. You can cancel your subscription at any time from the Settings page. You'll retain access until the end of your billing period.",
  },
  {
    q: "How are signals generated?",
    a: "QuantFlow runs an ensemble of 50+ statistical, ML, and AI models including GARCH, Heston, LSTM, PatchTST, Hawkes processes, and agentic LLM intelligence. Signals are aggregated with IC-weighted dynamic calibration.",
  },
  {
    q: "Is this financial advice?",
    a: "No. QuantFlow provides quantitative research tools and signals for informational purposes only. Always conduct your own due diligence before making investment decisions.",
  },
  {
    q: "What data sources are used?",
    a: "Price data from Yahoo Finance and Polygon.io, fundamentals from Alpha Vantage, macro data from FRED, and alternative data via real-time web intelligence.",
  },
];

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export default function PricingPage() {
  const isAuthenticated = useIsAuthenticated();
  const user = useUser();
  const [loadingTier, setLoadingTier] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleUpgrade = async (tierId: "premium" | "institutional") => {
    if (!isAuthenticated) {
      window.location.href = "/register";
      return;
    }
    if (tierId === "institutional") {
      window.location.href = "mailto:sales@quantflow.io";
      return;
    }
    setLoadingTier(tierId);
    setError(null);
    try {
      const session = await subscriptionApi.createCheckout(tierId);
      window.location.href = session.url;
    } catch (err) {
      if (err instanceof ApiError) {
        setError(err.message);
      } else {
        setError("Unable to start checkout. Please try again.");
      }
    } finally {
      setLoadingTier(null);
    }
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Nav */}
      <header className="flex h-16 items-center justify-between px-6 md:px-12">
        <Link href="/" className="flex items-center gap-2">
          <Zap className="h-6 w-6 text-primary" />
          <span className="text-lg font-bold">QuantFlow</span>
        </Link>
        {isAuthenticated ? (
          <Link href="/dashboard">
            <Button variant="ghost" size="sm">
              Dashboard
            </Button>
          </Link>
        ) : (
          <div className="flex items-center gap-2">
            <Link href="/login">
              <Button variant="ghost" size="sm">
                Sign in
              </Button>
            </Link>
            <Link href="/register">
              <Button size="sm">Get started free</Button>
            </Link>
          </div>
        )}
      </header>

      {/* Hero */}
      <div className="py-16 text-center">
        <h1 className="text-4xl font-bold md:text-5xl">
          Institutional-grade quant research.
          <br />
          <span className="text-primary">Accessible to everyone.</span>
        </h1>
        <p className="mx-auto mt-4 max-w-xl text-muted-foreground">
          Start free with 5 symbols and 10 models. Upgrade for unlimited access
          to 50+ models, AI intelligence, and portfolio optimisation.
        </p>
      </div>

      {/* Error */}
      {error && (
        <div className="mx-auto mb-6 max-w-md px-4">
          <Alert variant="destructive">
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        </div>
      )}

      {/* Tier cards */}
      <div className="mx-auto grid max-w-6xl gap-6 px-4 pb-16 md:grid-cols-3">
        {tiers.map((tier) => {
          const Icon = tier.icon;
          const isCurrentTier = user?.subscription_tier === tier.id;

          return (
            <Card
              key={tier.id}
              className={cn(
                "relative flex flex-col",
                tier.featured && "border-primary shadow-lg shadow-primary/10",
              )}
            >
              {tier.featured && (
                <div className="absolute -top-3 left-1/2 -translate-x-1/2">
                  <Badge className="bg-primary text-primary-foreground">
                    Most popular
                  </Badge>
                </div>
              )}

              <CardHeader>
                <div className="flex items-center gap-2">
                  <Icon className="h-5 w-5 text-primary" />
                  <CardTitle className="text-lg">{tier.name}</CardTitle>
                </div>
                <div className="flex items-baseline gap-1">
                  <span className="text-3xl font-bold">{tier.price}</span>
                  <span className="text-sm text-muted-foreground">
                    /{tier.period}
                  </span>
                </div>
                <CardDescription>{tier.description}</CardDescription>
              </CardHeader>

              <CardContent className="flex-1 space-y-2">
                {tier.features.map((f) => (
                  <div key={f} className="flex items-start gap-2 text-sm">
                    <Check className="mt-0.5 h-4 w-4 shrink-0 text-green-400" />
                    {f}
                  </div>
                ))}
              </CardContent>

              <CardFooter>
                {isCurrentTier ? (
                  <Button variant="secondary" className="w-full" disabled>
                    Current plan
                  </Button>
                ) : tier.id === "free" ? (
                  <Link href="/register" className="w-full">
                    <Button variant={tier.ctaVariant} className="w-full">
                      {tier.cta}
                    </Button>
                  </Link>
                ) : (
                  <Button
                    variant={tier.ctaVariant}
                    className="w-full"
                    disabled={loadingTier === tier.id}
                    onClick={() =>
                      handleUpgrade(tier.id as "premium" | "institutional")
                    }
                  >
                    {loadingTier === tier.id && (
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    )}
                    {tier.cta}
                  </Button>
                )}
              </CardFooter>
            </Card>
          );
        })}
      </div>

      {/* FAQ */}
      <div className="mx-auto max-w-2xl px-4 pb-24">
        <h2 className="mb-6 text-center text-2xl font-bold">
          Frequently asked questions
        </h2>
        <div className="space-y-4">
          {faqs.map((faq) => (
            <div key={faq.q} className="rounded-lg border p-4">
              <p className="mb-1 font-semibold">{faq.q}</p>
              <p className="text-sm text-muted-foreground">{faq.a}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
