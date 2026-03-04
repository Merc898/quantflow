"use client";

import { useEffect, useState, useCallback } from "react";
import Link from "next/link";
import { RefreshCw, TrendingUp, BarChart3, ShieldCheck, Activity } from "lucide-react";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { RecommendationCard } from "@/components/signals/RecommendationCard";
import { signals as signalsApi, ApiError } from "@/lib/api";
import { useUser, useStore } from "@/lib/store";
import { createSignalSocket } from "@/lib/websocket";
import {
  regimeLabel,
  regimeColor,
  formatPercent,
  cn,
} from "@/lib/utils";
import type { SignalResponse } from "@/lib/types";

// ---------------------------------------------------------------------------
// Stat card
// ---------------------------------------------------------------------------
function StatCard({
  title,
  value,
  subtitle,
  icon: Icon,
  trend,
}: {
  title: string;
  value: string;
  subtitle?: string;
  icon: React.ElementType;
  trend?: "up" | "down" | "neutral";
}) {
  return (
    <Card>
      <CardContent className="pt-6">
        <div className="flex items-start justify-between">
          <div>
            <p className="text-sm font-medium text-muted-foreground">{title}</p>
            <p
              className={cn(
                "mt-1 text-2xl font-bold",
                trend === "up" && "text-green-400",
                trend === "down" && "text-red-400",
              )}
            >
              {value}
            </p>
            {subtitle && (
              <p className="mt-0.5 text-xs text-muted-foreground">{subtitle}</p>
            )}
          </div>
          <div className="rounded-lg bg-primary/10 p-2">
            <Icon className="h-5 w-5 text-primary" />
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

// ---------------------------------------------------------------------------
// Dashboard page
// ---------------------------------------------------------------------------
export default function DashboardPage() {
  const user = useUser();
  const { upsertSignal, liveSignals, regime, macroSentiment } = useStore();
  const [watchlistSignals, setWatchlistSignals] = useState<SignalResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  const watchlist = user?.watchlist ?? [];

  const loadSignals = useCallback(async () => {
    if (!watchlist.length) {
      setLoading(false);
      return;
    }
    try {
      const result = await signalsApi.batch(watchlist);
      const sigs = result.results;
      sigs.forEach((s) => upsertSignal(s));
      setWatchlistSignals(sigs);
    } catch {
      // Keep existing data
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, [watchlist.join(",")]);

  useEffect(() => {
    loadSignals();
  }, [loadSignals]);

  // Subscribe to live signals for each watchlist symbol
  useEffect(() => {
    if (!watchlist.length) return;
    const sockets = watchlist.map((sym) => {
      const sock = createSignalSocket(sym);
      sock.subscribe((msg) => {
        if (msg.type === "signal" && msg.symbol === sym) {
          upsertSignal(msg.data);
          setWatchlistSignals((prev) => {
            const idx = prev.findIndex((s) => s.symbol === sym);
            if (idx === -1) return [...prev, msg.data];
            const next = [...prev];
            next[idx] = msg.data;
            return next;
          });
        }
      });
      return sock;
    });
    return () => sockets.forEach((s) => s.destroy());
  }, [watchlist.join(",")]);

  const handleRefresh = () => {
    setRefreshing(true);
    loadSignals();
  };

  // Aggregate stats from watchlist signals
  const bullishCount = watchlistSignals.filter(
    (s) => s.signal_strength > 0.1,
  ).length;
  const avgConfidence =
    watchlistSignals.length > 0
      ? watchlistSignals.reduce((sum, s) => sum + s.confidence, 0) /
        watchlistSignals.length
      : 0;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Dashboard</h1>
          <p className="text-sm text-muted-foreground">
            Welcome back, {user?.email?.split("@")[0] ?? "Trader"}
          </p>
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={handleRefresh}
          disabled={refreshing}
        >
          <RefreshCw
            className={cn("mr-2 h-4 w-4", refreshing && "animate-spin")}
          />
          Refresh
        </Button>
      </div>

      {/* Stats row */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard
          title="Watchlist"
          value={`${watchlist.length} symbols`}
          subtitle={`${bullishCount} bullish`}
          icon={TrendingUp}
          trend={bullishCount > watchlist.length / 2 ? "up" : "down"}
        />
        <StatCard
          title="Avg Confidence"
          value={formatPercent(avgConfidence)}
          subtitle="Across watchlist"
          icon={BarChart3}
        />
        <StatCard
          title="Market Regime"
          value={regime ? regimeLabel(regime) : "—"}
          icon={Activity}
        />
        <StatCard
          title="Macro Sentiment"
          value={
            macroSentiment !== null
              ? `${macroSentiment >= 0 ? "+" : ""}${(macroSentiment * 100).toFixed(0)}`
              : "—"
          }
          icon={ShieldCheck}
          trend={
            macroSentiment !== null
              ? macroSentiment > 0.1
                ? "up"
                : macroSentiment < -0.1
                  ? "down"
                  : "neutral"
              : "neutral"
          }
        />
      </div>

      {/* Live signal feed */}
      <div>
        <div className="mb-4 flex items-center justify-between">
          <h2 className="text-lg font-semibold">Live Signal Feed</h2>
          <Link href="/settings">
            <Button variant="ghost" size="sm">
              Manage watchlist
            </Button>
          </Link>
        </div>

        {loading ? (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {[...Array(3)].map((_, i) => (
              <Skeleton key={i} className="h-56 w-full rounded-lg" />
            ))}
          </div>
        ) : watchlistSignals.length > 0 ? (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {watchlistSignals.map((sig) => (
              <Link key={sig.symbol} href={`/signals/${sig.symbol}`}>
                <RecommendationCard signal={sig} size="sm" />
              </Link>
            ))}
          </div>
        ) : (
          <Card className="border-dashed">
            <CardContent className="py-12 text-center">
              <p className="text-muted-foreground">
                Your watchlist is empty.{" "}
                <Link href="/screener" className="text-primary hover:underline">
                  Browse the screener
                </Link>{" "}
                or{" "}
                <Link href="/settings" className="text-primary hover:underline">
                  add symbols
                </Link>
                .
              </p>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}
