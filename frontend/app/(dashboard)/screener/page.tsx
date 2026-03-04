"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Search, TrendingUp, TrendingDown, Minus, Loader2, Lock } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { signals as signalsApi, ApiError } from "@/lib/api";
import { useIsProTier } from "@/lib/store";
import {
  recommendationLabel,
  recommendationColor,
  formatPercent,
  regimeLabel,
  cn,
} from "@/lib/utils";
import type { ScreenerEntry } from "@/lib/types";

export default function ScreenerPage() {
  const isProTier = useIsProTier();
  const [data, setData] = useState<ScreenerEntry[]>([]);
  const [filtered, setFiltered] = useState<ScreenerEntry[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [minConf, setMinConf] = useState(0.6);
  const [search, setSearch] = useState("");

  const loadScreener = async () => {
    setLoading(true);
    setError(null);
    try {
      const results = await signalsApi.screener(minConf, 100);
      setData(results);
      setFiltered(results);
    } catch (err) {
      if (err instanceof ApiError) {
        setError(err.message);
      } else {
        setError("Failed to load screener data.");
      }
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (isProTier) loadScreener();
  }, [isProTier]);

  useEffect(() => {
    const q = search.trim().toUpperCase();
    setFiltered(data.filter((d) => !q || d.symbol.includes(q)));
  }, [search, data]);

  if (!isProTier) {
    return (
      <div className="flex flex-col items-center justify-center gap-4 py-24 text-center">
        <Lock className="h-10 w-10 text-muted-foreground" />
        <h2 className="text-xl font-semibold">Signal Screener</h2>
        <p className="max-w-sm text-sm text-muted-foreground">
          The screener ranks the entire tracked universe by signal strength.
          Available on Premium and Institutional plans.
        </p>
        <Button asChild>
          <a href="/pricing">Upgrade to Premium</a>
        </Button>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Signal Screener</h1>
        <p className="text-sm text-muted-foreground">
          Universe ranked by signal strength — updated every 4 hours
        </p>
      </div>

      {/* Filters */}
      <Card>
        <CardContent className="flex flex-wrap items-end gap-4 pt-6">
          <div className="space-y-1.5">
            <Label>Min confidence</Label>
            <Input
              type="number"
              min={0}
              max={1}
              step={0.05}
              value={minConf}
              onChange={(e) => setMinConf(Number(e.target.value))}
              className="w-28"
            />
          </div>
          <div className="flex-1 space-y-1.5">
            <Label>Filter by symbol</Label>
            <div className="relative">
              <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
              <Input
                placeholder="AAPL, MSFT…"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className="pl-9"
              />
            </div>
          </div>
          <Button onClick={loadScreener} disabled={loading}>
            {loading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
            Scan
          </Button>
        </CardContent>
      </Card>

      {error && (
        <Alert variant="destructive">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* Results table */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">
            {filtered.length} signals
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b text-muted-foreground">
                  <th className="pb-3 text-left font-medium">#</th>
                  <th className="pb-3 text-left font-medium">Symbol</th>
                  <th className="pb-3 text-left font-medium">Recommendation</th>
                  <th className="pb-3 text-right font-medium">Signal</th>
                  <th className="pb-3 text-right font-medium">Confidence</th>
                  <th className="pb-3 text-right font-medium">Position</th>
                  <th className="pb-3 text-left font-medium">Regime</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((entry, i) => {
                  const isBull = entry.signal_strength > 0.1;
                  const isBear = entry.signal_strength < -0.1;
                  const Icon = isBull ? TrendingUp : isBear ? TrendingDown : Minus;
                  const iconColor = isBull
                    ? "text-green-400"
                    : isBear
                      ? "text-red-400"
                      : "text-amber-400";

                  return (
                    <tr
                      key={entry.symbol}
                      className="border-b border-border/50 hover:bg-muted/30"
                    >
                      <td className="py-3 text-muted-foreground">{i + 1}</td>
                      <td className="py-3">
                        <Link
                          href={`/signals/${entry.symbol}`}
                          className="font-semibold text-primary hover:underline"
                        >
                          {entry.symbol}
                        </Link>
                      </td>
                      <td className="py-3">
                        <div className="flex items-center gap-1.5">
                          <Icon className={cn("h-3.5 w-3.5", iconColor)} />
                          <span className={recommendationColor(entry.recommendation)}>
                            {recommendationLabel(entry.recommendation)}
                          </span>
                        </div>
                      </td>
                      <td className="py-3 text-right tabular-nums">
                        <span
                          style={{
                            color:
                              entry.signal_strength >= 0
                                ? "#22c55e"
                                : "#ef4444",
                          }}
                        >
                          {entry.signal_strength >= 0 ? "+" : ""}
                          {(entry.signal_strength * 100).toFixed(1)}
                        </span>
                      </td>
                      <td className="py-3 text-right tabular-nums">
                        {formatPercent(entry.confidence)}
                      </td>
                      <td className="py-3 text-right tabular-nums">
                        {formatPercent(entry.suggested_position_size)}
                      </td>
                      <td className="py-3 text-xs text-muted-foreground">
                        {regimeLabel(entry.regime)}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>

            {filtered.length === 0 && !loading && (
              <p className="py-8 text-center text-sm text-muted-foreground">
                No signals found. Try lowering the minimum confidence.
              </p>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
