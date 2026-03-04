"use client";

import { TrendingDown, TrendingUp, Minus, ShieldAlert, Clock } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { SignalGauge } from "@/components/charts/SignalGauge";
import { ConfidenceBar } from "@/components/charts/ConfidenceBar";
import {
  cn,
  recommendationLabel,
  recommendationBgColor,
  formatPercent,
  formatDateTime,
  regimeLabel,
  regimeColor,
} from "@/lib/utils";
import type { SignalResponse } from "@/lib/types";

interface RecommendationCardProps {
  signal: SignalResponse;
  className?: string;
  size?: "sm" | "md" | "lg";
}

/** Large recommendation card shown on the signal deep-dive page and dashboard. */
export function RecommendationCard({
  signal,
  className,
  size = "md",
}: RecommendationCardProps) {
  const isBull = signal.signal_strength > 0.1;
  const isBear = signal.signal_strength < -0.1;

  const Icon = isBull ? TrendingUp : isBear ? TrendingDown : Minus;
  const iconColor = isBull
    ? "text-green-400"
    : isBear
      ? "text-red-400"
      : "text-amber-400";

  return (
    <Card
      className={cn(
        "border-2 transition-colors",
        recommendationBgColor(signal.recommendation),
        className,
      )}
    >
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-xl font-bold">{signal.symbol}</span>
            <Icon className={cn("h-5 w-5", iconColor)} />
          </div>
          <Badge variant="outline" className="text-xs text-muted-foreground">
            <Clock className="mr-1 h-3 w-3" />
            {formatDateTime(signal.timestamp)}
          </Badge>
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        <div className="flex items-center gap-6">
          {/* Gauge */}
          <SignalGauge
            signal={signal.signal_strength}
            recommendation={signal.recommendation}
            size={size}
          />

          {/* Stats */}
          <div className="flex-1 space-y-3">
            <div>
              <p className="text-xs text-muted-foreground">Recommendation</p>
              <p className="text-lg font-bold">
                {recommendationLabel(signal.recommendation)}
              </p>
            </div>

            <ConfidenceBar confidence={signal.confidence} />

            <div className="grid grid-cols-2 gap-3">
              <div>
                <p className="text-xs text-muted-foreground">Position size</p>
                <p className="font-semibold">
                  {formatPercent(signal.suggested_position_size)}
                </p>
              </div>
              <div>
                <p className="text-xs text-muted-foreground">Regime</p>
                <p className={cn("text-sm font-medium", regimeColor(signal.regime))}>
                  {regimeLabel(signal.regime)}
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Risk warnings */}
        {signal.risk_warnings.length > 0 && (
          <div className="space-y-1">
            {signal.risk_warnings.map((warn, i) => (
              <div
                key={i}
                className="flex items-start gap-2 rounded-md bg-amber-500/10 px-3 py-1.5 text-xs text-amber-400"
              >
                <ShieldAlert className="mt-0.5 h-3 w-3 shrink-0" />
                {warn}
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
