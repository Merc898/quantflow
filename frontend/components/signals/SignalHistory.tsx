"use client";

import { TrendingUp, TrendingDown, Minus } from "lucide-react";
import {
  cn,
  recommendationLabel,
  recommendationColor,
  formatDateTime,
  formatPercent,
} from "@/lib/utils";
import { RegimeTimeline } from "@/components/charts/RegimeTimeline";
import type { SignalResponse } from "@/lib/types";

interface SignalHistoryProps {
  signals: SignalResponse[];
  className?: string;
  showChart?: boolean;
}

/** Table + optional timeline chart of historical signals for a symbol. */
export function SignalHistory({
  signals,
  className,
  showChart = true,
}: SignalHistoryProps) {
  const sorted = [...signals].sort(
    (a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime(),
  );

  return (
    <div className={cn("space-y-4", className)}>
      {showChart && signals.length > 1 && (
        <RegimeTimeline signals={signals} height={160} />
      )}

      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b text-muted-foreground">
              <th className="pb-2 text-left font-medium">Date</th>
              <th className="pb-2 text-left font-medium">Recommendation</th>
              <th className="pb-2 text-right font-medium">Signal</th>
              <th className="pb-2 text-right font-medium">Confidence</th>
              <th className="pb-2 text-right font-medium">Position</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((sig, i) => {
              const isBull = sig.signal_strength > 0.1;
              const isBear = sig.signal_strength < -0.1;
              const Icon = isBull ? TrendingUp : isBear ? TrendingDown : Minus;
              const iconColor = isBull
                ? "text-green-400"
                : isBear
                  ? "text-red-400"
                  : "text-amber-400";

              return (
                <tr key={i} className="border-b border-border/50 hover:bg-muted/30">
                  <td className="py-2 text-muted-foreground">
                    {formatDateTime(sig.timestamp)}
                  </td>
                  <td className="py-2">
                    <div className="flex items-center gap-1.5">
                      <Icon className={cn("h-3.5 w-3.5", iconColor)} />
                      <span className={recommendationColor(sig.recommendation)}>
                        {recommendationLabel(sig.recommendation)}
                      </span>
                    </div>
                  </td>
                  <td className="py-2 text-right tabular-nums">
                    <span
                      style={{
                        color:
                          sig.signal_strength > 0 ? "#22c55e" : "#ef4444",
                      }}
                    >
                      {sig.signal_strength >= 0 ? "+" : ""}
                      {(sig.signal_strength * 100).toFixed(1)}
                    </span>
                  </td>
                  <td className="py-2 text-right tabular-nums">
                    {formatPercent(sig.confidence)}
                  </td>
                  <td className="py-2 text-right tabular-nums">
                    {formatPercent(sig.suggested_position_size)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {sorted.length === 0 && (
        <p className="py-6 text-center text-sm text-muted-foreground">
          No signal history available.
        </p>
      )}
    </div>
  );
}
