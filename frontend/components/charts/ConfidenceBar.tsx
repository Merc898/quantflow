"use client";

import { cn, formatPercent } from "@/lib/utils";

interface ConfidenceBarProps {
  confidence: number;   // [0, 1]
  label?: string;
  className?: string;
  showLabel?: boolean;
}

/** Horizontal progress bar for model confidence, colour-coded by level. */
export function ConfidenceBar({
  confidence,
  label = "Confidence",
  className,
  showLabel = true,
}: ConfidenceBarProps) {
  const pct = Math.round(confidence * 100);
  const color =
    confidence >= 0.75
      ? "bg-green-500"
      : confidence >= 0.5
        ? "bg-amber-500"
        : "bg-red-500";

  return (
    <div className={cn("space-y-1", className)}>
      {showLabel && (
        <div className="flex items-center justify-between text-xs">
          <span className="text-muted-foreground">{label}</span>
          <span className="font-medium">{formatPercent(confidence)}</span>
        </div>
      )}
      <div className="h-2 w-full overflow-hidden rounded-full bg-secondary">
        <div
          className={cn("h-full rounded-full transition-all", color)}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}
