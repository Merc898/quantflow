import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";
import type { Recommendation, MarketRegime, SubscriptionTier } from "./types";

/** Merge Tailwind classes safely. */
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/** Format a number as a percentage string. */
export function formatPercent(value: number, decimals = 1): string {
  return `${(value * 100).toFixed(decimals)}%`;
}

/** Format a number as a signed percentage string (e.g. "+3.4%"). */
export function formatSignedPercent(value: number, decimals = 1): string {
  const pct = (value * 100).toFixed(decimals);
  return value >= 0 ? `+${pct}%` : `${pct}%`;
}

/** Format a large number with K/M/B suffix. */
export function formatCompact(value: number): string {
  if (Math.abs(value) >= 1e9) return `${(value / 1e9).toFixed(1)}B`;
  if (Math.abs(value) >= 1e6) return `${(value / 1e6).toFixed(1)}M`;
  if (Math.abs(value) >= 1e3) return `${(value / 1e3).toFixed(1)}K`;
  return value.toFixed(2);
}

/** Format a date string to a human-readable form. */
export function formatDate(isoString: string): string {
  return new Date(isoString).toLocaleDateString("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

/** Format a datetime string with time. */
export function formatDateTime(isoString: string): string {
  return new Date(isoString).toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

/** Map a Recommendation to a display label. */
export function recommendationLabel(rec: Recommendation): string {
  return rec.replace(/_/g, " ");
}

/** Map a Recommendation to a Tailwind color class. */
export function recommendationColor(rec: Recommendation): string {
  const map: Record<Recommendation, string> = {
    STRONG_BUY: "text-green-400",
    BUY: "text-green-500",
    WEAK_BUY: "text-green-600",
    HOLD: "text-amber-400",
    WEAK_SELL: "text-red-400",
    SELL: "text-red-500",
    STRONG_SELL: "text-red-600",
  };
  return map[rec] ?? "text-foreground";
}

/** Map a Recommendation to a background color class. */
export function recommendationBgColor(rec: Recommendation): string {
  const map: Record<Recommendation, string> = {
    STRONG_BUY: "bg-green-400/20 border-green-400",
    BUY: "bg-green-500/20 border-green-500",
    WEAK_BUY: "bg-green-600/20 border-green-600",
    HOLD: "bg-amber-400/20 border-amber-400",
    WEAK_SELL: "bg-red-400/20 border-red-400",
    SELL: "bg-red-500/20 border-red-500",
    STRONG_SELL: "bg-red-600/20 border-red-600",
  };
  return map[rec] ?? "";
}

/** Map a signal strength [-1, +1] to a color. */
export function signalColor(strength: number): string {
  if (strength > 0.5) return "#22c55e";
  if (strength > 0.1) return "#86efac";
  if (strength > -0.1) return "#f59e0b";
  if (strength > -0.5) return "#fca5a5";
  return "#ef4444";
}

/** Map a market regime to a display label. */
export function regimeLabel(regime: MarketRegime): string {
  const map: Record<MarketRegime, string> = {
    BULL_LOW_VOL: "Bull · Low Vol",
    BULL_HIGH_VOL: "Bull · High Vol",
    BEAR_LOW_VOL: "Bear · Low Vol",
    BEAR_HIGH_VOL: "Bear · High Vol",
    SIDEWAYS: "Sideways",
    CRISIS: "Crisis",
    RECOVERY: "Recovery",
  };
  return map[regime] ?? regime;
}

/** Map a market regime to a Tailwind color. */
export function regimeColor(regime: MarketRegime): string {
  if (regime.startsWith("BULL")) return "text-green-400";
  if (regime === "CRISIS") return "text-red-500";
  if (regime.startsWith("BEAR")) return "text-red-400";
  if (regime === "RECOVERY") return "text-blue-400";
  return "text-amber-400";
}

/** Map a subscription tier to a display label. */
export function tierLabel(tier: SubscriptionTier): string {
  const map: Record<SubscriptionTier, string> = {
    free: "Free",
    premium: "Premium",
    institutional: "Institutional",
  };
  return map[tier];
}

/** Clamp a number between min and max. */
export function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

/** Convert a signal [-1, +1] to a [0, 100] gauge value. */
export function signalToGauge(signal: number): number {
  return clamp(((signal + 1) / 2) * 100, 0, 100);
}
