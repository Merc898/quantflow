"use client";

import {
  RadialBarChart,
  RadialBar,
  ResponsiveContainer,
  PolarAngleAxis,
} from "recharts";
import { cn, signalColor, signalToGauge } from "@/lib/utils";
import type { Recommendation } from "@/lib/types";
import { recommendationLabel, recommendationColor } from "@/lib/utils";

interface SignalGaugeProps {
  signal: number;       // [-1, +1]
  recommendation: Recommendation;
  size?: "sm" | "md" | "lg";
  className?: string;
}

const sizeMap = {
  sm: { outer: 80, inner: 60, fontSize: "text-sm" },
  md: { outer: 120, inner: 90, fontSize: "text-base" },
  lg: { outer: 160, inner: 120, fontSize: "text-xl" },
};

/**
 * Radial gauge showing Buy/Sell signal strength.
 * Green → bullish (signal > 0), Red → bearish (signal < 0), Amber → neutral.
 */
export function SignalGauge({
  signal,
  recommendation,
  size = "md",
  className,
}: SignalGaugeProps) {
  const gaugeValue = signalToGauge(signal);
  const color = signalColor(signal);
  const { outer, inner, fontSize } = sizeMap[size];

  const data = [{ value: gaugeValue, fill: color }];

  return (
    <div
      className={cn("relative flex flex-col items-center", className)}
      style={{ width: outer, height: outer }}
    >
      <ResponsiveContainer width="100%" height="100%">
        <RadialBarChart
          cx="50%"
          cy="50%"
          innerRadius={`${Math.round((inner / outer) * 100)}%`}
          outerRadius="100%"
          barSize={8}
          data={data}
          startAngle={225}
          endAngle={-45}
        >
          <PolarAngleAxis
            type="number"
            domain={[0, 100]}
            angleAxisId={0}
            tick={false}
          />
          {/* Background track */}
          <RadialBar
            background={{ fill: "#1f2937" }}
            dataKey="value"
            cornerRadius={4}
            angleAxisId={0}
          />
        </RadialBarChart>
      </ResponsiveContainer>

      {/* Centre label */}
      <div
        className="absolute inset-0 flex flex-col items-center justify-center text-center"
        style={{ fontSize: size === "sm" ? 10 : size === "md" ? 12 : 14 }}
      >
        <span
          className={cn("font-bold leading-none", fontSize)}
          style={{ color }}
        >
          {signal >= 0 ? "+" : ""}
          {(signal * 100).toFixed(0)}
        </span>
        <span className="mt-0.5 text-xs text-muted-foreground">
          {recommendationLabel(recommendation)}
        </span>
      </div>
    </div>
  );
}
