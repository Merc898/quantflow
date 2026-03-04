"use client";

import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import { cn, formatDate } from "@/lib/utils";
import type { SignalResponse } from "@/lib/types";

interface RegimeTimelineProps {
  signals: SignalResponse[];
  className?: string;
  height?: number;
}

/**
 * Area chart of signal strength over time, with regime-coloured shading.
 */
export function RegimeTimeline({
  signals,
  className,
  height = 200,
}: RegimeTimelineProps) {
  const data = [...signals]
    .sort(
      (a, b) =>
        new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime(),
    )
    .map((s) => ({
      date: formatDate(s.timestamp),
      signal: Number((s.signal_strength * 100).toFixed(1)),
      confidence: Number((s.confidence * 100).toFixed(1)),
    }));

  return (
    <div className={cn("w-full", className)} style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ top: 5, right: 10, left: -10, bottom: 5 }}>
          <defs>
            <linearGradient id="bullGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#22c55e" stopOpacity={0.4} />
              <stop offset="95%" stopColor="#22c55e" stopOpacity={0} />
            </linearGradient>
            <linearGradient id="bearGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#ef4444" stopOpacity={0.4} />
              <stop offset="95%" stopColor="#ef4444" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis
            dataKey="date"
            stroke="#6b7280"
            tick={{ fontSize: 10 }}
            interval="preserveStartEnd"
          />
          <YAxis
            domain={[-100, 100]}
            stroke="#6b7280"
            tick={{ fontSize: 10 }}
            tickFormatter={(v: number) => `${v}`}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "#1f2937",
              border: "1px solid #374151",
              borderRadius: "6px",
              fontSize: 12,
            }}
            formatter={(value: number, name: string) => [
              `${value}%`,
              name === "signal" ? "Signal" : "Confidence",
            ]}
          />
          <ReferenceLine y={0} stroke="#6b7280" strokeDasharray="4 2" />
          <Area
            type="monotone"
            dataKey="signal"
            stroke="#3b82f6"
            fill="url(#bullGrad)"
            strokeWidth={2}
            dot={false}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
