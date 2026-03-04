"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
  Cell,
} from "recharts";
import { cn } from "@/lib/utils";

interface ModelContribChartProps {
  contributions: Record<string, number>;
  className?: string;
  height?: number;
}

/**
 * Horizontal waterfall-style bar chart showing each model's
 * contribution to the final signal (positive = bullish, negative = bearish).
 */
export function ModelContribChart({
  contributions,
  className,
  height = 300,
}: ModelContribChartProps) {
  const data = Object.entries(contributions)
    .map(([name, value]) => ({ name: formatModelName(name), value }))
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value));

  return (
    <div className={cn("w-full", className)} style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 5, right: 30, left: 100, bottom: 5 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" horizontal={false} />
          <XAxis
            type="number"
            domain={[-1, 1]}
            tickFormatter={(v: number) => `${(v * 100).toFixed(0)}`}
            stroke="#6b7280"
            tick={{ fontSize: 11 }}
          />
          <YAxis
            type="category"
            dataKey="name"
            width={95}
            stroke="#6b7280"
            tick={{ fontSize: 11 }}
          />
          <Tooltip
            formatter={(value: number) => [
              `${(value * 100).toFixed(1)}`,
              "Signal contribution",
            ]}
            contentStyle={{
              backgroundColor: "#1f2937",
              border: "1px solid #374151",
              borderRadius: "6px",
              fontSize: 12,
            }}
          />
          <ReferenceLine x={0} stroke="#6b7280" strokeWidth={1} />
          <Bar dataKey="value" radius={[0, 4, 4, 0]} maxBarSize={20}>
            {data.map((entry, idx) => (
              <Cell
                key={idx}
                fill={entry.value >= 0 ? "#22c55e" : "#ef4444"}
                fillOpacity={0.8}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function formatModelName(name: string): string {
  return name
    .replace(/_/g, " ")
    .replace(/model$/i, "")
    .trim()
    .replace(/\b\w/g, (c) => c.toUpperCase());
}
