"use client";

import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceDot,
} from "recharts";
import { cn, formatPercent } from "@/lib/utils";
import type { FrontierPoint } from "@/lib/types";

interface EfficientFrontierProps {
  frontier: FrontierPoint[];
  optimal: FrontierPoint;
  className?: string;
  height?: number;
}

/** Scatter plot of the mean-variance efficient frontier. */
export function EfficientFrontierChart({
  frontier,
  optimal,
  className,
  height = 350,
}: EfficientFrontierProps) {
  const frontierData = frontier.map((p) => ({
    x: p.expected_volatility * 100,
    y: p.expected_return * 100,
    sharpe: p.sharpe_ratio,
  }));

  return (
    <div className={cn("w-full", className)} style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis
            type="number"
            dataKey="x"
            name="Volatility"
            unit="%"
            label={{
              value: "Volatility (%)",
              position: "insideBottom",
              offset: -10,
              fill: "#6b7280",
              fontSize: 12,
            }}
            stroke="#6b7280"
            tick={{ fontSize: 11 }}
          />
          <YAxis
            type="number"
            dataKey="y"
            name="Return"
            unit="%"
            label={{
              value: "Expected Return (%)",
              angle: -90,
              position: "insideLeft",
              fill: "#6b7280",
              fontSize: 12,
            }}
            stroke="#6b7280"
            tick={{ fontSize: 11 }}
          />
          <Tooltip
            cursor={{ strokeDasharray: "3 3" }}
            content={({ active, payload }) => {
              if (!active || !payload?.length) return null;
              const d = payload[0].payload as {
                x: number;
                y: number;
                sharpe: number;
              };
              return (
                <div className="rounded-md border bg-card p-2 text-xs shadow-md">
                  <p>Return: {d.y.toFixed(2)}%</p>
                  <p>Volatility: {d.x.toFixed(2)}%</p>
                  <p>Sharpe: {d.sharpe.toFixed(3)}</p>
                </div>
              );
            }}
          />
          <Scatter data={frontierData} fill="#3b82f6" opacity={0.7} />
          {/* Optimal portfolio marker */}
          <ReferenceDot
            x={optimal.expected_volatility * 100}
            y={optimal.expected_return * 100}
            r={8}
            fill="#22c55e"
            stroke="#ffffff"
            strokeWidth={2}
            label={{
              value: "★ Optimal",
              position: "top",
              fill: "#22c55e",
              fontSize: 11,
            }}
          />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}
