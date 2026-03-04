"use client";

import { useState } from "react";
import { Plus, Trash2, Loader2, Lock } from "lucide-react";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { EfficientFrontierChart } from "@/components/charts/EfficientFrontier";
import {
  portfolio as portfolioApi,
  type OptimizationMethod,
  type CovarianceMethod,
  ApiError,
} from "@/lib/api";
import { useIsProTier } from "@/lib/store";
import { formatPercent, cn } from "@/lib/utils";
import type { OptimizedPortfolio, EfficientFrontierResponse } from "@/lib/types";

export default function PortfolioPage() {
  const isProTier = useIsProTier();
  const [symbols, setSymbols] = useState<string[]>(["AAPL", "MSFT", "GOOGL"]);
  const [newSymbol, setNewSymbol] = useState("");
  const [method, setMethod] = useState<OptimizationMethod>("hrp");
  const [covMethod, setCovMethod] = useState<CovarianceMethod>("ledoit_wolf");
  const [result, setResult] = useState<OptimizedPortfolio | null>(null);
  const [frontier, setFrontier] = useState<EfficientFrontierResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const addSymbol = () => {
    const sym = newSymbol.trim().toUpperCase();
    if (sym && !symbols.includes(sym)) {
      setSymbols((prev) => [...prev, sym]);
    }
    setNewSymbol("");
  };

  const removeSymbol = (sym: string) => {
    setSymbols((prev) => prev.filter((s) => s !== sym));
  };

  const handleOptimize = async () => {
    if (symbols.length < 2) {
      setError("Please add at least 2 symbols.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const [opt, front] = await Promise.all([
        portfolioApi.optimize({
          symbols,
          method,
          covariance_method: covMethod,
        }),
        portfolioApi.efficientFrontier(symbols, 30),
      ]);
      setResult(opt);
      setFrontier(front);
    } catch (err) {
      if (err instanceof ApiError) {
        setError(err.message);
      } else {
        setError("Optimisation failed. Please check your symbols and try again.");
      }
    } finally {
      setLoading(false);
    }
  };

  if (!isProTier) {
    return (
      <div className="flex flex-col items-center justify-center gap-4 py-24 text-center">
        <Lock className="h-10 w-10 text-muted-foreground" />
        <h2 className="text-xl font-semibold">Portfolio Optimizer</h2>
        <p className="max-w-sm text-sm text-muted-foreground">
          Portfolio optimisation (MVO, HRP, Black-Litterman) is available on
          Premium and Institutional plans.
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
        <h1 className="text-2xl font-bold">Portfolio Optimiser</h1>
        <p className="text-sm text-muted-foreground">
          MVO · HRP · Black-Litterman · Ledoit-Wolf covariance
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        {/* Controls */}
        <Card className="lg:col-span-1">
          <CardHeader>
            <CardTitle className="text-base">Configuration</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Symbol list */}
            <div className="space-y-2">
              <Label>Symbols</Label>
              <div className="flex flex-wrap gap-1.5">
                {symbols.map((sym) => (
                  <Badge
                    key={sym}
                    variant="secondary"
                    className="cursor-pointer gap-1"
                    onClick={() => removeSymbol(sym)}
                  >
                    {sym}
                    <Trash2 className="h-3 w-3" />
                  </Badge>
                ))}
              </div>
              <div className="flex gap-2">
                <Input
                  placeholder="Add symbol…"
                  value={newSymbol}
                  onChange={(e) => setNewSymbol(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && addSymbol()}
                  className="text-sm"
                />
                <Button size="sm" variant="outline" onClick={addSymbol}>
                  <Plus className="h-4 w-4" />
                </Button>
              </div>
            </div>

            {/* Method */}
            <div className="space-y-1.5">
              <Label>Optimisation method</Label>
              <Select
                value={method}
                onValueChange={(v) => setMethod(v as OptimizationMethod)}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="hrp">HRP (Hierarchical Risk Parity)</SelectItem>
                  <SelectItem value="mvo">MVO (Mean-Variance)</SelectItem>
                  <SelectItem value="black_litterman">Black-Litterman</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Covariance */}
            <div className="space-y-1.5">
              <Label>Covariance estimator</Label>
              <Select
                value={covMethod}
                onValueChange={(v) => setCovMethod(v as CovarianceMethod)}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="ledoit_wolf">Ledoit-Wolf</SelectItem>
                  <SelectItem value="sample">Sample</SelectItem>
                  <SelectItem value="oas">OAS</SelectItem>
                  <SelectItem value="rmt">RMT (Random Matrix)</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {error && (
              <Alert variant="destructive">
                <AlertDescription className="text-xs">{error}</AlertDescription>
              </Alert>
            )}

            <Button
              className="w-full"
              onClick={handleOptimize}
              disabled={loading || symbols.length < 2}
            >
              {loading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
              Optimise Portfolio
            </Button>
          </CardContent>
        </Card>

        {/* Results */}
        <div className="space-y-6 lg:col-span-2">
          {result && (
            <>
              {/* Weights */}
              <Card>
                <CardHeader>
                  <CardTitle className="text-base">Optimal Weights</CardTitle>
                  <CardDescription>
                    Method: {result.method} · Sharpe:{" "}
                    {result.sharpe_ratio.toFixed(3)} · Return:{" "}
                    {formatPercent(result.expected_return)} · Vol:{" "}
                    {formatPercent(result.expected_volatility)}
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    {result.weights
                      .sort((a, b) => b.weight - a.weight)
                      .map((w) => (
                        <div key={w.symbol}>
                          <div className="flex justify-between text-sm">
                            <span className="font-medium">{w.symbol}</span>
                            <span>{formatPercent(w.weight)}</span>
                          </div>
                          <div className="mt-1 h-2 overflow-hidden rounded-full bg-secondary">
                            <div
                              className="h-full rounded-full bg-primary transition-all"
                              style={{ width: `${w.weight * 100}%` }}
                            />
                          </div>
                        </div>
                      ))}
                  </div>
                </CardContent>
              </Card>

              {/* Efficient frontier */}
              {frontier && (
                <Card>
                  <CardHeader>
                    <CardTitle className="text-base">Efficient Frontier</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <EfficientFrontierChart
                      frontier={frontier.frontier}
                      optimal={frontier.optimal}
                      height={300}
                    />
                  </CardContent>
                </Card>
              )}
            </>
          )}

          {!result && !loading && (
            <Card className="border-dashed">
              <CardContent className="py-16 text-center text-sm text-muted-foreground">
                Configure your portfolio above and click <strong>Optimise</strong>{" "}
                to see results.
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
