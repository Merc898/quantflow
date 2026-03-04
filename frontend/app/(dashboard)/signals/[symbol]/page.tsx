"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { Loader2, RefreshCw, Lock } from "lucide-react";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Skeleton } from "@/components/ui/skeleton";
import { RecommendationCard } from "@/components/signals/RecommendationCard";
import { SignalHistory } from "@/components/signals/SignalHistory";
import { ModelContribChart } from "@/components/charts/ModelContribChart";
import { signals as signalsApi, intelligence as intelligenceApi, ApiError } from "@/lib/api";
import { useIsProTier } from "@/lib/store";
import { createSignalSocket } from "@/lib/websocket";
import { formatDate } from "@/lib/utils";
import type { SignalResponse, ExplainabilityReport, IntelligenceReport } from "@/lib/types";

export default function SignalPage() {
  const params = useParams<{ symbol: string }>();
  const symbol = params.symbol.toUpperCase();
  const isProTier = useIsProTier();

  const [signal, setSignal] = useState<SignalResponse | null>(null);
  const [history, setHistory] = useState<SignalResponse[]>([]);
  const [explain, setExplain] = useState<ExplainabilityReport | null>(null);
  const [intel, setIntel] = useState<IntelligenceReport | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let isMounted = true;

    const load = async () => {
      setLoading(true);
      setError(null);
      try {
        const [sig, hist] = await Promise.all([
          signalsApi.get(symbol),
          signalsApi.history(symbol, isProTier ? 90 : 30),
        ]);
        if (!isMounted) return;
        setSignal(sig);
        setHistory(hist.signals);

        // Premium extras (non-blocking)
        if (isProTier) {
          Promise.all([
            signalsApi.explain(symbol).catch(() => null),
            intelligenceApi.get(symbol).catch(() => null),
          ]).then(([exp, int]) => {
            if (!isMounted) return;
            if (exp) setExplain(exp);
            if (int) setIntel(int);
          });
        }
      } catch (err) {
        if (!isMounted) return;
        if (err instanceof ApiError && err.status === 404) {
          setError(`No signal data found for ${symbol}.`);
        } else {
          setError("Failed to load signal data. Please try again.");
        }
      } finally {
        if (isMounted) setLoading(false);
      }
    };

    load();

    // Live updates via WebSocket
    const sock = createSignalSocket(symbol);
    const unsub = sock.subscribe((msg) => {
      if (msg.type === "signal" && msg.symbol === symbol && isMounted) {
        setSignal(msg.data);
        setHistory((prev) => [msg.data, ...prev].slice(0, 100));
      }
    });

    return () => {
      isMounted = false;
      unsub();
      sock.destroy();
    };
  }, [symbol, isProTier]);

  if (loading) {
    return (
      <div className="space-y-6">
        <Skeleton className="h-12 w-48" />
        <Skeleton className="h-56 w-full" />
        <Skeleton className="h-80 w-full" />
      </div>
    );
  }

  if (error) {
    return (
      <Alert variant="destructive" className="mt-8">
        <AlertDescription>{error}</AlertDescription>
      </Alert>
    );
  }

  if (!signal) return null;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">{symbol}</h1>
          <p className="text-sm text-muted-foreground">
            Signal deep dive · Last updated {formatDate(signal.timestamp)}
          </p>
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={() => window.location.reload()}
        >
          <RefreshCw className="mr-2 h-4 w-4" />
          Refresh
        </Button>
      </div>

      {/* Main recommendation */}
      <RecommendationCard signal={signal} size="lg" />

      {/* Tabs */}
      <Tabs defaultValue="history">
        <TabsList>
          <TabsTrigger value="history">History</TabsTrigger>
          <TabsTrigger value="models">Model Contributions</TabsTrigger>
          <TabsTrigger value="intelligence">Intelligence</TabsTrigger>
        </TabsList>

        {/* History tab */}
        <TabsContent value="history">
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Signal History</CardTitle>
            </CardHeader>
            <CardContent>
              <SignalHistory signals={history} showChart />
            </CardContent>
          </Card>
        </TabsContent>

        {/* Model contributions tab */}
        <TabsContent value="models">
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Model Contributions</CardTitle>
            </CardHeader>
            <CardContent>
              {isProTier && explain ? (
                <ModelContribChart
                  contributions={explain.model_contributions}
                  height={350}
                />
              ) : (
                <div className="flex flex-col items-center justify-center gap-3 py-12 text-center">
                  <Lock className="h-8 w-8 text-muted-foreground" />
                  <p className="text-sm text-muted-foreground">
                    Model explainability is a Premium feature.
                  </p>
                  <Button size="sm" asChild>
                    <a href="/pricing">Upgrade to Premium</a>
                  </Button>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Intelligence tab */}
        <TabsContent value="intelligence">
          <Card>
            <CardHeader>
              <CardTitle className="text-base">AI Market Intelligence</CardTitle>
            </CardHeader>
            <CardContent>
              {isProTier && intel ? (
                <div className="space-y-4">
                  <div>
                    <p className="mb-1 text-xs font-semibold uppercase text-muted-foreground">
                      AI Analysis
                    </p>
                    <p className="text-sm leading-relaxed">{intel.ai_analysis}</p>
                  </div>
                  {intel.catalysts.length > 0 && (
                    <div>
                      <p className="mb-1 text-xs font-semibold uppercase text-muted-foreground">
                        Key Catalysts
                      </p>
                      <ul className="list-inside list-disc space-y-0.5 text-sm">
                        {intel.catalysts.map((c, i) => (
                          <li key={i}>{c}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                  {intel.risk_factors.length > 0 && (
                    <div>
                      <p className="mb-1 text-xs font-semibold uppercase text-muted-foreground">
                        Risk Factors
                      </p>
                      <ul className="list-inside list-disc space-y-0.5 text-sm text-red-400">
                        {intel.risk_factors.map((r, i) => (
                          <li key={i}>{r}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center gap-3 py-12 text-center">
                  <Lock className="h-8 w-8 text-muted-foreground" />
                  <p className="text-sm text-muted-foreground">
                    AI intelligence reports are a Premium feature.
                  </p>
                  <Button size="sm" asChild>
                    <a href="/pricing">Upgrade to Premium</a>
                  </Button>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
