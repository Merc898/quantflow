"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Brain, Globe, Loader2, ExternalLink, Lock } from "lucide-react";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { intelligence as intelligenceApi, ApiError } from "@/lib/api";
import { useIsProTier } from "@/lib/store";
import { formatDate, formatPercent, cn } from "@/lib/utils";
import type { MacroReport, IntelligenceReport, NewsItem } from "@/lib/types";

export default function IntelligencePage() {
  const isProTier = useIsProTier();
  const [macro, setMacro] = useState<MacroReport | null>(null);
  const [symbolInput, setSymbolInput] = useState("AAPL");
  const [intel, setIntel] = useState<IntelligenceReport | null>(null);
  const [news, setNews] = useState<NewsItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [macroLoading, setMacroLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!isProTier) return;
    setMacroLoading(true);
    intelligenceApi
      .macroReport()
      .then(setMacro)
      .catch(() => null)
      .finally(() => setMacroLoading(false));
  }, [isProTier]);

  const loadSymbol = async (sym: string) => {
    setLoading(true);
    setError(null);
    try {
      const [report, newsItems] = await Promise.all([
        intelligenceApi.get(sym).catch(() => null),
        intelligenceApi.news(sym, 30),
      ]);
      setIntel(report);
      setNews(newsItems);
    } catch (err) {
      if (err instanceof ApiError && err.status === 404) {
        setError(`No intelligence data for ${sym}. The agent pipeline may not have run yet.`);
      } else {
        setError("Failed to load intelligence report.");
      }
    } finally {
      setLoading(false);
    }
  };

  if (!isProTier) {
    return (
      <div className="flex flex-col items-center justify-center gap-4 py-24 text-center">
        <Brain className="h-10 w-10 text-muted-foreground" />
        <h2 className="text-xl font-semibold">AI Market Intelligence</h2>
        <p className="max-w-sm text-sm text-muted-foreground">
          Agentic AI analysis combining GPT-4o, Claude, Perplexity, and
          real-time web scraping. Available on Premium and Institutional plans.
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
        <h1 className="text-2xl font-bold">AI Market Intelligence</h1>
        <p className="text-sm text-muted-foreground">
          Powered by GPT-4o · Claude · Perplexity · Real-time web intelligence
        </p>
      </div>

      <Tabs defaultValue="symbol">
        <TabsList>
          <TabsTrigger value="symbol">Symbol Analysis</TabsTrigger>
          <TabsTrigger value="macro">Macro Report</TabsTrigger>
        </TabsList>

        {/* Symbol analysis */}
        <TabsContent value="symbol" className="space-y-4">
          <div className="flex gap-2">
            <Input
              placeholder="Enter symbol (e.g. AAPL)"
              value={symbolInput}
              onChange={(e) => setSymbolInput(e.target.value.toUpperCase())}
              onKeyDown={(e) => e.key === "Enter" && loadSymbol(symbolInput)}
              className="max-w-xs"
            />
            <Button
              onClick={() => loadSymbol(symbolInput)}
              disabled={loading || !symbolInput}
            >
              {loading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
              Analyse
            </Button>
          </div>

          {error && (
            <Alert variant="destructive">
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          {intel && (
            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle className="text-base">{intel.symbol} Intelligence Report</CardTitle>
                  <Badge
                    variant={
                      intel.sentiment_label === "BULLISH"
                        ? "bull"
                        : intel.sentiment_label === "BEARISH"
                          ? "bear"
                          : "neutral"
                    }
                  >
                    {intel.sentiment_label}
                  </Badge>
                </div>
                <CardDescription>
                  Sentiment: {(intel.overall_sentiment * 100).toFixed(0)} ·
                  Confidence: {formatPercent(intel.confidence)} ·
                  Updated {formatDate(intel.timestamp)}
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <p className="mb-1.5 text-xs font-semibold uppercase text-muted-foreground">
                    AI Analysis
                  </p>
                  <p className="text-sm leading-relaxed">{intel.ai_analysis}</p>
                </div>
                {intel.catalysts.length > 0 && (
                  <div>
                    <p className="mb-1.5 text-xs font-semibold uppercase text-muted-foreground">
                      Bullish Catalysts
                    </p>
                    <ul className="space-y-0.5 text-sm">
                      {intel.catalysts.map((c, i) => (
                        <li key={i} className="flex items-start gap-1.5">
                          <span className="mt-1 h-1.5 w-1.5 shrink-0 rounded-full bg-green-400" />
                          {c}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
                {intel.risk_factors.length > 0 && (
                  <div>
                    <p className="mb-1.5 text-xs font-semibold uppercase text-muted-foreground">
                      Risk Factors
                    </p>
                    <ul className="space-y-0.5 text-sm">
                      {intel.risk_factors.map((r, i) => (
                        <li key={i} className="flex items-start gap-1.5">
                          <span className="mt-1 h-1.5 w-1.5 shrink-0 rounded-full bg-red-400" />
                          {r}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </CardContent>
            </Card>
          )}

          {/* News feed */}
          {news.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Recent News</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                {news.map((item, i) => (
                  <div
                    key={i}
                    className="flex items-start justify-between gap-3 border-b border-border/50 pb-3 last:border-0 last:pb-0"
                  >
                    <div className="min-w-0 flex-1">
                      <a
                        href={item.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="flex items-center gap-1 text-sm font-medium hover:text-primary"
                      >
                        {item.title}
                        <ExternalLink className="h-3 w-3 shrink-0" />
                      </a>
                      <div className="mt-0.5 flex items-center gap-2 text-xs text-muted-foreground">
                        <span>{item.source}</span>
                        <span>·</span>
                        <span>{formatDate(item.published_at)}</span>
                      </div>
                    </div>
                    <Badge
                      variant={
                        item.sentiment_score > 0.1
                          ? "bull"
                          : item.sentiment_score < -0.1
                            ? "bear"
                            : "neutral"
                      }
                      className="shrink-0 text-xs"
                    >
                      {item.sentiment_score >= 0 ? "+" : ""}
                      {(item.sentiment_score * 100).toFixed(0)}
                    </Badge>
                  </div>
                ))}
              </CardContent>
            </Card>
          )}
        </TabsContent>

        {/* Macro report */}
        <TabsContent value="macro" className="space-y-4">
          {macroLoading && (
            <Skeleton className="h-48 w-full rounded-lg" />
          )}
          {macro && (
            <Card>
              <CardHeader>
                <div className="flex items-center gap-3">
                  <Globe className="h-5 w-5 text-primary" />
                  <div>
                    <CardTitle className="text-base">Global Macro Report</CardTitle>
                    <CardDescription>
                      Updated {formatDate(macro.timestamp)}
                    </CardDescription>
                  </div>
                  <Badge
                    variant={
                      macro.market_regime === "RISK_ON"
                        ? "bull"
                        : macro.market_regime === "RISK_OFF"
                          ? "bear"
                          : "neutral"
                    }
                    className="ml-auto"
                  >
                    {macro.market_regime.replace("_", " ")}
                  </Badge>
                </div>
              </CardHeader>
              <CardContent className="space-y-4">
                <p className="text-sm leading-relaxed">{macro.narrative}</p>
                <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                  {Object.entries(macro.components).map(([sym, comp]) => (
                    <div
                      key={sym}
                      className="rounded-lg bg-muted/50 p-3 text-center"
                    >
                      <p className="text-sm font-bold">{sym}</p>
                      <p
                        className={cn(
                          "text-lg font-bold",
                          comp.signal > 0 ? "text-green-400" : "text-red-400",
                        )}
                      >
                        {comp.signal >= 0 ? "+" : ""}
                        {(comp.signal * 100).toFixed(0)}
                      </p>
                      <p className="text-xs text-muted-foreground">
                        Sentiment: {(comp.sentiment * 100).toFixed(0)}
                      </p>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
}
