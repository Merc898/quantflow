"use client";

import { useState, useEffect } from "react";
import { Plus, Trash2, Loader2, KeyRound, Copy, Check } from "lucide-react";
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
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Separator } from "@/components/ui/separator";
import { auth as authApi, ApiError } from "@/lib/api";
import { useUser, useStore } from "@/lib/store";
import { tierLabel } from "@/lib/utils";
import type { ApiKeyResponse } from "@/lib/types";

export default function SettingsPage() {
  const user = useUser();
  const { updateUser } = useStore();
  const isProTier =
    user?.subscription_tier === "premium" ||
    user?.subscription_tier === "institutional";

  // Watchlist
  const [watchlist, setWatchlist] = useState<string[]>(user?.watchlist ?? []);
  const [newSymbol, setNewSymbol] = useState("");
  const [watchlistSaving, setWatchlistSaving] = useState(false);
  const [watchlistError, setWatchlistError] = useState<string | null>(null);
  const [watchlistSaved, setWatchlistSaved] = useState(false);

  // API keys
  const [apiKeys, setApiKeys] = useState<ApiKeyResponse[]>([]);
  const [newKeyName, setNewKeyName] = useState("");
  const [newKeyValue, setNewKeyValue] = useState<string | null>(null);
  const [keyLoading, setKeyLoading] = useState(false);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (isProTier) {
      authApi.listApiKeys().then(setApiKeys).catch(() => null);
    }
  }, [isProTier]);

  // Watchlist management
  const addSymbol = () => {
    const sym = newSymbol.trim().toUpperCase();
    const limit = isProTier ? 100 : 5;
    if (sym && !watchlist.includes(sym) && watchlist.length < limit) {
      setWatchlist((prev) => [...prev, sym]);
    }
    setNewSymbol("");
  };

  const removeSymbol = (sym: string) => {
    setWatchlist((prev) => prev.filter((s) => s !== sym));
  };

  const saveWatchlist = async () => {
    setWatchlistSaving(true);
    setWatchlistError(null);
    try {
      const updated = await authApi.updateWatchlist(watchlist);
      updateUser({ watchlist: updated.watchlist });
      setWatchlistSaved(true);
      setTimeout(() => setWatchlistSaved(false), 2000);
    } catch (err) {
      if (err instanceof ApiError) {
        setWatchlistError(err.message);
      }
    } finally {
      setWatchlistSaving(false);
    }
  };

  // API key management
  const createApiKey = async () => {
    if (!newKeyName.trim()) return;
    setKeyLoading(true);
    try {
      const key = await authApi.createApiKey(newKeyName.trim());
      setApiKeys((prev) => [key, ...prev]);
      setNewKeyValue(key.key ?? null);
      setNewKeyName("");
    } catch {
      // Silent
    } finally {
      setKeyLoading(false);
    }
  };

  const revokeApiKey = async (keyId: string) => {
    try {
      await authApi.deleteApiKey(keyId);
      setApiKeys((prev) => prev.filter((k) => k.id !== keyId));
    } catch {
      // Silent
    }
  };

  const copyKey = () => {
    if (newKeyValue) {
      navigator.clipboard.writeText(newKeyValue);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  return (
    <div className="mx-auto max-w-2xl space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Settings</h1>
        <p className="text-sm text-muted-foreground">
          Manage your account, watchlist, and API access
        </p>
      </div>

      {/* Account info */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Account</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium">{user?.email}</p>
              <p className="text-xs text-muted-foreground">Email address</p>
            </div>
          </div>
          <Separator />
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium">
                {user ? tierLabel(user.subscription_tier) : "—"} Plan
              </p>
              <p className="text-xs text-muted-foreground">Current subscription</p>
            </div>
            {!isProTier && (
              <Button size="sm" asChild>
                <a href="/pricing">Upgrade</a>
              </Button>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Watchlist */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Watchlist</CardTitle>
          <CardDescription>
            {isProTier
              ? "Unlimited symbols on Premium"
              : `Up to 5 symbols on the Free plan (${watchlist.length}/5 used)`}
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-wrap gap-1.5">
            {watchlist.map((sym) => (
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
            {watchlist.length === 0 && (
              <p className="text-sm text-muted-foreground">
                No symbols in watchlist
              </p>
            )}
          </div>

          <div className="flex gap-2">
            <Input
              placeholder="Add symbol (e.g. TSLA)"
              value={newSymbol}
              onChange={(e) => setNewSymbol(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && addSymbol()}
              disabled={!isProTier && watchlist.length >= 5}
            />
            <Button
              variant="outline"
              size="icon"
              onClick={addSymbol}
              disabled={!isProTier && watchlist.length >= 5}
            >
              <Plus className="h-4 w-4" />
            </Button>
          </div>

          {watchlistError && (
            <Alert variant="destructive">
              <AlertDescription className="text-xs">
                {watchlistError}
              </AlertDescription>
            </Alert>
          )}

          <Button
            onClick={saveWatchlist}
            disabled={watchlistSaving}
            size="sm"
          >
            {watchlistSaving ? (
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
            ) : watchlistSaved ? (
              <Check className="mr-2 h-4 w-4 text-green-400" />
            ) : null}
            {watchlistSaved ? "Saved!" : "Save watchlist"}
          </Button>
        </CardContent>
      </Card>

      {/* API Keys — Premium only */}
      {isProTier && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <KeyRound className="h-4 w-4" />
              API Keys
            </CardTitle>
            <CardDescription>
              Use API keys to access QuantFlow programmatically.
              Keys are shown once — store them safely.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* New key shown once */}
            {newKeyValue && (
              <Alert variant="success">
                <AlertDescription className="space-y-2">
                  <p className="text-xs font-semibold">
                    Copy your API key now — it won't be shown again.
                  </p>
                  <div className="flex items-center gap-2">
                    <code className="flex-1 break-all rounded bg-muted px-2 py-1 text-xs">
                      {newKeyValue}
                    </code>
                    <Button size="icon" variant="ghost" onClick={copyKey}>
                      {copied ? (
                        <Check className="h-4 w-4 text-green-400" />
                      ) : (
                        <Copy className="h-4 w-4" />
                      )}
                    </Button>
                  </div>
                </AlertDescription>
              </Alert>
            )}

            {/* Create key */}
            <div className="flex gap-2">
              <Input
                placeholder="Key name (e.g. Production)"
                value={newKeyName}
                onChange={(e) => setNewKeyName(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && createApiKey()}
              />
              <Button
                size="sm"
                onClick={createApiKey}
                disabled={keyLoading || !newKeyName.trim()}
              >
                {keyLoading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                Create
              </Button>
            </div>

            {/* Existing keys */}
            {apiKeys.length > 0 && (
              <div className="space-y-2">
                {apiKeys.map((key) => (
                  <div
                    key={key.id}
                    className="flex items-center justify-between rounded-md border px-3 py-2"
                  >
                    <div>
                      <p className="text-sm font-medium">{key.name}</p>
                      <p className="text-xs text-muted-foreground">
                        Created {new Date(key.created_at).toLocaleDateString()}
                      </p>
                    </div>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="text-destructive hover:text-destructive"
                      onClick={() => revokeApiKey(key.id)}
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
}
