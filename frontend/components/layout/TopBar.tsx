"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { Search, Bell, Zap } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { useUser, useRegime } from "@/lib/store";
import { regimeLabel, regimeColor } from "@/lib/utils";

export function TopBar() {
  const user = useUser();
  const regime = useRegime();
  const router = useRouter();
  const [searchQuery, setSearchQuery] = useState("");

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    const symbol = searchQuery.trim().toUpperCase();
    if (symbol) {
      router.push(`/signals/${symbol}`);
      setSearchQuery("");
    }
  };

  return (
    <header className="flex h-16 items-center justify-between border-b bg-card px-6">
      {/* Symbol search */}
      <form onSubmit={handleSearch} className="flex items-center gap-2">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            type="text"
            placeholder="Search symbol… (e.g. AAPL)"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-64 pl-9"
          />
        </div>
        <Button type="submit" size="sm" variant="outline">
          Analyse
        </Button>
      </form>

      {/* Centre: market regime */}
      {regime && (
        <div className="flex items-center gap-2">
          <span className="text-xs text-muted-foreground">Market regime:</span>
          <Badge variant="outline" className={regimeColor(regime)}>
            {regimeLabel(regime)}
          </Badge>
        </div>
      )}

      {/* Right: upgrade + notifications */}
      <div className="flex items-center gap-3">
        {user?.subscription_tier === "free" && (
          <Link href="/pricing">
            <Button size="sm" variant="default" className="gap-1">
              <Zap className="h-3 w-3" />
              Upgrade
            </Button>
          </Link>
        )}
        <Button variant="ghost" size="icon" aria-label="Notifications">
          <Bell className="h-4 w-4" />
        </Button>
      </div>
    </header>
  );
}
