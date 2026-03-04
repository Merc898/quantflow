"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  BarChart3,
  Brain,
  ChevronLeft,
  ChevronRight,
  Layers,
  LayoutDashboard,
  LogOut,
  Search,
  Settings,
  Zap,
} from "lucide-react";
import { cn, tierLabel } from "@/lib/utils";
import { useStore, useUser } from "@/lib/store";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";

const navItems = [
  { href: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
  { href: "/screener", label: "Screener", icon: Search },
  { href: "/portfolio", label: "Portfolio", icon: Layers },
  { href: "/intelligence", label: "Intelligence", icon: Brain },
  { href: "/settings", label: "Settings", icon: Settings },
];

export function Sidebar() {
  const pathname = usePathname();
  const user = useUser();
  const { sidebarCollapsed, toggleSidebar, logout } = useStore();

  return (
    <aside
      className={cn(
        "flex h-screen flex-col border-r bg-card transition-all duration-300",
        sidebarCollapsed ? "w-16" : "w-60",
      )}
    >
      {/* Logo */}
      <div className="flex h-16 items-center justify-between px-4">
        {!sidebarCollapsed && (
          <Link href="/dashboard" className="flex items-center gap-2">
            <Zap className="h-6 w-6 text-primary" />
            <span className="text-lg font-bold">QuantFlow</span>
          </Link>
        )}
        {sidebarCollapsed && (
          <Link href="/dashboard">
            <Zap className="h-6 w-6 text-primary" />
          </Link>
        )}
        <Button
          variant="ghost"
          size="icon"
          onClick={toggleSidebar}
          className="ml-auto"
          aria-label="Toggle sidebar"
        >
          {sidebarCollapsed ? (
            <ChevronRight className="h-4 w-4" />
          ) : (
            <ChevronLeft className="h-4 w-4" />
          )}
        </Button>
      </div>

      <Separator />

      {/* Nav */}
      <nav className="flex-1 space-y-1 p-3">
        {navItems.map(({ href, label, icon: Icon }) => {
          const isActive = pathname === href || pathname.startsWith(`${href}/`);
          return (
            <Link
              key={href}
              href={href}
              className={cn(
                "flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors",
                isActive
                  ? "bg-primary text-primary-foreground"
                  : "text-muted-foreground hover:bg-accent hover:text-accent-foreground",
                sidebarCollapsed && "justify-center px-2",
              )}
              title={sidebarCollapsed ? label : undefined}
            >
              <Icon className="h-4 w-4 shrink-0" />
              {!sidebarCollapsed && <span>{label}</span>}
            </Link>
          );
        })}
      </nav>

      <Separator />

      {/* User */}
      <div className={cn("p-3", sidebarCollapsed && "flex justify-center")}>
        {!sidebarCollapsed && user && (
          <div className="mb-2 rounded-md bg-muted px-3 py-2">
            <p className="truncate text-xs font-medium">{user.email}</p>
            <Badge
              variant={
                user.subscription_tier === "free" ? "secondary" : "default"
              }
              className="mt-1 text-xs"
            >
              {tierLabel(user.subscription_tier)}
            </Badge>
          </div>
        )}
        <Button
          variant="ghost"
          size={sidebarCollapsed ? "icon" : "sm"}
          className={cn(
            "text-muted-foreground hover:text-destructive",
            !sidebarCollapsed && "w-full justify-start gap-2",
          )}
          onClick={logout}
          title="Log out"
        >
          <LogOut className="h-4 w-4" />
          {!sidebarCollapsed && "Log out"}
        </Button>
      </div>
    </aside>
  );
}
