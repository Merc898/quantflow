"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { Sidebar } from "@/components/layout/Sidebar";
import { TopBar } from "@/components/layout/TopBar";
import { useIsAuthenticated, useStore } from "@/lib/store";
import { createMarketSocket } from "@/lib/websocket";

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const isAuthenticated = useIsAuthenticated();
  const router = useRouter();
  const { setMarket } = useStore();

  // Redirect unauthenticated users to login
  useEffect(() => {
    if (!isAuthenticated) {
      router.push("/login");
    }
  }, [isAuthenticated, router]);

  // Subscribe to global market stream
  useEffect(() => {
    if (!isAuthenticated) return;
    const sock = createMarketSocket();
    const unsub = sock.subscribe((msg) => {
      if (msg.type === "market") {
        setMarket(msg.data.regime, msg.data.sentiment);
      }
    });
    return () => {
      unsub();
      sock.destroy();
    };
  }, [isAuthenticated, setMarket]);

  if (!isAuthenticated) return null;

  return (
    <div className="flex h-screen overflow-hidden bg-background">
      <Sidebar />
      <div className="flex flex-1 flex-col overflow-hidden">
        <TopBar />
        <main className="flex-1 overflow-y-auto p-6">{children}</main>
      </div>
    </div>
  );
}
