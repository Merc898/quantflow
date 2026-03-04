/**
 * Zustand global state store for QuantFlow.
 */

import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { UserResponse, SignalResponse, MarketRegime } from "./types";
import { setStoredToken, setStoredUser, clearAuth } from "./api";

// ---------------------------------------------------------------------------
// Auth slice
// ---------------------------------------------------------------------------

interface AuthState {
  user: UserResponse | null;
  token: string | null;
  setAuth: (token: string, user: UserResponse) => void;
  logout: () => void;
  updateUser: (user: Partial<UserResponse>) => void;
}

// ---------------------------------------------------------------------------
// Signals slice
// ---------------------------------------------------------------------------

interface SignalsState {
  liveSignals: Record<string, SignalResponse>; // symbol → latest signal
  upsertSignal: (signal: SignalResponse) => void;
}

// ---------------------------------------------------------------------------
// Market slice
// ---------------------------------------------------------------------------

interface MarketState {
  regime: MarketRegime | null;
  macroSentiment: number | null;
  setMarket: (regime: MarketRegime, sentiment: number) => void;
}

// ---------------------------------------------------------------------------
// UI slice
// ---------------------------------------------------------------------------

interface UiState {
  sidebarCollapsed: boolean;
  toggleSidebar: () => void;
  toasts: Toast[];
  addToast: (toast: Omit<Toast, "id">) => void;
  removeToast: (id: string) => void;
}

export interface Toast {
  id: string;
  title: string;
  description?: string;
  variant?: "default" | "destructive" | "success";
}

// ---------------------------------------------------------------------------
// Combined store
// ---------------------------------------------------------------------------

type StoreState = AuthState & SignalsState & MarketState & UiState;

let toastCounter = 0;

export const useStore = create<StoreState>()(
  persist(
    (set, get) => ({
      // Auth
      user: null,
      token: null,
      setAuth: (token: string, user: UserResponse) => {
        setStoredToken(token);
        setStoredUser(user);
        set({ user, token });
      },
      logout: () => {
        clearAuth();
        set({ user: null, token: null });
      },
      updateUser: (partial: Partial<UserResponse>) => {
        const current = get().user;
        if (current) {
          const updated = { ...current, ...partial };
          setStoredUser(updated);
          set({ user: updated });
        }
      },

      // Signals
      liveSignals: {},
      upsertSignal: (signal: SignalResponse) => {
        set((state) => ({
          liveSignals: { ...state.liveSignals, [signal.symbol]: signal },
        }));
      },

      // Market
      regime: null,
      macroSentiment: null,
      setMarket: (regime: MarketRegime, sentiment: number) => {
        set({ regime, macroSentiment: sentiment });
      },

      // UI
      sidebarCollapsed: false,
      toggleSidebar: () => {
        set((state) => ({ sidebarCollapsed: !state.sidebarCollapsed }));
      },
      toasts: [],
      addToast: (toast: Omit<Toast, "id">) => {
        const id = `toast-${++toastCounter}`;
        set((state) => ({ toasts: [...state.toasts, { ...toast, id }] }));
        // Auto-remove after 5 seconds
        setTimeout(() => {
          set((state) => ({
            toasts: state.toasts.filter((t) => t.id !== id),
          }));
        }, 5000);
      },
      removeToast: (id: string) => {
        set((state) => ({
          toasts: state.toasts.filter((t) => t.id !== id),
        }));
      },
    }),
    {
      name: "quantflow-store",
      partialize: (state) => ({
        user: state.user,
        token: state.token,
        sidebarCollapsed: state.sidebarCollapsed,
      }),
    },
  ),
);

// ---------------------------------------------------------------------------
// Convenience selectors
// ---------------------------------------------------------------------------

export const useUser = () => useStore((s) => s.user);
export const useToken = () => useStore((s) => s.token);
export const useIsAuthenticated = () => useStore((s) => s.user !== null);
export const useIsProTier = () =>
  useStore(
    (s) =>
      s.user?.subscription_tier === "premium" ||
      s.user?.subscription_tier === "institutional",
  );
export const useLiveSignal = (symbol: string) =>
  useStore((s) => s.liveSignals[symbol] ?? null);
export const useRegime = () => useStore((s) => s.regime);
