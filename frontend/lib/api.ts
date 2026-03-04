/**
 * QuantFlow typed API client.
 *
 * Uses the Next.js rewrite proxy so all calls go to /api/* which proxies
 * to the FastAPI backend (no CORS issues in dev).
 *
 * Auth token is read from localStorage (key: "qf_token").
 */

import type {
  Token,
  UserResponse,
  ApiKeyResponse,
  SignalResponse,
  SignalHistory,
  ExplainabilityReport,
  BatchSignalResponse,
  ScreenerEntry,
  OptimizedPortfolio,
  EfficientFrontierResponse,
  StressTestResponse,
  RiskReport,
  IntelligenceReport,
  NewsItem,
  MacroReport,
  SubscriptionStatus,
  CheckoutSession,
  WatchlistUpdate,
} from "./types";

// ---------------------------------------------------------------------------
// Token storage
// ---------------------------------------------------------------------------

const TOKEN_KEY = "qf_token";
const USER_KEY = "qf_user";

export function getStoredToken(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem(TOKEN_KEY);
}

export function setStoredToken(token: string): void {
  localStorage.setItem(TOKEN_KEY, token);
}

export function setStoredUser(user: UserResponse): void {
  localStorage.setItem(USER_KEY, JSON.stringify(user));
}

export function getStoredUser(): UserResponse | null {
  if (typeof window === "undefined") return null;
  const raw = localStorage.getItem(USER_KEY);
  if (!raw) return null;
  try {
    return JSON.parse(raw) as UserResponse;
  } catch {
    return null;
  }
}

export function clearAuth(): void {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(USER_KEY);
}

// ---------------------------------------------------------------------------
// Base fetch
// ---------------------------------------------------------------------------

const BASE_URL = "/api/v1";

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
    public detail?: unknown,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(
  method: string,
  path: string,
  body?: unknown,
  options: RequestInit = {},
): Promise<T> {
  const token = getStoredToken();
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...(options.headers as Record<string, string>),
  };
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  const res = await fetch(`${BASE_URL}${path}`, {
    method,
    headers,
    body: body !== undefined ? JSON.stringify(body) : undefined,
    ...options,
  });

  if (!res.ok) {
    let detail: unknown;
    try {
      detail = await res.json();
    } catch {
      detail = await res.text();
    }
    const message =
      typeof detail === "object" &&
      detail !== null &&
      "detail" in detail
        ? String((detail as { detail: unknown }).detail)
        : `HTTP ${res.status}`;
    throw new ApiError(res.status, message, detail);
  }

  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

const get = <T>(path: string) => request<T>("GET", path);
const post = <T>(path: string, body?: unknown) =>
  request<T>("POST", path, body);
const put = <T>(path: string, body?: unknown) =>
  request<T>("PUT", path, body);
const del = <T>(path: string) => request<T>("DELETE", path);

// ---------------------------------------------------------------------------
// Auth
// ---------------------------------------------------------------------------

export interface WatchlistUpdate {
  watchlist: string[];
}

export const auth = {
  register: (email: string, password: string) =>
    post<Token>("/auth/register", { email, password }),

  login: (email: string, password: string) =>
    post<Token>("/auth/login", { email, password }),

  me: () => get<UserResponse>("/auth/me"),

  updateWatchlist: (watchlist: string[]) =>
    put<UserResponse>("/auth/me/watchlist", { watchlist }),

  createApiKey: (name: string) =>
    post<ApiKeyResponse>("/auth/api-keys", { name }),

  listApiKeys: () => get<ApiKeyResponse[]>("/auth/api-keys"),

  deleteApiKey: (keyId: string) => del<void>(`/auth/api-keys/${keyId}`),
};

// ---------------------------------------------------------------------------
// Signals
// ---------------------------------------------------------------------------

export const signals = {
  get: (symbol: string) =>
    get<SignalResponse>(`/signals/${symbol.toUpperCase()}`),

  history: (symbol: string, days = 30) =>
    get<SignalHistory>(`/signals/${symbol.toUpperCase()}/history?days=${days}`),

  explain: (symbol: string) =>
    get<ExplainabilityReport>(`/signals/${symbol.toUpperCase()}/explain`),

  batch: (symbols: string[]) =>
    post<BatchSignalResponse>("/signals/batch", { symbols }),

  screener: (minConfidence = 0.6, limit = 50) =>
    get<ScreenerEntry[]>(
      `/signals/universe/screener?min_confidence=${minConfidence}&limit=${limit}`,
    ),
};

// ---------------------------------------------------------------------------
// Portfolio
// ---------------------------------------------------------------------------

export type OptimizationMethod = "mvo" | "hrp" | "black_litterman";
export type CovarianceMethod = "sample" | "ledoit_wolf" | "oas" | "rmt";

export interface OptimizeRequest {
  symbols: string[];
  method?: OptimizationMethod;
  covariance_method?: CovarianceMethod;
  risk_free_rate?: number;
  target_return?: number;
}

export interface StressTestRequest {
  symbols: string[];
  weights: Record<string, number>;
  scenarios?: string[];
}

export const portfolio = {
  optimize: (req: OptimizeRequest) =>
    post<OptimizedPortfolio>("/portfolio/optimize", req),

  efficientFrontier: (symbols: string[], nPoints = 30) =>
    post<EfficientFrontierResponse>(
      `/portfolio/efficient-frontier?n_points=${nPoints}`,
      { symbols },
    ),

  stressTest: (req: StressTestRequest) =>
    post<StressTestResponse>("/portfolio/stress-test", req),

  riskReport: (symbols: string[]) =>
    post<RiskReport>("/portfolio/risk-report", { symbols }),
};

// ---------------------------------------------------------------------------
// Intelligence
// ---------------------------------------------------------------------------

export const intelligence = {
  get: (symbol: string) =>
    get<IntelligenceReport>(`/intelligence/${symbol.toUpperCase()}`),

  news: (symbol: string, days = 7) =>
    get<NewsItem[]>(`/intelligence/${symbol.toUpperCase()}/news?days=${days}`),

  macroReport: () => get<MacroReport>("/intelligence/macro/report"),
};

// ---------------------------------------------------------------------------
// Subscription
// ---------------------------------------------------------------------------

export const subscription = {
  status: () => get<SubscriptionStatus>("/subscription/status"),

  createCheckout: (tier: "premium" | "institutional") =>
    post<CheckoutSession>("/subscription/create-checkout-session", { tier }),

  cancel: () => del<void>("/subscription/cancel"),
};
