/**
 * TypeScript types matching the QuantFlow FastAPI backend schemas.
 * Keep in sync with quantflow/api/auth/schemas.py and routers/*.py
 */

// ---------------------------------------------------------------------------
// Auth
// ---------------------------------------------------------------------------

export type SubscriptionTier = "free" | "premium" | "institutional";

export interface UserResponse {
  id: string;
  email: string;
  subscription_tier: SubscriptionTier;
  is_verified: boolean;
  watchlist: string[];
  created_at: string;
}

export interface Token {
  access_token: string;
  token_type: "bearer";
  expires_in: number;
  user: UserResponse;
}

export interface ApiKeyResponse {
  id: string;
  name: string;
  key?: string; // only present on creation
  created_at: string;
}

// ---------------------------------------------------------------------------
// Signals
// ---------------------------------------------------------------------------

export type Recommendation =
  | "STRONG_BUY"
  | "BUY"
  | "WEAK_BUY"
  | "HOLD"
  | "WEAK_SELL"
  | "SELL"
  | "STRONG_SELL";

export type MarketRegime =
  | "BULL_LOW_VOL"
  | "BULL_HIGH_VOL"
  | "BEAR_LOW_VOL"
  | "BEAR_HIGH_VOL"
  | "SIDEWAYS"
  | "CRISIS"
  | "RECOVERY";

export interface SignalResponse {
  symbol: string;
  recommendation: Recommendation;
  signal_strength: number;       // [-1, +1]
  confidence: number;            // [0, 1]
  suggested_position_size: number;
  regime: MarketRegime;
  timestamp: string;
  risk_warnings: string[];
}

export interface SignalHistory {
  signals: SignalResponse[];
  symbol: string;
  total: number;
}

export interface ExplainabilityReport {
  symbol: string;
  recommendation: Recommendation;
  model_contributions: Record<string, number>;
  feature_importances: Record<string, number>;
  narrative: string;
  risk_factors: string[];
  timestamp: string;
}

export interface BatchSignalResponse {
  results: SignalResponse[];
  errors: Record<string, string>;
}

export interface ScreenerEntry extends SignalResponse {
  sector?: string;
  market_cap?: number;
}

// ---------------------------------------------------------------------------
// Portfolio
// ---------------------------------------------------------------------------

export interface WeightEntry {
  symbol: string;
  weight: number;
  signal_strength: number;
  expected_return: number;
  volatility: number;
}

export interface OptimizedPortfolio {
  weights: WeightEntry[];
  expected_return: number;
  expected_volatility: number;
  sharpe_ratio: number;
  method: string;
  timestamp: string;
}

export interface FrontierPoint {
  expected_return: number;
  expected_volatility: number;
  sharpe_ratio: number;
  weights: Record<string, number>;
}

export interface EfficientFrontierResponse {
  frontier: FrontierPoint[];
  optimal: FrontierPoint;
}

export interface StressScenario {
  name: string;
  portfolio_return: number;
  max_drawdown: number;
  var_95: number;
}

export interface StressTestResponse {
  scenarios: StressScenario[];
  monte_carlo: {
    mean_return: number;
    std_return: number;
    var_95: number;
    es_95: number;
    worst_case: number;
    best_case: number;
  };
}

export interface RiskReport {
  var_95: number;
  var_99: number;
  es_95: number;
  es_99: number;
  max_drawdown: number;
  sharpe_ratio: number;
  symbols: string[];
  timestamp: string;
}

// ---------------------------------------------------------------------------
// Intelligence
// ---------------------------------------------------------------------------

export interface NewsItem {
  title: string;
  url: string;
  source: string;
  published_at: string;
  sentiment_score: number;
  summary: string;
}

export interface IntelligenceReport {
  symbol: string;
  overall_sentiment: number;   // [-1, +1]
  sentiment_label: string;
  confidence: number;
  news_items: NewsItem[];
  ai_analysis: string;
  risk_factors: string[];
  catalysts: string[];
  timestamp: string;
}

export interface MacroReport {
  market_regime: "RISK_ON" | "NEUTRAL" | "RISK_OFF";
  regime_confidence: number;
  avg_sentiment: number;
  components: Record<string, { sentiment: number; signal: number }>;
  narrative: string;
  timestamp: string;
}

// ---------------------------------------------------------------------------
// Subscription
// ---------------------------------------------------------------------------

export interface SubscriptionStatus {
  tier: SubscriptionTier;
  stripe_customer_id?: string;
  stripe_subscription_id?: string;
  cancel_at_period_end: boolean;
}

export interface CheckoutSession {
  url: string;
}

// ---------------------------------------------------------------------------
// WebSocket
// ---------------------------------------------------------------------------

export interface WsSignalMessage {
  type: "signal";
  symbol: string;
  data: SignalResponse;
}

export interface WsMarketMessage {
  type: "market";
  data: {
    regime: MarketRegime;
    sentiment: number;
    timestamp: string;
  };
}

export interface WsHeartbeat {
  type: "heartbeat";
  timestamp: string;
}

export type WsMessage = WsSignalMessage | WsMarketMessage | WsHeartbeat;

// ---------------------------------------------------------------------------
// UI helpers
// ---------------------------------------------------------------------------

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  page_size: number;
}
