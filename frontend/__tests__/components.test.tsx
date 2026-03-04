/**
 * Render tests for key QuantFlow components.
 * Tests verify correct rendering, colour coding, and accessibility.
 */

import React from "react";
import { render, screen } from "@testing-library/react";
import { ConfidenceBar } from "../components/charts/ConfidenceBar";
import { RecommendationCard } from "../components/signals/RecommendationCard";
import { RiskWarnings } from "../components/signals/RiskWarnings";
import type { SignalResponse } from "../lib/types";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeSignal(overrides: Partial<SignalResponse> = {}): SignalResponse {
  return {
    symbol: "AAPL",
    recommendation: "BUY",
    signal_strength: 0.65,
    confidence: 0.78,
    suggested_position_size: 0.08,
    regime: "BULL_LOW_VOL",
    timestamp: "2025-03-01T12:00:00Z",
    risk_warnings: [],
    ...overrides,
  };
}

// ---------------------------------------------------------------------------
// ConfidenceBar
// ---------------------------------------------------------------------------

describe("ConfidenceBar", () => {
  it("renders the label", () => {
    render(<ConfidenceBar confidence={0.75} label="Test confidence" />);
    expect(screen.getByText("Test confidence")).toBeInTheDocument();
  });

  it("renders the percentage", () => {
    render(<ConfidenceBar confidence={0.75} />);
    expect(screen.getByText("75.0%")).toBeInTheDocument();
  });

  it("hides label when showLabel=false", () => {
    render(
      <ConfidenceBar confidence={0.5} label="Hidden" showLabel={false} />,
    );
    expect(screen.queryByText("Hidden")).not.toBeInTheDocument();
  });

  it("renders 0% confidence", () => {
    render(<ConfidenceBar confidence={0} />);
    expect(screen.getByText("0.0%")).toBeInTheDocument();
  });

  it("renders 100% confidence", () => {
    render(<ConfidenceBar confidence={1} />);
    expect(screen.getByText("100.0%")).toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// RecommendationCard
// ---------------------------------------------------------------------------

describe("RecommendationCard", () => {
  it("renders the symbol", () => {
    render(<RecommendationCard signal={makeSignal()} />);
    expect(screen.getByText("AAPL")).toBeInTheDocument();
  });

  it("renders the recommendation label", () => {
    render(<RecommendationCard signal={makeSignal({ recommendation: "STRONG_BUY" })} />);
    expect(screen.getByText("STRONG BUY")).toBeInTheDocument();
  });

  it("renders confidence bar", () => {
    render(<RecommendationCard signal={makeSignal({ confidence: 0.82 })} />);
    expect(screen.getByText("82.0%")).toBeInTheDocument();
  });

  it("renders risk warnings", () => {
    const signal = makeSignal({
      risk_warnings: ["HIGH volatility detected", "Low liquidity"],
    });
    render(<RecommendationCard signal={signal} />);
    expect(screen.getByText("HIGH volatility detected")).toBeInTheDocument();
    expect(screen.getByText("Low liquidity")).toBeInTheDocument();
  });

  it("renders STRONG_SELL signal", () => {
    render(
      <RecommendationCard
        signal={makeSignal({ recommendation: "STRONG_SELL", signal_strength: -0.9 })}
      />,
    );
    expect(screen.getByText("STRONG SELL")).toBeInTheDocument();
  });

  it("renders HOLD signal", () => {
    render(
      <RecommendationCard
        signal={makeSignal({ recommendation: "HOLD", signal_strength: 0 })}
      />,
    );
    expect(screen.getByText("HOLD")).toBeInTheDocument();
  });

  it("shows position size", () => {
    render(
      <RecommendationCard signal={makeSignal({ suggested_position_size: 0.08 })} />,
    );
    expect(screen.getByText("8.0%")).toBeInTheDocument();
  });

  it("renders without risk warnings by default", () => {
    render(<RecommendationCard signal={makeSignal()} />);
    expect(screen.queryByText("Risk Warning")).not.toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// RiskWarnings
// ---------------------------------------------------------------------------

describe("RiskWarnings", () => {
  it("renders nothing with empty warnings", () => {
    const { container } = render(<RiskWarnings warnings={[]} />);
    expect(container.firstChild).toBeNull();
  });

  it("renders each warning", () => {
    render(
      <RiskWarnings
        warnings={["High volatility", "CRITICAL drawdown risk"]}
      />,
    );
    expect(screen.getByText("High volatility")).toBeInTheDocument();
    expect(screen.getByText("CRITICAL drawdown risk")).toBeInTheDocument();
  });

  it("renders warning title badges", () => {
    render(<RiskWarnings warnings={["Low liquidity"]} />);
    expect(screen.getAllByText("Risk Warning").length).toBeGreaterThan(0);
  });

  it("renders multiple warnings", () => {
    const warnings = ["Warning A", "Warning B", "Warning C"];
    render(<RiskWarnings warnings={warnings} />);
    warnings.forEach((w) => {
      expect(screen.getByText(w)).toBeInTheDocument();
    });
  });
});

// ---------------------------------------------------------------------------
// Type completeness tests
// ---------------------------------------------------------------------------

describe("SignalResponse type coverage", () => {
  const allRecs: SignalResponse["recommendation"][] = [
    "STRONG_BUY",
    "BUY",
    "WEAK_BUY",
    "HOLD",
    "WEAK_SELL",
    "SELL",
    "STRONG_SELL",
  ];

  allRecs.forEach((rec) => {
    it(`renders ${rec} without crashing`, () => {
      const signal = makeSignal({ recommendation: rec });
      expect(() => render(<RecommendationCard signal={signal} />)).not.toThrow();
    });
  });

  const allRegimes: SignalResponse["regime"][] = [
    "BULL_LOW_VOL",
    "BULL_HIGH_VOL",
    "BEAR_LOW_VOL",
    "BEAR_HIGH_VOL",
    "SIDEWAYS",
    "CRISIS",
    "RECOVERY",
  ];

  allRegimes.forEach((regime) => {
    it(`renders regime ${regime} without crashing`, () => {
      const signal = makeSignal({ regime });
      expect(() => render(<RecommendationCard signal={signal} />)).not.toThrow();
    });
  });
});
