/**
 * Unit tests for frontend utility functions.
 */

import {
  formatPercent,
  formatSignedPercent,
  formatCompact,
  recommendationLabel,
  recommendationColor,
  recommendationBgColor,
  signalColor,
  regimeLabel,
  regimeColor,
  tierLabel,
  clamp,
  signalToGauge,
} from "../lib/utils";
import type { Recommendation, MarketRegime, SubscriptionTier } from "../lib/types";

describe("formatPercent", () => {
  it("formats zero", () => {
    expect(formatPercent(0)).toBe("0.0%");
  });

  it("formats 1.0 as 100%", () => {
    expect(formatPercent(1)).toBe("100.0%");
  });

  it("formats decimals correctly", () => {
    expect(formatPercent(0.1234, 2)).toBe("12.34%");
  });

  it("formats negative", () => {
    expect(formatPercent(-0.05)).toBe("-5.0%");
  });
});

describe("formatSignedPercent", () => {
  it("adds plus for positive", () => {
    expect(formatSignedPercent(0.1)).toBe("+10.0%");
  });

  it("does not add plus for negative", () => {
    expect(formatSignedPercent(-0.1)).toBe("-10.0%");
  });

  it("handles zero", () => {
    expect(formatSignedPercent(0)).toBe("+0.0%");
  });
});

describe("formatCompact", () => {
  it("formats billions", () => {
    expect(formatCompact(2.5e9)).toBe("2.5B");
  });

  it("formats millions", () => {
    expect(formatCompact(3.2e6)).toBe("3.2M");
  });

  it("formats thousands", () => {
    expect(formatCompact(1500)).toBe("1.5K");
  });

  it("formats small numbers as decimals", () => {
    expect(formatCompact(42.5)).toBe("42.50");
  });

  it("handles negative", () => {
    expect(formatCompact(-1e9)).toBe("-1.0B");
  });
});

describe("recommendationLabel", () => {
  it("converts underscores to spaces", () => {
    expect(recommendationLabel("STRONG_BUY")).toBe("STRONG BUY");
  });

  it("handles HOLD", () => {
    expect(recommendationLabel("HOLD")).toBe("HOLD");
  });

  it("handles WEAK_SELL", () => {
    expect(recommendationLabel("WEAK_SELL")).toBe("WEAK SELL");
  });
});

describe("recommendationColor", () => {
  const bullish: Recommendation[] = ["STRONG_BUY", "BUY", "WEAK_BUY"];
  const bearish: Recommendation[] = ["STRONG_SELL", "SELL", "WEAK_SELL"];

  bullish.forEach((rec) => {
    it(`${rec} is green`, () => {
      expect(recommendationColor(rec)).toContain("green");
    });
  });

  bearish.forEach((rec) => {
    it(`${rec} is red`, () => {
      expect(recommendationColor(rec)).toContain("red");
    });
  });

  it("HOLD is amber", () => {
    expect(recommendationColor("HOLD")).toContain("amber");
  });
});

describe("signalColor", () => {
  it("returns green for strong bull", () => {
    expect(signalColor(0.9)).toBe("#22c55e");
  });

  it("returns red for strong bear", () => {
    expect(signalColor(-0.9)).toBe("#ef4444");
  });

  it("returns amber for neutral", () => {
    expect(signalColor(0)).toBe("#f59e0b");
  });
});

describe("regimeLabel", () => {
  it("formats BULL_LOW_VOL", () => {
    expect(regimeLabel("BULL_LOW_VOL")).toBe("Bull · Low Vol");
  });

  it("formats CRISIS", () => {
    expect(regimeLabel("CRISIS")).toBe("Crisis");
  });
});

describe("regimeColor", () => {
  it("BULL regimes are green", () => {
    expect(regimeColor("BULL_LOW_VOL")).toContain("green");
    expect(regimeColor("BULL_HIGH_VOL")).toContain("green");
  });

  it("CRISIS is red-500", () => {
    expect(regimeColor("CRISIS")).toContain("red");
  });

  it("BEAR regimes are red", () => {
    expect(regimeColor("BEAR_LOW_VOL")).toContain("red");
  });
});

describe("tierLabel", () => {
  it("maps free", () => {
    expect(tierLabel("free")).toBe("Free");
  });

  it("maps premium", () => {
    expect(tierLabel("premium")).toBe("Premium");
  });

  it("maps institutional", () => {
    expect(tierLabel("institutional")).toBe("Institutional");
  });
});

describe("clamp", () => {
  it("clamps below min", () => {
    expect(clamp(-10, 0, 100)).toBe(0);
  });

  it("clamps above max", () => {
    expect(clamp(200, 0, 100)).toBe(100);
  });

  it("passes through in range", () => {
    expect(clamp(50, 0, 100)).toBe(50);
  });
});

describe("signalToGauge", () => {
  it("maps -1 to 0", () => {
    expect(signalToGauge(-1)).toBe(0);
  });

  it("maps 0 to 50", () => {
    expect(signalToGauge(0)).toBe(50);
  });

  it("maps 1 to 100", () => {
    expect(signalToGauge(1)).toBe(100);
  });

  it("maps 0.5 to 75", () => {
    expect(signalToGauge(0.5)).toBe(75);
  });

  it("clamps out of range", () => {
    expect(signalToGauge(2)).toBe(100);
    expect(signalToGauge(-2)).toBe(0);
  });
});
