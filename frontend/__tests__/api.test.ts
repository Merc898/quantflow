/**
 * Unit tests for the API client (auth, tokens, error handling).
 */

import {
  getStoredToken,
  setStoredToken,
  setStoredUser,
  getStoredUser,
  clearAuth,
  ApiError,
} from "../lib/api";
import type { UserResponse } from "../lib/types";

// ---------------------------------------------------------------------------
// localStorage mock
// ---------------------------------------------------------------------------

const localStorageMock = (() => {
  let store: Record<string, string> = {};
  return {
    getItem: (key: string) => store[key] ?? null,
    setItem: (key: string, value: string) => { store[key] = value; },
    removeItem: (key: string) => { delete store[key]; },
    clear: () => { store = {}; },
  };
})();

Object.defineProperty(window, "localStorage", { value: localStorageMock });

// ---------------------------------------------------------------------------
// Token storage tests
// ---------------------------------------------------------------------------

describe("Token storage", () => {
  beforeEach(() => localStorageMock.clear());

  it("getStoredToken returns null when nothing stored", () => {
    expect(getStoredToken()).toBeNull();
  });

  it("setStoredToken + getStoredToken round-trips", () => {
    setStoredToken("my-jwt-token");
    expect(getStoredToken()).toBe("my-jwt-token");
  });

  it("clearAuth removes token", () => {
    setStoredToken("some-token");
    clearAuth();
    expect(getStoredToken()).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// User storage tests
// ---------------------------------------------------------------------------

describe("User storage", () => {
  beforeEach(() => localStorageMock.clear());

  const mockUser: UserResponse = {
    id: "user-123",
    email: "test@example.com",
    subscription_tier: "free",
    is_verified: true,
    watchlist: ["AAPL", "MSFT"],
    created_at: "2025-01-01T00:00:00Z",
  };

  it("getStoredUser returns null when nothing stored", () => {
    expect(getStoredUser()).toBeNull();
  });

  it("setStoredUser + getStoredUser round-trips", () => {
    setStoredUser(mockUser);
    const retrieved = getStoredUser();
    expect(retrieved).not.toBeNull();
    expect(retrieved!.id).toBe(mockUser.id);
    expect(retrieved!.email).toBe(mockUser.email);
    expect(retrieved!.watchlist).toEqual(mockUser.watchlist);
  });

  it("clearAuth removes user", () => {
    setStoredUser(mockUser);
    clearAuth();
    expect(getStoredUser()).toBeNull();
  });

  it("getStoredUser returns null for corrupted JSON", () => {
    localStorageMock.setItem("qf_user", "not-valid-json{{{");
    expect(getStoredUser()).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// ApiError tests
// ---------------------------------------------------------------------------

describe("ApiError", () => {
  it("is an instance of Error", () => {
    const err = new ApiError(404, "Not found");
    expect(err).toBeInstanceOf(Error);
    expect(err).toBeInstanceOf(ApiError);
  });

  it("preserves status and message", () => {
    const err = new ApiError(422, "Validation error", { field: "email" });
    expect(err.status).toBe(422);
    expect(err.message).toBe("Validation error");
    expect(err.detail).toEqual({ field: "email" });
  });

  it("name is ApiError", () => {
    const err = new ApiError(500, "Internal server error");
    expect(err.name).toBe("ApiError");
  });

  it("can be caught as Error", () => {
    expect(() => {
      throw new ApiError(401, "Unauthorized");
    }).toThrow(Error);
  });

  it("can be caught as ApiError", () => {
    expect(() => {
      throw new ApiError(403, "Forbidden");
    }).toThrow(ApiError);
  });
});

// ---------------------------------------------------------------------------
// Type shape tests
// ---------------------------------------------------------------------------

describe("UserResponse shape", () => {
  it("free tier user has correct shape", () => {
    const user: UserResponse = {
      id: "abc",
      email: "user@test.com",
      subscription_tier: "free",
      is_verified: false,
      watchlist: [],
      created_at: new Date().toISOString(),
    };
    expect(user.subscription_tier).toBe("free");
    expect(Array.isArray(user.watchlist)).toBe(true);
  });

  it("premium tier user", () => {
    const user: UserResponse = {
      id: "def",
      email: "pro@test.com",
      subscription_tier: "premium",
      is_verified: true,
      watchlist: ["AAPL", "MSFT", "GOOGL"],
      created_at: new Date().toISOString(),
    };
    expect(user.subscription_tier).toBe("premium");
    expect(user.watchlist.length).toBe(3);
  });
});
