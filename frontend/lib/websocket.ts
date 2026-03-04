/**
 * QuantFlow WebSocket manager.
 *
 * Manages WebSocket connections for real-time signal streaming.
 * Handles auto-reconnect with exponential back-off, ping/pong,
 * and typed message dispatch.
 */

import type { WsMessage, SignalResponse } from "./types";
import { getStoredToken } from "./api";

type MessageHandler = (msg: WsMessage) => void;
type ErrorHandler = (err: Event) => void;

const WS_BASE =
  typeof window !== "undefined"
    ? (process.env.NEXT_PUBLIC_WS_URL ?? `ws://${window.location.host}`)
    : "ws://localhost:8000";

const MAX_RECONNECT_DELAY = 30_000;
const INITIAL_RECONNECT_DELAY = 1_000;

class QuantFlowWebSocket {
  private ws: WebSocket | null = null;
  private reconnectDelay = INITIAL_RECONNECT_DELAY;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private destroyed = false;
  private readonly handlers = new Set<MessageHandler>();
  private readonly errorHandlers = new Set<ErrorHandler>();

  constructor(private readonly path: string) {}

  /** Connect (or reconnect) to the WebSocket endpoint. */
  connect(): void {
    if (this.destroyed) return;
    const token = getStoredToken();
    const url = token
      ? `${WS_BASE}/ws/${this.path}?token=${token}`
      : `${WS_BASE}/ws/${this.path}`;

    this.ws = new WebSocket(url);

    this.ws.onopen = () => {
      this.reconnectDelay = INITIAL_RECONNECT_DELAY;
    };

    this.ws.onmessage = (event: MessageEvent) => {
      try {
        const msg = JSON.parse(event.data as string) as WsMessage;
        this.handlers.forEach((h) => h(msg));
      } catch {
        // Ignore malformed messages
      }
    };

    this.ws.onerror = (event: Event) => {
      this.errorHandlers.forEach((h) => h(event));
    };

    this.ws.onclose = () => {
      if (!this.destroyed) {
        this.scheduleReconnect();
      }
    };
  }

  private scheduleReconnect(): void {
    if (this.reconnectTimer !== null) return;
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      this.reconnectDelay = Math.min(
        this.reconnectDelay * 2,
        MAX_RECONNECT_DELAY,
      );
      this.connect();
    }, this.reconnectDelay);
  }

  /** Send a raw string message (e.g. "ping"). */
  send(data: string): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(data);
    }
  }

  /** Subscribe to incoming messages. Returns an unsubscribe function. */
  subscribe(handler: MessageHandler): () => void {
    this.handlers.add(handler);
    return () => this.handlers.delete(handler);
  }

  /** Subscribe to WebSocket errors. Returns an unsubscribe function. */
  onError(handler: ErrorHandler): () => void {
    this.errorHandlers.add(handler);
    return () => this.errorHandlers.delete(handler);
  }

  /** Close the connection permanently (no reconnect). */
  destroy(): void {
    this.destroyed = true;
    if (this.reconnectTimer !== null) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    this.ws?.close();
    this.ws = null;
    this.handlers.clear();
    this.errorHandlers.clear();
  }

  get isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }
}

// ---------------------------------------------------------------------------
// Factory helpers
// ---------------------------------------------------------------------------

/** Create a WebSocket for a single symbol's signal stream. */
export function createSignalSocket(symbol: string): QuantFlowWebSocket {
  const sock = new QuantFlowWebSocket(`signals/${symbol.toUpperCase()}`);
  sock.connect();
  return sock;
}

/** Create a WebSocket for the global market stream. */
export function createMarketSocket(): QuantFlowWebSocket {
  const sock = new QuantFlowWebSocket("market");
  sock.connect();
  return sock;
}

export type { QuantFlowWebSocket };
