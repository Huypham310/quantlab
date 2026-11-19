// Shared API mocks for the e2e specs. Nothing here talks to a real backend.
import type { Page, Route } from "@playwright/test";
import { MOCK_API } from "../playwright.config";

const CORS = {
  "access-control-allow-origin": "*",
  "access-control-allow-headers": "*",
  "access-control-allow-methods": "GET,POST,DELETE,OPTIONS",
};

// trả JSON giả cho fetch; origin API khác origin của trang nên cần header CORS
/** Answer a fetch with JSON (the API origin differs from the page's, so CORS headers are needed). */
export function json(route: Route, body: unknown, status = 200) {
  return route.fulfill({
    status,
    contentType: "application/json",
    headers: CORS,
    body: JSON.stringify(body),
  });
}

/** Register a handler for one method + path; CORS preflights are answered automatically. */
export async function mockApi(
  page: Page,
  method: string,
  path: string | RegExp,
  handler: (route: Route) => unknown,
) {
  await page.route(
    (url) =>
      url.origin === MOCK_API &&
      (typeof path === "string" ? url.pathname === path : path.test(url.pathname)),
    (route) => {
      const req = route.request();
      if (req.method() === "OPTIONS") return route.fulfill({ status: 204, headers: CORS });
      // sai method thì nhường cho handler khác đăng ký cùng path (GET vs POST /paper/orders)
      if (req.method() !== method) return route.fallback();
      return handler(route);
    },
  );
}

// --- fixtures -------------------------------------------------------------------------------

export const SYMBOLS = [
  {
    source: "coinbase",
    symbol: "BTC-USD",
    timeframe: "1h",
    bars: 120,
    first: "2026-01-01T00:00:00Z",
    last: "2026-01-05T23:00:00Z",
  },
  {
    source: "coinbase",
    symbol: "ETH-USD",
    timeframe: "1h",
    bars: 120,
    first: "2026-01-01T00:00:00Z",
    last: "2026-01-05T23:00:00Z",
  },
];

/** n hourly candles with a gentle sine wave so the chart has something to draw. */
export function makeBars(n = 120, base = 60_000) {
  const t0 = Date.UTC(2026, 0, 1);
  return Array.from({ length: n }, (_, i) => {
    const open = base + Math.sin(i / 6) * 400;
    const close = base + Math.sin((i + 1) / 6) * 400;
    return {
      ts: new Date(t0 + i * 3_600_000).toISOString(),
      open,
      high: Math.max(open, close) + 50,
      low: Math.min(open, close) - 50,
      close,
      volume: 10 + (i % 7),
    };
  });
}

export const STRATEGIES = [
  { name: "sma_cross", params: ["fast", "slow"], grid: { fast: [12, 24], slow: [96, 168] } },
  { name: "momentum", params: ["lookback"], grid: { lookback: [24, 48] } },
  { name: "buy_and_hold", params: [], grid: {} },
];

export const ACCOUNT = {
  cash: 10_000,
  equity: 10_250.5,
  positions: {},
  last: { "BTC-USD": 60_000 },
  open_orders: 1,
  fees_paid: 3.25,
  killed: false,
  breaches: 0,
};

export function order(over: Partial<Record<string, unknown>> = {}) {
  return {
    id: 1,
    symbol: "BTC-USD",
    side: "buy",
    qty: 0.01,
    type: "limit",
    limit_price: 59_000,
    created_ms: 1_767_225_600_000,
    status: "open",
    filled_qty: 0,
    avg_price: 0,
    reason: "",
    marketable: false,
    ...over,
  };
}
