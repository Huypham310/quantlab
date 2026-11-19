import { expect, test } from "@playwright/test";
import { json, makeBars, mockApi, SYMBOLS } from "./mocks";

// Each test mocks only what the Live page asks for: /symbols, /bars/{symbol} and the trade socket.
test.beforeEach(async ({ page }) => {
  await mockApi(page, "GET", "/symbols", (r) => json(r, SYMBOLS));
});

test("renders candles for the default symbol", async ({ page }) => {
  await mockApi(page, "GET", "/bars/BTC-USD", (r) => json(r, makeBars(120)));
  await page.routeWebSocket(/\/ws\/trades\//, () => {}); // socket stays open but silent

  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Live" })).toBeVisible();
  // lightweight-charts draws into <canvas> elements inside the chart host
  await expect(page.locator("canvas").first()).toBeVisible();
  await expect(page.getByText("Loading bars")).toHaveCount(0);
});

test("shows the last print pushed over the websocket", async ({ page }) => {
  await mockApi(page, "GET", "/bars/BTC-USD", (r) => json(r, makeBars(30)));
  await page.routeWebSocket(/\/ws\/trades\/BTC-USD/, (ws) => {
    ws.send(
      JSON.stringify({
        exchange: "coinbase",
        symbol: "BTC-USD",
        trade_id: "42",
        price: 61234.5,
        size: 0.25,
        side: "buy",
        ts_ms: Date.UTC(2026, 0, 6, 12, 0, 0),
      }),
    );
  });

  await page.goto("/");
  const line = page.getByText(/last print/);
  await expect(line).toBeVisible();
  await expect(line).toContainText("61234.50");
  await expect(line).toContainText("buy 0.25");
});

test("switching symbol refetches bars for it", async ({ page }) => {
  const asked: string[] = [];
  await mockApi(page, "GET", /^\/bars\/.+/, (r) => {
    asked.push(new URL(r.request().url()).pathname);
    return json(r, makeBars(40));
  });
  await page.routeWebSocket(/\/ws\/trades\//, () => {});

  await page.goto("/");
  await expect(page.locator("canvas").first()).toBeVisible();
  await page.getByLabel("Symbol").selectOption("ETH-USD");
  await expect.poll(() => asked).toContain("/bars/ETH-USD");
});

test("empty history tells the user to ingest first", async ({ page }) => {
  await mockApi(page, "GET", "/bars/BTC-USD", (r) => json(r, []));
  await page.routeWebSocket(/\/ws\/trades\//, () => {});

  await page.goto("/");
  await expect(page.getByText("No bars for this symbol yet")).toBeVisible();
  await expect(page.locator("canvas")).toHaveCount(0);
});

test("API error is surfaced instead of a blank page", async ({ page }) => {
  await mockApi(page, "GET", "/bars/BTC-USD", (r) =>
    json(r, { detail: "database unavailable" }, 503),
  );
  await page.routeWebSocket(/\/ws\/trades\//, () => {});

  await page.goto("/");
  await expect(page.getByText("database unavailable")).toBeVisible();
});
