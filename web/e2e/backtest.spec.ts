import { expect, test } from "@playwright/test";
import { json, mockApi, STRATEGIES } from "./mocks";

const DONE = {
  job_id: "job-123",
  status: "SUCCESS",
  progress: null,
  error: null,
  result: {
    metrics: { sharpe: 0.8123, max_drawdown: 0.2, total_return: 0.35, trades: 17 },
    equity: [
      { ts: "2026-01-01T00:00:00Z", value: 10_000 },
      { ts: "2026-01-01T01:00:00Z", value: 10_500 },
      { ts: "2026-01-01T02:00:00Z", value: 13_500 },
    ],
  },
};

test.beforeEach(async ({ page }) => {
  await mockApi(page, "GET", "/strategies", (r) => json(r, STRATEGIES));
});

test("strategy dropdown is filled from the API", async ({ page }) => {
  await page.goto("/backtest");
  const options = page.getByLabel("Strategy").locator("option");
  await expect(options).toHaveText(["sma_cross", "momentum", "buy_and_hold"]);
});

test("run posts the form and renders metrics from the finished job", async ({ page }) => {
  let posted: Record<string, unknown> | null = null;
  await mockApi(page, "POST", "/backtests", (r) => {
    posted = r.request().postDataJSON();
    return json(r, { job_id: "job-123", kind: "backtest" }, 202);
  });
  await mockApi(page, "GET", "/jobs/job-123", (r) => json(r, DONE));
  await page.routeWebSocket(/\/ws\/jobs\//, () => {});

  await page.goto("/backtest");
  await page.getByLabel("Symbol").fill("eth-usd"); // input upper-cases what is typed
  await page.getByLabel("Strategy").selectOption("momentum");
  await page.getByRole("button", { name: "Run" }).click();

  await expect(page.getByText("job-123")).toBeVisible();
  await expect(page.getByText("SUCCESS")).toBeVisible();
  await expect(page.getByText("0.8123")).toBeVisible(); // sharpe, 4 decimals
  await expect(page.getByText("equity curve: 3 points")).toBeVisible();
  await expect(page.getByText(/10000 → 13500/)).toBeVisible();

  expect(posted).toMatchObject({
    symbol: "ETH-USD",
    strategy: "momentum",
    timeframe: "1h",
    fee_bps: 10,
    slippage_bps: 5,
    cash: 10_000,
  });
});

test("websocket updates move a pending job to success", async ({ page }) => {
  await mockApi(page, "POST", "/backtests", (r) =>
    json(r, { job_id: "job-123", kind: "backtest" }, 202),
  );
  await mockApi(page, "GET", "/jobs/job-123", (r) =>
    json(r, { job_id: "job-123", status: "PENDING" }),
  );
  await page.routeWebSocket(/\/ws\/jobs\/job-123/, (ws) => {
    // gửi progress trước rồi mới tới kết quả cuối
    // progress first, then the final result, like the real job socket
    ws.send(JSON.stringify({ job_id: "job-123", status: "PROGRESS", progress: { pct: 50 } }));
    setTimeout(() => ws.send(JSON.stringify(DONE)), 300);
  });

  await page.goto("/backtest");
  await page.getByRole("button", { name: "Run" }).click();
  await expect(page.getByText("SUCCESS")).toBeVisible();
  await expect(page.getByText("0.8123")).toBeVisible();
});

test("validation error (422) is shown to the user", async ({ page }) => {
  await mockApi(page, "POST", "/backtests", (r) =>
    json(r, { detail: [{ msg: "unknown strategy 'sma_cross'" }] }, 422),
  );

  await page.goto("/backtest");
  await page.getByRole("button", { name: "Run" }).click();
  await expect(page.getByText("unknown strategy 'sma_cross'")).toBeVisible();
  await expect(page.getByText(/^job /)).toHaveCount(0);
});

test("failed job shows the worker error", async ({ page }) => {
  await mockApi(page, "POST", "/backtests", (r) =>
    json(r, { job_id: "job-9", kind: "backtest" }, 202),
  );
  await mockApi(page, "GET", "/jobs/job-9", (r) =>
    json(r, { job_id: "job-9", status: "FAILURE", error: "no bars for BTC-USD" }),
  );
  await page.routeWebSocket(/\/ws\/jobs\//, () => {});

  await page.goto("/backtest");
  await page.getByRole("button", { name: "Run" }).click();
  await expect(page.getByText("FAILURE")).toBeVisible();
  await expect(page.getByText("no bars for BTC-USD")).toBeVisible();
});
