import { expect, test } from "@playwright/test";
import { ACCOUNT, json, mockApi, order } from "./mocks";

// trang tự poll 2s/lần nên mock đọc từ state có thể đổi
// The page polls /paper/account and /paper/orders every 2 s, so mocks read from mutable state.
test("account cards show equity, cash, open orders and breaches", async ({ page }) => {
  await mockApi(page, "GET", "/paper/account", (r) => json(r, ACCOUNT));
  await mockApi(page, "GET", "/paper/orders", (r) => json(r, []));

  await page.goto("/paper");
  const card = (label: string) => page.locator("div", { hasText: new RegExp(`^${label}`) }).last();
  await expect(card("Equity")).toContainText("10250.50");
  await expect(card("Cash")).toContainText("10000.00");
  await expect(card("Open orders")).toContainText("1");
  await expect(card("Breaches")).toContainText("0");
  await expect(page.getByText("No orders yet.")).toBeVisible();
  await expect(page.getByText("Kill switch is on")).toHaveCount(0);
});

test("kill switch banner appears when the broker is stopped", async ({ page }) => {
  await mockApi(page, "GET", "/paper/account", (r) => json(r, { ...ACCOUNT, killed: true }));
  await mockApi(page, "GET", "/paper/orders", (r) => json(r, []));

  await page.goto("/paper");
  await expect(page.getByText(/Kill switch is on/)).toBeVisible();
});

test("market order ticket posts the right body and reports the fill", async ({ page }) => {
  let posted: Record<string, unknown> | null = null;
  await mockApi(page, "GET", "/paper/account", (r) => json(r, ACCOUNT));
  await mockApi(page, "GET", "/paper/orders", (r) => json(r, []));
  await mockApi(page, "POST", "/paper/orders", (r) => {
    posted = r.request().postDataJSON();
    return json(
      r,
      order({ id: 7, side: "sell", type: "market", limit_price: null, status: "filled", filled_qty: 0.5, avg_price: 60_010 }),
      201,
    );
  });

  await page.goto("/paper");
  await page.getByLabel("Symbol").fill("eth-usd");
  await page.getByLabel("Side").selectOption("sell");
  await page.getByLabel("Qty").fill("0.5");
  await page.getByRole("button", { name: "Submit market" }).click();

  await expect(page.getByText("filled sell 0.5 @ 60010")).toBeVisible();
  expect(posted).toEqual({ symbol: "ETH-USD", side: "sell", qty: 0.5, type: "market" });
});

test("risk rejection reason is shown", async ({ page }) => {
  await mockApi(page, "GET", "/paper/account", (r) => json(r, ACCOUNT));
  await mockApi(page, "GET", "/paper/orders", (r) => json(r, []));
  await mockApi(page, "POST", "/paper/orders", (r) =>
    json(r, order({ status: "rejected", reason: "position limit" }), 201),
  );

  await page.goto("/paper");
  await page.getByRole("button", { name: "Submit market" }).click();
  await expect(page.getByText("rejected: position limit")).toBeVisible();
});

test("bad quantity (422) shows the API message", async ({ page }) => {
  await mockApi(page, "GET", "/paper/account", (r) => json(r, ACCOUNT));
  await mockApi(page, "GET", "/paper/orders", (r) => json(r, []));
  await mockApi(page, "POST", "/paper/orders", (r) =>
    json(r, { detail: [{ msg: "Input should be greater than 0" }] }, 422),
  );

  await page.goto("/paper");
  await page.getByLabel("Qty").fill("0");
  await page.getByRole("button", { name: "Submit market" }).click();
  await expect(page.getByText("Input should be greater than 0")).toBeVisible();
});

test("open orders can be cancelled", async ({ page }) => {
  let orders = [order({ id: 3 }), order({ id: 4, status: "filled", filled_qty: 0.01 })];
  let deleted: string | null = null;
  await mockApi(page, "GET", "/paper/account", (r) => json(r, ACCOUNT));
  await mockApi(page, "GET", "/paper/orders", (r) => json(r, orders));
  await mockApi(page, "DELETE", /^\/paper\/orders\/\d+$/, (r) => {
    deleted = new URL(r.request().url()).pathname;
    orders = orders.filter((o) => o.id !== 3); // the refetch after cancel sees it gone
    return r.fulfill({ status: 204, headers: { "access-control-allow-origin": "*" } });
  });

  await page.goto("/paper");
  const rows = page.locator("tbody tr");
  await expect(rows).toHaveCount(2);
  // only the open order has a cancel button
  await expect(page.getByRole("button", { name: "cancel" })).toHaveCount(1);
  await page.getByRole("button", { name: "cancel" }).click();

  await expect(rows).toHaveCount(1);
  expect(deleted).toBe("/paper/orders/3");
});

