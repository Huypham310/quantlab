import { expect, test, type Page } from "@playwright/test";
import { ACCOUNT, json, makeBars, mockApi, STRATEGIES, SYMBOLS } from "./mocks";

// mock hết mọi endpoint để chuyển trang không đập vào API chết
// Stub every endpoint any page touches, so navigation never hits a dead API.
async function stubEverything(page: Page) {
  await mockApi(page, "GET", "/symbols", (r) => json(r, SYMBOLS));
  await mockApi(page, "GET", /^\/bars\/.+/, (r) => json(r, makeBars(30)));
  await mockApi(page, "GET", "/strategies", (r) => json(r, STRATEGIES));
  await mockApi(page, "GET", "/paper/account", (r) => json(r, ACCOUNT));
  await mockApi(page, "GET", "/paper/orders", (r) => json(r, []));
  await page.routeWebSocket(/\/ws\//, () => {});
}

test.beforeEach(async ({ page }) => {
  await stubEverything(page);
});

test("header shows the brand and three nav links", async ({ page }) => {
  await page.goto("/");
  const nav = page.getByRole("navigation");
  await expect(page.getByRole("link", { name: "QuantLab" })).toBeVisible();
  await expect(nav.getByRole("link")).toHaveText(["Live", "Backtest", "Paper"]);
});

test("clicking through every link lands on the right page", async ({ page }) => {
  await page.goto("/");
  const nav = page.getByRole("navigation");

  await nav.getByRole("link", { name: "Backtest" }).click();
  await expect(page).toHaveURL(/\/backtest$/);
  await expect(page.getByRole("heading", { name: "Backtest" })).toBeVisible();

  await nav.getByRole("link", { name: "Paper" }).click();
  await expect(page).toHaveURL(/\/paper$/);
  await expect(page.getByRole("heading", { name: "Paper" })).toBeVisible();

  await nav.getByRole("link", { name: "Live" }).click();
  await expect(page).toHaveURL(/\/$/);
  await expect(page.getByRole("heading", { name: "Live" })).toBeVisible();
});

test("only the current page's link is highlighted", async ({ page }) => {
  await page.goto("/backtest");
  const nav = page.getByRole("navigation");
  await expect(nav.getByRole("link", { name: "Backtest" })).toHaveClass(/text-teal-800/);
  await expect(nav.getByRole("link", { name: "Live" })).not.toHaveClass(/text-teal-800/);
  await expect(nav.getByRole("link", { name: "Paper" })).not.toHaveClass(/text-teal-800/);
});

test("brand link goes back to the live page", async ({ page }) => {
  await page.goto("/paper");
  await page.getByRole("link", { name: "QuantLab" }).click();
  await expect(page).toHaveURL(/\/$/);
  await expect(page.getByRole("heading", { name: "Live" })).toBeVisible();
});

test("direct visits work and the browser back button returns", async ({ page }) => {
  await page.goto("/paper");
  await expect(page.getByRole("heading", { name: "Paper" })).toBeVisible();
  await page.getByRole("navigation").getByRole("link", { name: "Backtest" }).click();
  await expect(page).toHaveURL(/\/backtest$/);
  await page.goBack();
  await expect(page).toHaveURL(/\/paper$/);
});
