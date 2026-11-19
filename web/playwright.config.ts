import { defineConfig, devices } from "@playwright/test";

// Port 3100 so the e2e server never clashes with a `npm run dev` the developer already has on 3000.
const PORT = 3100;

// UI trỏ vào origin giả này, mọi request đều bị page.route() chặn lại.
// The UI is pointed at this origin; every request to it is answered by page.route() mocks
// (see e2e/mocks.ts), so no FastAPI / Postgres / Redis has to be running.
export const MOCK_API = "http://127.0.0.1:8765";

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: true,
  forbidOnly: Boolean(process.env.CI),
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? [["github"], ["html", { open: "never" }]] : "list",
  use: {
    baseURL: `http://localhost:${PORT}`,
    trace: "retain-on-failure",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: {
    // dev server compiles pages on first hit, so the first test can be slow -> generous timeout
    command: `npm run dev -- --port ${PORT}`,
    url: `http://localhost:${PORT}`,
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
    env: { NEXT_PUBLIC_API_URL: MOCK_API },
  },
});
