import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

import App from "./App";

beforeEach(() => {
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => ({
      ok: true,
      status: 200,
      statusText: "OK",
      text: async () =>
        JSON.stringify({
          api_version: "1.0.0",
          rag_enabled: false,
          faiss_configured: false,
        }),
    })) as unknown as typeof fetch,
  );
});

afterEach(() => {
  vi.unstubAllGlobals();
});

test("app smoke render shows user mode by default and can switch to workbench", async () => {
  render(<App />);

  expect(screen.getByText("GP Medical Report Analysis")).toBeTruthy();
  expect(screen.getByText(/Guided medical analysis/i)).toBeTruthy();
  expect(screen.getByRole("tab", { name: "User interface" })).toBeTruthy();

  fireEvent.click(screen.getByRole("tab", { name: "Dev workbench" }));

  expect(screen.getByRole("button", { name: "Manual labs" })).toBeTruthy();
  expect(screen.getByRole("button", { name: "Report image" })).toBeTruthy();
  expect(screen.getByRole("button", { name: "Symptoms text" })).toBeTruthy();

  await waitFor(() => {
    expect(screen.getByText(/API v1\.0\.0/i)).toBeTruthy();
  });
});
