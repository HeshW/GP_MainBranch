import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, test, vi } from "vitest";

import { ChatInterface } from "./ChatInterface";

function buildStreamResponse(chunks: string[]): Response {
  const encoder = new TextEncoder();
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of chunks) {
        controller.enqueue(encoder.encode(`data: ${chunk}\n\n`));
      }
      controller.close();
    },
  });

  return new Response(body, {
    status: 200,
    headers: { "Content-Type": "text/event-stream" },
  });
}

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

test("chat interface consumes stream endpoint and renders streamed chunks", async () => {
  const fetchMock = vi.fn().mockResolvedValue(buildStreamResponse(["Hello ", "doctor"]));
  vi.stubGlobal("fetch", fetchMock as unknown as typeof fetch);

  render(<ChatInterface />);

  fireEvent.change(screen.getByPlaceholderText("اكتب سؤالك الطبي هنا"), {
    target: { value: "hi" },
  });
  fireEvent.click(screen.getByRole("button", { name: "إرسال" }));

  await waitFor(() => {
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/v1/chat/stream",
      expect.objectContaining({ method: "POST" }),
    );
  });

  await waitFor(() => {
    expect(screen.getByText("Hello doctor")).toBeTruthy();
  });
});

test("chat interface falls back to non-stream chat when stream request fails", async () => {
  const fetchMock = vi
    .fn()
    .mockRejectedValueOnce(new Error("stream failed"))
    .mockResolvedValueOnce(
      new Response(JSON.stringify({ response: "Fallback reply" }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );

  vi.stubGlobal("fetch", fetchMock as unknown as typeof fetch);

  render(<ChatInterface />);

  fireEvent.change(screen.getByPlaceholderText("اكتب سؤالك الطبي هنا"), {
    target: { value: "need help" },
  });
  fireEvent.click(screen.getByRole("button", { name: "إرسال" }));

  await waitFor(() => {
    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      "/api/v1/chat/stream",
      expect.objectContaining({ method: "POST" }),
    );
  });

  await waitFor(() => {
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      "/api/v1/chat",
      expect.objectContaining({ method: "POST" }),
    );
  });

  await waitFor(() => {
    expect(screen.getByText("Fallback reply")).toBeTruthy();
  });
});
