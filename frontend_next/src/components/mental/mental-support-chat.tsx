"use client";

import { FormEvent, useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { postMentalHealthChat } from "@/lib/api";
import type { MentalHealthChatResponse } from "@/lib/api";

type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  text: string;
  safetyStatus?: string;
};

function SupportIcon() {
  return (
    <svg viewBox="0 0 48 48" className="h-7 w-7" aria-hidden="true">
      <path
        d="M15 16.5a8 8 0 0 1 15.5-2.7A7 7 0 1 1 33 27H20.5A6.5 6.5 0 0 1 15 16.5Z"
        fill="none"
        stroke="currentColor"
        strokeWidth="3"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path
        d="M17 32c4.5 5 9.5 5 14 0"
        fill="none"
        stroke="currentColor"
        strokeWidth="3"
        strokeLinecap="round"
      />
    </svg>
  );
}

export function MentalSupportChat() {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: "welcome",
      role: "assistant",
      text: "Hi, I am here for emotional support. What feels heaviest right now?",
      safetyStatus: "safe",
    },
  ]);
  const [lastResponse, setLastResponse] = useState<MentalHealthChatResponse | null>(null);
  const [isSending, setIsSending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const language = useMemo<"en" | "ar">(() => (/[\u0600-\u06FF]/.test(input) ? "ar" : "en"), [input]);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const message = input.trim();
    if (!message || isSending) return;

    setInput("");
    setError(null);
    setIsSending(true);
    const userMessage: ChatMessage = {
      id: `user-${Date.now()}`,
      role: "user",
      text: message,
    };
    setMessages((current) => [...current, userMessage]);

    try {
      const response = await postMentalHealthChat({ message, language });
      setLastResponse(response);
      setMessages((current) => [
        ...current,
        {
          id: `assistant-${Date.now()}`,
          role: "assistant",
          text: response.reply,
          safetyStatus: response.safety_status,
        },
      ]);
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "Unable to reach Mental Support.");
    } finally {
      setIsSending(false);
    }
  }

  return (
    <section className="min-h-[calc(100svh-64px)] bg-[var(--brand-bg)]">
      <div className="mx-auto grid max-w-7xl gap-8 px-4 py-10 sm:px-6 lg:grid-cols-[360px_minmax(0,1fr)] lg:px-8">
        <aside className="rounded-lg border border-[var(--brand-border)] bg-[var(--brand-surface)] p-6 shadow-[var(--brand-shadow)]">
          <div className="flex h-14 w-14 items-center justify-center rounded-lg bg-[var(--brand-soft)] text-[var(--brand-primary)]">
            <SupportIcon />
          </div>
          <h1 className="mt-6 text-3xl font-bold tracking-tight text-[var(--brand-heading)]">Mental Support</h1>
          <p className="mt-3 text-base leading-7 text-[var(--brand-muted)]">Talk to a supportive AI assistant</p>
          <p className="mt-6 rounded-lg border border-amber-200 bg-amber-50 p-4 text-sm leading-6 text-amber-950">
            This assistant provides emotional support only and is not a replacement for a therapist or emergency care.
          </p>
          <p className="mt-4 text-xs leading-5 text-[var(--brand-muted)]">
            It does not provide formal diagnosis, medication prescriptions, or licensed therapy.
          </p>
        </aside>

        <div className="flex min-h-[640px] flex-col rounded-lg border border-[var(--brand-border)] bg-[var(--brand-surface)] shadow-[var(--brand-shadow)]">
          <div className="border-b border-[var(--brand-border)] px-5 py-4">
            <p className="text-sm font-semibold text-[var(--brand-heading)]">Support chat</p>
            <p className="mt-1 text-xs text-[var(--brand-muted)]">
              {lastResponse?.model_loaded
                ? "LoRA model loaded"
                : lastResponse?.safety_status === "unavailable"
                  ? "Model unavailable fallback"
                  : "Guardrails active"}
              {lastResponse?.latency_ms ? ` · ${lastResponse.latency_ms} ms` : ""}
            </p>
          </div>

          <div className="flex-1 space-y-4 overflow-y-auto px-5 py-5">
            {messages.map((message) => {
              const isCrisis = message.safetyStatus === "crisis";
              const isUnavailable = message.safetyStatus === "unavailable";
              return (
                <div
                  key={message.id}
                  className={`max-w-[82%] rounded-lg px-4 py-3 text-sm leading-6 ${
                    message.role === "user"
                      ? "ml-auto bg-[var(--brand-primary)] text-white"
                      : isCrisis
                        ? "border border-red-200 bg-red-50 text-red-950"
                        : isUnavailable
                          ? "border border-amber-200 bg-amber-50 text-amber-950"
                        : "bg-[var(--brand-soft)] text-[var(--brand-text)]"
                  }`}
                >
                  {isCrisis ? <p className="mb-2 text-xs font-bold uppercase tracking-wide">Crisis support</p> : null}
                  {isUnavailable ? (
                    <p className="mb-2 text-xs font-bold uppercase tracking-wide">Safety fallback</p>
                  ) : null}
                  <p className="whitespace-pre-wrap">{message.text}</p>
                </div>
              );
            })}
            {error ? (
              <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-950">{error}</div>
            ) : null}
          </div>

          <form onSubmit={handleSubmit} className="border-t border-[var(--brand-border)] p-4">
            <div className="flex gap-3">
              <textarea
                value={input}
                onChange={(event) => setInput(event.target.value)}
                placeholder="Share what you are feeling..."
                className="min-h-12 flex-1 resize-none rounded-lg border border-[var(--brand-border)] bg-[var(--brand-bg)] px-4 py-3 text-sm text-[var(--brand-text)] outline-none transition focus:border-[var(--brand-primary)]"
                rows={2}
              />
              <Button type="submit" disabled={isSending || !input.trim()} className="h-12 self-end rounded-lg">
                {isSending ? "Sending" : "Send"}
              </Button>
            </div>
          </form>
        </div>
      </div>
    </section>
  );
}
