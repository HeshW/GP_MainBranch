"use client";

import { FormEvent, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { SectionHeading } from "@/components/ui/section-heading";
import { usePreferences } from "@/contexts/preferences-context";
import { getCopy } from "@/lib/i18n";

export default function ContactPage() {
  const { language } = usePreferences();
  const t = getCopy(language);
  const [sent, setSent] = useState(false);

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setSent(true);
    event.currentTarget.reset();
  }

  return (
    <main className="nabda-soft-section min-h-[calc(100svh-64px)]">
      <div className="mx-auto max-w-7xl px-4 py-14 sm:px-6 lg:px-8">
        <SectionHeading
          eyebrow={t.contact.eyebrow}
          title={t.contact.title}
          description={t.contact.body}
        />

        <div className="mt-10 grid gap-6 lg:grid-cols-[0.9fr_1.1fr]">
          <div className="space-y-4">
            {[
              ["SUP", t.contact.support, [t.contact.email, t.contact.phone]],
              ["LOC", t.contact.location, [t.contact.address, t.contact.hours]],
              ["24/7", "Urgent care routing", ["For severe symptoms, contact local emergency services immediately."]],
            ].map(([mark, title, lines]) => (
              <Card key={String(title)} className="p-6">
                <div className="flex gap-4">
                  <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-2xl bg-[var(--brand-soft)] text-xs font-black text-[var(--brand-primary)]">
                    {mark}
                  </div>
                  <div>
                    <h2 className="text-lg font-semibold text-[var(--brand-heading)]">{String(title)}</h2>
                    {(lines as string[]).map((line) => (
                      <p key={line} className="mt-2 text-sm leading-6 text-[var(--brand-muted)]">
                        {line}
                      </p>
                    ))}
                  </div>
                </div>
              </Card>
            ))}
          </div>

          <div className="space-y-4">
            <Card className="p-6">
              <h2 className="text-2xl font-semibold text-[var(--brand-heading)]">Send a message</h2>
              <form className="mt-6 space-y-4" onSubmit={handleSubmit}>
                <label className="block text-sm font-semibold text-[var(--brand-text)]">
                  Name
                  <input
                    name="name"
                    className="mt-2 w-full rounded-2xl border border-[var(--brand-border)] bg-[var(--brand-soft)] px-4 py-3 text-sm text-[var(--brand-text)] outline-none transition focus:border-[var(--brand-primary)] focus:bg-[var(--brand-surface)] focus:ring-4 focus:ring-blue-500/10"
                    required
                  />
                </label>
                <label className="block text-sm font-semibold text-[var(--brand-text)]">
                  Email
                  <input
                    name="email"
                    type="email"
                    className="mt-2 w-full rounded-2xl border border-[var(--brand-border)] bg-[var(--brand-soft)] px-4 py-3 text-sm text-[var(--brand-text)] outline-none transition focus:border-[var(--brand-primary)] focus:bg-[var(--brand-surface)] focus:ring-4 focus:ring-blue-500/10"
                    required
                  />
                </label>
                <label className="block text-sm font-semibold text-[var(--brand-text)]">
                  Message
                  <textarea
                    name="message"
                    rows={5}
                    className="mt-2 w-full resize-y rounded-2xl border border-[var(--brand-border)] bg-[var(--brand-soft)] px-4 py-3 text-sm text-[var(--brand-text)] outline-none transition focus:border-[var(--brand-primary)] focus:bg-[var(--brand-surface)] focus:ring-4 focus:ring-blue-500/10"
                    required
                  />
                </label>
                <Button type="submit">Send message</Button>
                {sent ? (
                  <p className="text-sm font-medium text-[var(--brand-primary)]">
                    Thanks. Your message was received in this demo.
                  </p>
                ) : null}
              </form>
            </Card>

            <div className="relative overflow-hidden rounded-3xl border border-[var(--brand-border)] bg-[var(--brand-soft)] p-5 shadow-[var(--brand-shadow)]">
              <div className="absolute inset-0 opacity-60 [background-image:linear-gradient(var(--brand-border)_1px,transparent_1px),linear-gradient(90deg,var(--brand-border)_1px,transparent_1px)] [background-size:34px_34px]" />
              <div className="relative flex min-h-52 items-end">
                <div className="rounded-2xl border border-[var(--brand-border)] bg-[var(--brand-surface-glass)] px-4 py-3 text-sm shadow-sm backdrop-blur">
                  <p className="font-semibold text-[var(--brand-heading)]">Nabda Care Office</p>
                  <p className="mt-1 text-xs text-[var(--brand-muted)]">{t.contact.address}</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}
