"use client";

import { Card } from "@/components/ui/card";
import { SectionHeading } from "@/components/ui/section-heading";
import { usePreferences } from "@/contexts/preferences-context";
import { getCopy } from "@/lib/i18n";
import Link from "next/link";

const serviceMarks = ["AI", "OCR", "LAB", "MAP", "MH"];

export default function ServicesPage() {
  const { language } = usePreferences();
  const t = getCopy(language);

  return (
    <main className="nabda-soft-section min-h-[calc(100svh-64px)]">
      <div className="mx-auto max-w-7xl px-4 py-14 sm:px-6 lg:px-8">
        <SectionHeading
          eyebrow={t.services.eyebrow}
          title={t.services.title}
          description={t.services.body}
        />

        <div className="mt-10 grid gap-6 md:grid-cols-2">
          {t.services.items.map(([title, text], index) => (
            <Card key={title} className="p-6">
              <div className="mb-6 flex h-12 w-12 items-center justify-center rounded-2xl bg-[var(--brand-soft)] text-xs font-black tracking-wide text-[var(--brand-primary)]">
                {serviceMarks[index] ?? `0${index + 1}`}
              </div>
              <h2 className="text-xl font-semibold text-[var(--brand-heading)]">{title}</h2>
              <p className="mt-3 text-sm leading-7 text-[var(--brand-muted)]">{text}</p>
            </Card>
          ))}
          <Link
            href="/mental-support"
            className="nabda-card-hover rounded-lg border border-[var(--brand-border)] bg-[var(--brand-surface)] p-6 shadow-[var(--brand-shadow)] transition"
          >
            <div className="mb-6 flex h-12 w-12 items-center justify-center rounded-lg bg-[var(--brand-soft)] text-xs font-black tracking-wide text-[var(--brand-primary)]">
              {serviceMarks[4]}
            </div>
            <h2 className="text-xl font-semibold text-[var(--brand-heading)]">Mental Support</h2>
            <p className="mt-3 text-sm leading-7 text-[var(--brand-muted)]">
              Talk to a supportive AI assistant for stress, sadness, and coping support without diagnosis.
            </p>
          </Link>
        </div>
      </div>
    </main>
  );
}
