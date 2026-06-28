"use client";

import { Card } from "@/components/ui/card";
import { SectionHeading } from "@/components/ui/section-heading";
import { usePreferences } from "@/contexts/preferences-context";
import { getCopy } from "@/lib/i18n";

const serviceMarks = ["AI", "OCR", "LAB", "MAP"];

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
        </div>
      </div>
    </main>
  );
}
