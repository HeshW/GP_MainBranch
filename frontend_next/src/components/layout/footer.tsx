"use client";

import { usePreferences } from "@/contexts/preferences-context";
import { getCopy } from "@/lib/i18n";

export function Footer() {
  const { language } = usePreferences();
  const t = getCopy(language);

  return (
    <footer className="border-t border-[var(--brand-border)] bg-[var(--brand-surface)]">
      <div className="mx-auto flex max-w-7xl flex-col gap-3 px-4 py-8 text-sm text-[var(--brand-muted)] sm:px-6 lg:px-8 md:flex-row md:items-center md:justify-between">
        <p className="font-semibold text-[var(--brand-primary)]">
          {t.brandArabic} - {t.brand} | {t.tagline}
        </p>
        <p>{t.footerNote}</p>
      </div>
    </footer>
  );
}
