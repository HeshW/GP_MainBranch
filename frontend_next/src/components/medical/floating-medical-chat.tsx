"use client";

import { useState } from "react";
import { usePathname } from "next/navigation";
import { usePreferences } from "@/contexts/preferences-context";
import { getCopy } from "@/lib/i18n";
import { MedicalChat } from "./medical-chat";

export function FloatingMedicalChat() {
  const pathname = usePathname();
  const { language } = usePreferences();
  const t = getCopy(language);
  const [open, setOpen] = useState(false);

  if (pathname.startsWith("/chatbot")) return null;

  return (
    <div className="fixed bottom-5 right-5 z-50">
      {open ? (
        <div className="mb-3 w-[min(400px,calc(100vw-2rem))] overflow-hidden rounded-3xl border border-[var(--brand-border)] bg-[var(--brand-surface)] shadow-[var(--brand-shadow-hover)]">
          <div className="flex items-center justify-between border-b border-[var(--brand-border)] bg-[var(--brand-surface)] px-4 py-3">
            <div>
              <p className="text-xs font-semibold uppercase text-[var(--brand-primary)]">{t.brand}</p>
              <p className="text-sm font-semibold text-[var(--brand-heading)]">{t.chat.placeholder}</p>
            </div>
            <button
              type="button"
              className="rounded-2xl border border-[var(--brand-border)] px-3 py-2 text-sm font-semibold text-[var(--brand-muted)] transition hover:bg-[var(--brand-soft)]"
              onClick={() => setOpen(false)}
              aria-label="Close Nabda chat"
            >
              Close
            </button>
          </div>
          <div className="h-[680px] max-h-[calc(100vh-10rem)]">
            <MedicalChat compact />
          </div>
        </div>
      ) : null}

      <button
        type="button"
        className="ml-auto flex h-12 items-center gap-2 rounded-full bg-[var(--brand-primary)] px-3.5 pr-4 text-sm font-semibold text-white shadow-lg shadow-blue-900/15 transition hover:bg-[var(--brand-primary-strong)] hover:shadow-xl hover:shadow-blue-900/20"
        onClick={() => setOpen((current) => !current)}
        aria-expanded={open}
        aria-label="Open Nabda assistant"
      >
        <span className="flex h-7 w-7 items-center justify-center rounded-full bg-[var(--brand-surface)] text-xs font-bold text-[var(--brand-primary)]">
          ن
        </span>
        {t.brandArabic}
      </button>
    </div>
  );
}
