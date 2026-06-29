"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useAuth } from "@/contexts/auth-context";
import { usePreferences } from "@/contexts/preferences-context";
import { getCopy } from "@/lib/i18n";

const links = [
  { href: "/", key: "home" },
  { href: "/chatbot", key: "assistant" },
  { href: "/mental-support", key: "assistant", label: "Mental Support" },
  { href: "/doctors", key: "doctors" },
  { href: "/services", key: "services" },
  { href: "/contact", key: "contact" },
] as const;

export function Navbar() {
  const pathname = usePathname();
  const { isAuthenticated, user, logout, isReady } = useAuth();
  const { language, theme, toggleLanguage, toggleTheme } = usePreferences();
  const t = getCopy(language);

  return (
    <header className="sticky top-0 z-40 border-b border-[var(--brand-border)] bg-[var(--brand-surface-glass)] text-[var(--brand-text)] shadow-sm backdrop-blur-xl">
      <div className="mx-auto flex max-w-7xl items-center justify-between gap-4 px-4 py-2.5 sm:px-6 lg:px-8">
        <Link href="/" className="flex min-w-0 items-center gap-2">
          <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-2xl bg-[var(--brand-primary)] text-lg font-black text-white shadow-sm shadow-blue-900/15">
            N
          </span>
          <span className="truncate text-lg font-bold text-[var(--brand-heading)]">{t.brand}</span>
        </Link>

        <nav className="hidden items-center gap-1 md:flex">
          {links.map((link) => {
            const active = pathname === link.href || pathname.startsWith(`${link.href}/`);
            return (
              <Link
                key={link.href}
                href={link.href}
                className={`rounded-2xl px-3 py-2 text-sm font-semibold transition ${
                  active
                    ? "bg-[var(--brand-soft)] text-[var(--brand-primary)] shadow-sm"
                    : "text-[var(--brand-muted)] hover:bg-[var(--brand-soft)] hover:text-[var(--brand-heading)]"
                }`}
              >
                {"label" in link ? link.label : t.nav[link.key]}
              </Link>
            );
          })}
        </nav>

        <div className="flex items-center gap-2">
          <button
            type="button"
            className="rounded-2xl border border-[var(--brand-border)] bg-[var(--brand-surface)] px-3 py-2 text-sm font-semibold text-[var(--brand-text)] transition hover:bg-[var(--brand-soft)]"
            onClick={toggleLanguage}
          >
            {t.nav.language}
          </button>
          <button
            type="button"
            className="rounded-2xl border border-[var(--brand-border)] bg-[var(--brand-surface)] px-3 py-2 text-sm font-semibold text-[var(--brand-text)] transition hover:bg-[var(--brand-soft)]"
            onClick={toggleTheme}
            aria-label="Toggle theme"
          >
            {theme === "care" ? "Night" : "Care"}
          </button>
          <Link
            href="/chatbot"
            className="hidden rounded-2xl bg-[var(--brand-primary)] px-4 py-2 text-sm font-semibold text-white shadow-lg shadow-blue-900/15 transition hover:bg-[var(--brand-primary-strong)] sm:inline-flex"
          >
            {t.nav.start}
          </Link>
          {isReady && isAuthenticated ? (
            <div className="hidden items-center gap-3 rounded-2xl border border-[var(--brand-border)] bg-[var(--brand-surface)] px-3 py-2 text-sm lg:flex">
              <span className="font-medium text-[var(--brand-text)]">
                {t.nav.hi}, {user?.name}
              </span>
              <button className="font-semibold text-[var(--brand-primary)] transition hover:text-[var(--brand-primary-strong)]" onClick={logout}>
                {t.nav.signOut}
              </button>
            </div>
          ) : (
            <Link
              href="/login"
              className="hidden rounded-2xl border border-[var(--brand-border)] bg-[var(--brand-surface)] px-4 py-2 text-sm font-semibold text-[var(--brand-primary)] shadow-sm transition hover:bg-[var(--brand-soft)] sm:inline-flex"
            >
              {t.nav.login}
            </Link>
          )}
        </div>
      </div>
    </header>
  );
}
