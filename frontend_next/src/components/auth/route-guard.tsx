"use client";

import type { ReactNode } from "react";
import { useEffect } from "react";
import { usePathname, useRouter } from "next/navigation";
import { useAuth } from "@/contexts/auth-context";

type RouteGuardProps = {
  children: ReactNode;
  requireAuth?: boolean;
  guestOnly?: boolean;
  fallbackLabel?: string;
};

export function RouteGuard({
  children,
  requireAuth = false,
  guestOnly = false,
  fallbackLabel = "Loading...",
}: RouteGuardProps) {
  const { isReady, isAuthenticated } = useAuth();
  const router = useRouter();
  const pathname = usePathname();

  useEffect(() => {
    if (!isReady) {
      return;
    }

    if (requireAuth && !isAuthenticated) {
      router.replace(`/login?next=${encodeURIComponent(pathname)}`);
      return;
    }

    if (guestOnly && isAuthenticated) {
      router.replace("/");
    }
  }, [guestOnly, isAuthenticated, isReady, pathname, requireAuth, router]);

  if (!isReady || (requireAuth && !isAuthenticated) || (guestOnly && isAuthenticated)) {
    return (
      <div className="mx-auto max-w-7xl px-4 py-16 sm:px-6 lg:px-8">
        <div className="rounded-3xl border border-[var(--brand-border)] bg-white/90 px-6 py-12 text-center text-slate-600 shadow-[var(--brand-shadow)]">
          {fallbackLabel}
        </div>
      </div>
    );
  }

  return <>{children}</>;
}
