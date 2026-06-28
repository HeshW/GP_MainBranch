import type { SelectHTMLAttributes } from "react";

type SelectProps = SelectHTMLAttributes<HTMLSelectElement>;

export function Select({ className, ...props }: SelectProps) {
  return (
    <select
      className={[
        "w-full rounded-2xl border border-[var(--brand-border)] bg-[var(--brand-surface)] px-4 py-3 text-sm text-[var(--brand-text)] shadow-sm outline-none transition focus:border-[var(--brand-primary)] focus:ring-4 focus:ring-blue-500/10",
        className,
      ]
        .filter(Boolean)
        .join(" ")}
      {...props}
    />
  );
}
