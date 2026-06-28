import type { InputHTMLAttributes } from "react";

type InputProps = InputHTMLAttributes<HTMLInputElement>;

export function Input({ className, ...props }: InputProps) {
  return (
    <input
      className={[
        "w-full rounded-2xl border border-[var(--brand-border)] bg-[var(--brand-surface)] px-4 py-3 text-sm text-[var(--brand-text)] shadow-sm outline-none transition placeholder:text-slate-400 focus:border-[var(--brand-primary)] focus:ring-4 focus:ring-blue-500/10",
        className,
      ]
        .filter(Boolean)
        .join(" ")}
      {...props}
    />
  );
}
