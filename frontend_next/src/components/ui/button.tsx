import type { ButtonHTMLAttributes } from "react";

type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "primary" | "secondary" | "ghost";
};

export function Button({ className, variant = "primary", ...props }: ButtonProps) {
  const styles = {
    primary:
      "bg-[var(--brand-primary)] text-white shadow-lg shadow-blue-900/15 hover:bg-[var(--brand-primary-strong)] hover:shadow-xl hover:shadow-blue-900/20",
    secondary:
      "border border-[var(--brand-border)] bg-[var(--brand-surface)] text-[var(--brand-primary)] shadow-sm hover:border-[var(--brand-border-strong)] hover:bg-[var(--brand-soft)] hover:shadow-md",
    ghost: "bg-transparent text-[var(--brand-primary)] hover:bg-[var(--brand-soft)]",
  }[variant];

  return (
    <button
      className={[
        "inline-flex items-center justify-center rounded-2xl px-5 py-3 text-sm font-semibold transition duration-200 disabled:cursor-not-allowed disabled:opacity-60",
        styles,
        className,
      ]
        .filter(Boolean)
        .join(" ")}
      {...props}
    />
  );
}
