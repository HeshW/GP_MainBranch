import Link from "next/link";
import { Button } from "@/components/ui/button";

type HeroAction = {
  href: string;
  label: string;
  variant?: "primary" | "secondary" | "ghost";
};

type MedicalHeroProps = {
  eyebrow: string;
  title: string;
  body: string;
  actions: HeroAction[];
  align?: "left" | "center";
};

export function MedicalHero({ eyebrow, title, body, actions, align = "left" }: MedicalHeroProps) {
  const centered = align === "center";

  return (
    <div className={centered ? "mx-auto max-w-3xl text-center" : "max-w-3xl"}>
      <p className="text-sm font-bold uppercase tracking-[0.18em] text-[var(--brand-primary)]">
        {eyebrow}
      </p>
      <h1 className="mt-4 text-4xl font-bold tracking-tight text-[var(--brand-heading)] sm:text-5xl lg:text-6xl">
        {title}
      </h1>
      <p className="mt-5 max-w-2xl text-base leading-8 text-[var(--brand-muted)] sm:text-lg">
        {body}
      </p>
      <div className={["mt-7 flex flex-wrap gap-3", centered ? "justify-center" : ""].join(" ")}>
        {actions.map((action) => (
          <Link key={`${action.href}-${action.label}`} href={action.href}>
            <Button variant={action.variant}>{action.label}</Button>
          </Link>
        ))}
      </div>
    </div>
  );
}
