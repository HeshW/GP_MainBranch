type SectionHeadingProps = {
  eyebrow?: string;
  title: string;
  description?: string;
};

export function SectionHeading({ eyebrow, title, description }: SectionHeadingProps) {
  return (
    <div className="max-w-2xl">
      {eyebrow ? <p className="text-sm font-semibold uppercase tracking-[0.16em] text-[var(--brand-primary)]">{eyebrow}</p> : null}
      <h2 className="mt-3 text-3xl font-semibold text-[var(--brand-heading)] md:text-4xl">{title}</h2>
      {description ? <p className="mt-3 text-base leading-7 text-[var(--brand-muted)]">{description}</p> : null}
    </div>
  );
}
