"use client";

import { Card } from "@/components/ui/card";
import { DoctorTeam } from "@/components/doctors/doctor-team";
import { MedicalHero } from "@/components/medical-hero";
import { usePreferences } from "@/contexts/preferences-context";
import { getCopy } from "@/lib/i18n";

export default function HomePage() {
  const { language } = usePreferences();
  const t = getCopy(language);

  return (
    <div className="bg-[var(--brand-bg)] text-[var(--brand-text)]">
      <section className="relative min-h-[calc(100svh-64px)] overflow-hidden bg-[var(--brand-bg)]">
        <div
          className="absolute inset-0 bg-[url('/hero-medical-robots.png')] bg-[length:auto_92%] bg-[position:right_center] bg-no-repeat opacity-95"
          aria-hidden="true"
        />
        <div
          className="dark-hero-overlay absolute inset-0 bg-[linear-gradient(90deg,var(--brand-bg)_0%,rgba(255,255,255,0.94)_36%,rgba(255,255,255,0.60)_58%,rgba(255,255,255,0.10)_100%)]"
          aria-hidden="true"
        />

        <div className="relative mx-auto flex min-h-[calc(100svh-64px)] max-w-7xl items-center px-4 py-14 sm:px-6 lg:px-8">
          <div className="w-full max-w-3xl">
            <MedicalHero
              eyebrow={t.home.eyebrow}
              title={t.home.title}
              body={t.home.body}
              actions={[
                { href: "/chatbot", label: t.home.openAssistant },
                { href: "/doctors", label: t.home.findDoctors, variant: "secondary" },
              ]}
            />

            <div className="mt-10 grid gap-3 sm:grid-cols-3">
              {t.home.stats.map(([value, label], index) => (
                <div
                  key={label}
                  className="rounded-3xl border border-[var(--brand-border)] bg-[var(--brand-surface-glass)] px-5 py-5 shadow-[var(--brand-shadow)] backdrop-blur-xl"
                >
                  <div className="mb-3 flex h-10 w-10 items-center justify-center rounded-2xl bg-[var(--brand-soft)] text-sm font-bold text-[var(--brand-primary)]">
                    {index + 1}
                  </div>
                  <p className="text-2xl font-bold text-[var(--brand-heading)]">{value}</p>
                  <p className="mt-1 text-xs font-semibold uppercase tracking-[0.16em] text-[var(--brand-muted)]">
                    {label}
                  </p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      <main>
        <section className="mx-auto max-w-7xl px-4 py-20 sm:px-6 lg:px-8">
          <DoctorTeam
            eyebrow={t.doctors.teamEyebrow}
            title={t.doctors.teamTitle}
            body={t.doctors.teamBody}
            limit={4}
          />
        </section>

        <section className="nabda-soft-section py-20">
          <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
            <div className="mb-10 max-w-2xl">
              <p className="text-sm font-semibold uppercase tracking-[0.16em] text-[var(--brand-primary)]">
                {t.services?.eyebrow ?? "Services"}
              </p>
              <h2 className="mt-3 text-3xl font-bold tracking-tight text-[var(--brand-heading)] sm:text-4xl">
                What Nabda can help with
              </h2>
              <p className="mt-4 text-sm leading-7 text-[var(--brand-muted)]">
                A calm medical workflow for triage, report analysis, and doctor handoff.
              </p>
            </div>

            <div className="grid gap-6 md:grid-cols-3">
              {t.home.flows.map(([title, description], index) => (
                <Card key={title}>
                  <div className="mb-4 flex h-11 w-11 items-center justify-center rounded-2xl bg-[var(--brand-soft)] text-sm font-bold text-[var(--brand-primary)]">
                    0{index + 1}
                  </div>
                  <h2 className="text-xl font-semibold text-[var(--brand-heading)]">{title}</h2>
                  <p className="mt-3 text-sm leading-7 text-[var(--brand-muted)]">{description}</p>
                </Card>
              ))}
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}
