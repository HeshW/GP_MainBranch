"use client";

import { DoctorLocator } from "@/components/doctor-locator";

export default function DoctorsPage() {
  return (
    <main className="bg-[var(--brand-bg)] text-[var(--brand-text)]">
      <section id="doctor-locator" className="mx-auto max-w-7xl scroll-mt-24 px-4 py-12 sm:px-6 lg:px-8">
        <div className="mb-8 max-w-2xl">
          <p className="text-sm font-semibold uppercase tracking-[0.16em] text-[var(--brand-primary)]">
            Doctor finder
          </p>
          <h1 className="mt-3 text-4xl font-bold tracking-tight text-[var(--brand-heading)] sm:text-5xl">
            Find nearby doctors
          </h1>
          <p className="mt-4 text-sm leading-7 text-[var(--brand-muted)]">
            Search by specialty, use your location, and open Google Maps directions when you are ready.
          </p>
        </div>
        <DoctorLocator />
      </section>
    </main>
  );
}
