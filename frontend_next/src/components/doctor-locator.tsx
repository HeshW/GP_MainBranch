"use client";

import { useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { usePreferences } from "@/contexts/preferences-context";
import { getCopy } from "@/lib/i18n";
import {
  buildGoogleMapsDirectionsUrl,
  buildGoogleMapsDoctorSearchUrl,
  fakeDoctors,
  findNearestDoctor,
  haversineDistanceKm,
  type Coordinates,
  type Doctor,
} from "@/lib/doctor-locator";

type LocationState =
  | { status: "loading" }
  | { status: "success"; coordinates: Coordinates; source: "default" | "browser" }
  | { status: "error"; message: string };

const defaultLocation: Coordinates = {
  latitude: 30.0444,
  longitude: 31.2357,
};

const specialties = ["Internal Medicine", "Pediatrics", "Dermatology", "Orthopedics"];

function initials(name: string) {
  return name
    .replace(/^Dr\.\s+/i, "")
    .split(" ")
    .map((part) => part[0])
    .slice(0, 2)
    .join("");
}

function DoctorListCard({ doctor, distanceKm }: { doctor: Doctor; distanceKm: number }) {
  return (
    <Card className="p-4">
      <div className="flex gap-4">
        <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-2xl bg-[var(--brand-soft)] text-sm font-bold text-[var(--brand-primary)]">
          {initials(doctor.name)}
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex items-start justify-between gap-3">
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.14em] text-[var(--brand-primary)]">
                {doctor.specialty}
              </p>
              <h3 className="mt-1 text-lg font-semibold text-[var(--brand-heading)]">{doctor.name}</h3>
            </div>
            <span className="rounded-full bg-[var(--brand-soft)] px-3 py-1 text-xs font-bold text-[var(--brand-primary)]">
              {doctor.rating.toFixed(1)}
            </span>
          </div>
          <p className="mt-1 text-sm font-medium text-[var(--brand-text)]">{doctor.clinicName}</p>
          <p className="mt-2 text-sm leading-6 text-[var(--brand-muted)]">{doctor.address}</p>
          <div className="mt-3 flex flex-wrap gap-2 text-xs font-semibold text-[var(--brand-muted)]">
            <span className="rounded-full border border-[var(--brand-border)] px-3 py-1">
              {distanceKm.toFixed(1)} km
            </span>
            <span className="rounded-full border border-[var(--brand-border)] px-3 py-1">
              {doctor.waitTime} wait
            </span>
            <span className="rounded-full border border-[var(--brand-border)] px-3 py-1">
              {doctor.city}
            </span>
          </div>
        </div>
      </div>
    </Card>
  );
}

export function DoctorLocator() {
  const { language } = usePreferences();
  const t = getCopy(language).doctors.locator;
  const [specialty, setSpecialty] = useState("");
  const [locationState, setLocationState] = useState<LocationState>({
    status: "success",
    coordinates: defaultLocation,
    source: "default",
  });

  function requestLocation() {
    if (typeof navigator === "undefined" || !navigator.geolocation) {
      setLocationState({
        status: "error",
        message: t.unavailable,
      });
      return;
    }

    setLocationState({ status: "loading" });
    navigator.geolocation.getCurrentPosition(
      (position) => {
        setLocationState({
          status: "success",
          source: "browser",
          coordinates: {
            latitude: position.coords.latitude,
            longitude: position.coords.longitude,
          },
        });
      },
      () => {
        setLocationState({
          status: "error",
          message: t.denied,
        });
      },
      {
        enableHighAccuracy: true,
        timeout: 7000,
      },
    );
  }

  const activeLocation =
    locationState.status === "success" ? locationState.coordinates : defaultLocation;

  const nearestDoctor = useMemo(() => findNearestDoctor(activeLocation), [activeLocation]);
  const scoredDoctors = useMemo(
    () =>
      fakeDoctors
        .map((doctor) => ({
          doctor,
          distanceKm: haversineDistanceKm(activeLocation, doctor),
        }))
        .sort((left, right) => left.distanceKm - right.distanceKm),
    [activeLocation],
  );
  const mapsSearchUrl = buildGoogleMapsDoctorSearchUrl(activeLocation, specialty || "doctor");
  const directionsUrl = nearestDoctor
    ? buildGoogleMapsDirectionsUrl(nearestDoctor.doctor, activeLocation)
    : "#";

  return (
    <div className="grid gap-6 lg:grid-cols-[0.92fr_1.08fr]">
      <div className="space-y-5">
        <Card className="p-5 sm:p-6">
          <label className="block">
            <span className="mb-2 block text-sm font-semibold text-[var(--brand-text)]">
              {t.term}
            </span>
            <input
              value={specialty}
              onChange={(event) => setSpecialty(event.target.value)}
              className="w-full rounded-2xl border border-[var(--brand-border)] bg-[var(--brand-soft)] px-4 py-3 text-sm text-[var(--brand-text)] shadow-sm outline-none transition focus:border-[var(--brand-primary)] focus:bg-[var(--brand-surface)] focus:ring-4 focus:ring-blue-500/10"
              placeholder="Search specialty, doctor, or clinic"
            />
          </label>

          <div className="mt-4 flex flex-wrap gap-2">
            {specialties.map((item) => (
              <button
                key={item}
                type="button"
                className="rounded-full border border-[var(--brand-border)] bg-[var(--brand-surface)] px-3 py-2 text-xs font-semibold text-[var(--brand-text)] transition hover:border-[var(--brand-primary)] hover:bg-[var(--brand-soft)]"
                onClick={() => setSpecialty(item)}
              >
                {item}
              </button>
            ))}
          </div>

          <div className="mt-5 grid gap-3 sm:grid-cols-2">
            <Button type="button" onClick={requestLocation}>
              {t.useLocation}
            </Button>
            <a href={mapsSearchUrl} target="_blank" rel="noreferrer" className="inline-flex">
              <Button type="button" variant="secondary" className="w-full">
                {t.maps}
              </Button>
            </a>
          </div>

          {locationState.status === "loading" ? (
            <div className="mt-5 rounded-2xl border border-[var(--brand-border)] bg-[var(--brand-soft)] px-4 py-3 text-sm text-[var(--brand-primary)] shadow-sm">
              {t.loading}
            </div>
          ) : null}

          {locationState.status === "error" ? (
            <div className="mt-5 rounded-2xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-900 shadow-sm">
              {locationState.message}
            </div>
          ) : null}
        </Card>

        <div className="relative overflow-hidden rounded-3xl border border-[var(--brand-border)] bg-[var(--brand-soft)] p-5 shadow-[var(--brand-shadow)]">
          <div className="absolute inset-0 opacity-60 [background-image:linear-gradient(var(--brand-border)_1px,transparent_1px),linear-gradient(90deg,var(--brand-border)_1px,transparent_1px)] [background-size:34px_34px]" />
          <div className="relative min-h-56">
            <div className="absolute left-[58%] top-[34%] h-4 w-4 rounded-full bg-[var(--brand-primary)] shadow-[0_0_0_10px_rgba(7,119,232,0.14)]" />
            <div className="absolute left-[22%] top-[58%] h-3 w-3 rounded-full bg-emerald-400 shadow-[0_0_0_8px_rgba(52,211,153,0.14)]" />
            <div className="absolute bottom-5 left-5 rounded-2xl border border-[var(--brand-border)] bg-[var(--brand-surface-glass)] px-4 py-3 text-sm shadow-sm backdrop-blur">
              <p className="font-semibold text-[var(--brand-heading)]">Map preview</p>
              <p className="mt-1 text-xs text-[var(--brand-muted)]">
                Cairo default, updated when browser location is allowed.
              </p>
            </div>
          </div>
        </div>

        {nearestDoctor ? (
          <Card className="p-5 sm:p-6">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div>
                <p className="text-sm font-semibold uppercase tracking-[0.16em] text-[var(--brand-primary)]">
                  {t.recommended}
                </p>
                <h2 className="mt-3 text-2xl font-semibold text-[var(--brand-heading)]">
                  {nearestDoctor.doctor.name}
                </h2>
                <p className="mt-2 text-sm text-[var(--brand-muted)]">
                  {nearestDoctor.doctor.specialty} {t.at} {nearestDoctor.doctor.clinicName}
                </p>
              </div>
              <span className="rounded-full bg-[var(--brand-soft)] px-3 py-2 text-sm font-bold text-[var(--brand-primary)]">
                {nearestDoctor.doctor.rating.toFixed(1)}
              </span>
            </div>

            <div className="mt-5 grid gap-3 text-sm sm:grid-cols-3">
              <div className="rounded-2xl bg-[var(--brand-soft)] px-4 py-3">
                <span className="block text-xs font-semibold uppercase text-[var(--brand-primary)]">{t.wait}</span>
                <span className="mt-1 block font-semibold text-[var(--brand-heading)]">
                  {nearestDoctor.doctor.waitTime}
                </span>
              </div>
              <div className="rounded-2xl bg-[var(--brand-soft)] px-4 py-3">
                <span className="block text-xs font-semibold uppercase text-[var(--brand-primary)]">Distance</span>
                <span className="mt-1 block font-semibold text-[var(--brand-heading)]">
                  {nearestDoctor.distanceKm.toFixed(2)} km
                </span>
              </div>
              <div className="rounded-2xl bg-[var(--brand-soft)] px-4 py-3">
                <span className="block text-xs font-semibold uppercase text-[var(--brand-primary)]">{t.rating}</span>
                <span className="mt-1 block font-semibold text-[var(--brand-heading)]">
                  {nearestDoctor.doctor.rating.toFixed(1)}
                </span>
              </div>
            </div>

            <p className="mt-4 text-sm leading-6 text-[var(--brand-muted)]">
              {nearestDoctor.doctor.address}, {nearestDoctor.doctor.city}
            </p>

            <div className="mt-5 flex flex-wrap gap-3">
              <a href={directionsUrl} target="_blank" rel="noreferrer">
                <Button type="button">{t.directions}</Button>
              </a>
              <a href={`tel:${nearestDoctor.doctor.phone.replace(/\s/g, "")}`}>
                <Button type="button" variant="secondary">Call clinic</Button>
              </a>
            </div>
          </Card>
        ) : null}
      </div>

      <div className="space-y-4">
        <div className="flex items-end justify-between gap-4">
          <div>
            <p className="text-sm font-semibold uppercase tracking-[0.16em] text-[var(--brand-primary)]">
              Nearby network
            </p>
            <h2 className="mt-2 text-2xl font-semibold text-[var(--brand-heading)]">Available doctors</h2>
          </div>
          <span className="rounded-full border border-[var(--brand-border)] bg-[var(--brand-surface)] px-3 py-1 text-xs font-semibold text-[var(--brand-muted)]">
            {scoredDoctors.length} results
          </span>
        </div>

        {scoredDoctors.map(({ doctor, distanceKm }) => (
          <DoctorListCard key={doctor.id} doctor={doctor} distanceKm={distanceKm} />
        ))}
      </div>
    </div>
  );
}
