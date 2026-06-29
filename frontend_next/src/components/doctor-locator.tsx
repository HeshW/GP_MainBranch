"use client";

import Image from "next/image";
import { useEffect, useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { usePreferences } from "@/contexts/preferences-context";
import {
  filterDoctors,
  getGoogleMapsUrl,
  rankDoctors,
  type Coordinates,
  type Doctor,
  type RankedDoctor,
} from "@/lib/doctor-locator";
import { fetchFakeDoctors } from "@/lib/mock-doctors";
import { getCopy } from "@/lib/i18n";

type LocationState =
  | { status: "loading"; coordinates: Coordinates; source: "default"; message?: string }
  | { status: "success"; coordinates: Coordinates; source: "browser" | "default"; message?: string }
  | { status: "error"; coordinates: Coordinates; source: "default"; message: string };

const defaultLocation: Coordinates = {
  latitude: 30.0444,
  longitude: 31.2357,
};

const actionLinkClass =
  "inline-flex min-h-11 items-center justify-center rounded-2xl bg-[var(--brand-primary)] px-4 py-3 text-sm font-semibold text-white shadow-lg shadow-blue-900/15 transition duration-200 hover:bg-[var(--brand-primary-strong)] hover:shadow-xl hover:shadow-blue-900/20";

function formatPrice(price: number) {
  return `${price.toLocaleString("en-EG")} EGP`;
}

function formatDistance(distanceKm?: number) {
  if (typeof distanceKm !== "number") {
    return null;
  }

  return distanceKm < 1 ? `${Math.round(distanceKm * 1000)} m` : `${distanceKm.toFixed(1)} km`;
}

function LocationNotice({ locationState }: { locationState: LocationState }) {
  if (locationState.status === "loading") {
    return (
      <div className="rounded-2xl border border-[var(--brand-border)] bg-[var(--brand-soft)] px-4 py-3 text-sm font-semibold text-[var(--brand-primary)]">
        Detecting your location...
      </div>
    );
  }

  if (locationState.status === "error") {
    return (
      <div className="rounded-2xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm font-medium text-amber-900 shadow-sm">
        {locationState.message} Search still works, and map buttons will open the selected clinic location.
      </div>
    );
  }

  if (locationState.source === "browser") {
    return (
      <div className="rounded-2xl border border-emerald-200 bg-emerald-50 px-4 py-3 text-sm font-medium text-emerald-900 shadow-sm">
        Location ready. Results are ranked by distance, rating, and consultation fee.
      </div>
    );
  }

  return (
    <div className="rounded-2xl border border-[var(--brand-border)] bg-[var(--brand-soft)] px-4 py-3 text-sm text-[var(--brand-muted)]">
      Using Cairo as the default area until browser location is available.
    </div>
  );
}

function DoctorCard({
  rankedDoctor,
  origin,
  showDistance,
}: {
  rankedDoctor: RankedDoctor;
  origin?: Coordinates;
  showDistance: boolean;
}) {
  const { doctor, distanceKm } = rankedDoctor;
  const distance = showDistance ? formatDistance(distanceKm) : null;

  return (
    <Card className="overflow-hidden p-0">
      <div className="grid gap-0 sm:grid-cols-[168px_1fr]">
        <div className="relative aspect-[4/3] min-h-44 bg-[var(--brand-soft)] sm:aspect-auto">
          <Image
            src={doctor.image}
            alt={doctor.name}
            fill
            sizes="(min-width: 1024px) 170px, (min-width: 640px) 30vw, 100vw"
            className="object-cover"
          />
        </div>

        <div className="p-5">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div>
              <p className="text-xs font-bold uppercase tracking-[0.14em] text-[var(--brand-primary)]">
                {doctor.specialty}
              </p>
              <h3 className="mt-1 text-xl font-semibold text-[var(--brand-heading)]">{doctor.name}</h3>
            </div>
            <span
              className={[
                "rounded-full px-3 py-1 text-xs font-bold",
                doctor.availableToday
                  ? "bg-emerald-50 text-emerald-700"
                  : "bg-slate-100 text-slate-600",
              ].join(" ")}
            >
              {doctor.availableToday ? "Available today" : "Next availability"}
            </span>
          </div>

          <div className="mt-4 grid gap-2 text-sm sm:grid-cols-2 lg:grid-cols-4">
            <InfoPill label="Rating" value={doctor.rating.toFixed(1)} />
            <InfoPill label="Fee" value={formatPrice(doctor.price)} />
            <InfoPill label="Experience" value={`${doctor.experienceYears} years`} />
            {distance ? <InfoPill label="Distance" value={distance} /> : <InfoPill label="Area" value={doctor.area} />}
          </div>

          <p className="mt-4 text-sm leading-6 text-[var(--brand-muted)]">{doctor.address}</p>

          <div className="mt-5 flex flex-wrap gap-3">
            <a href={getGoogleMapsUrl(doctor, origin)} target="_blank" rel="noreferrer" className={actionLinkClass}>
              Open in Google Maps
            </a>
            <a href={`tel:${doctor.phone.replace(/\s/g, "")}`}>
              <Button type="button" variant="secondary">
                Call
              </Button>
            </a>
          </div>
        </div>
      </div>
    </Card>
  );
}

function InfoPill({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-2xl bg-[var(--brand-soft)] px-3 py-2">
      <span className="block text-[11px] font-bold uppercase tracking-[0.08em] text-[var(--brand-primary)]">
        {label}
      </span>
      <span className="mt-1 block text-sm font-semibold text-[var(--brand-heading)]">{value}</span>
    </div>
  );
}

export function DoctorLocator() {
  const { language } = usePreferences();
  const t = getCopy(language).doctors.locator;
  const [doctors, setDoctors] = useState<Doctor[]>([]);
  const [doctorError, setDoctorError] = useState("");
  const [isLoadingDoctors, setIsLoadingDoctors] = useState(true);
  const [searchTerm, setSearchTerm] = useState("");
  const [locationState, setLocationState] = useState<LocationState>({
    status: "success",
    coordinates: defaultLocation,
    source: "default",
    message: "Using Cairo as the default area until you allow browser location.",
  });

  function requestLocation() {
    if (typeof navigator === "undefined" || !navigator.geolocation) {
      setLocationState({
        status: "error",
        coordinates: defaultLocation,
        source: "default",
        message: t.unavailable,
      });
      return;
    }

    if (!window.isSecureContext) {
      setLocationState({
        status: "error",
        coordinates: defaultLocation,
        source: "default",
        message:
          "Browser location only works on HTTPS or localhost. Open the site on localhost, or use the fallback results.",
      });
      return;
    }

    setLocationState({
      status: "loading",
      coordinates: defaultLocation,
      source: "default",
    });

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
      (error) => {
        const message =
          error.code === error.PERMISSION_DENIED
            ? "Location permission is blocked. Enable location for this site from the browser address bar, then try again."
            : error.code === error.TIMEOUT
              ? "Location lookup timed out. Check browser location access and try again."
              : t.denied;

        setLocationState({
          status: "error",
          coordinates: defaultLocation,
          source: "default",
          message,
        });
      },
      {
        enableHighAccuracy: false,
        timeout: 12000,
        maximumAge: 60000,
      },
    );
  }

  useEffect(() => {
    let isMounted = true;

    fetchFakeDoctors()
      .then((items) => {
        if (isMounted) {
          setDoctors(items);
          setDoctorError("");
        }
      })
      .catch(() => {
        if (isMounted) {
          setDoctorError("Doctors data could not be loaded.");
        }
      })
      .finally(() => {
        if (isMounted) {
          setIsLoadingDoctors(false);
        }
      });

    return () => {
      isMounted = false;
    };
  }, []);

  const specialtyOptions = useMemo(
    () => Array.from(new Set(doctors.map((doctor) => doctor.specialty))).sort(),
    [doctors],
  );
  const browserLocation =
    locationState.status === "success" && locationState.source === "browser"
      ? locationState.coordinates
      : undefined;
  const filteredDoctors = useMemo(() => filterDoctors(doctors, searchTerm), [doctors, searchTerm]);
  const rankedDoctors = useMemo(
    () => rankDoctors(filteredDoctors, locationState.coordinates),
    [filteredDoctors, locationState.coordinates],
  );
  const recommendedDoctor = rankedDoctors[0];

  return (
    <div className="grid gap-6 lg:grid-cols-[0.84fr_1.16fr]">
      <div className="space-y-5">
        <Card className="p-5 sm:p-6">
          <div>
            <label htmlFor="doctor-search" className="mb-2 block text-sm font-semibold text-[var(--brand-text)]">
              Search doctors
            </label>
            <Input
              id="doctor-search"
              value={searchTerm}
              onChange={(event) => setSearchTerm(event.target.value)}
              placeholder="Name, specialty, address, or area"
            />
          </div>

          <div className="mt-4 flex flex-wrap gap-2">
            {specialtyOptions.map((item) => (
              <button
                key={item}
                type="button"
                className="rounded-full border border-[var(--brand-border)] bg-[var(--brand-surface)] px-3 py-2 text-xs font-semibold text-[var(--brand-text)] transition hover:border-[var(--brand-primary)] hover:bg-[var(--brand-soft)]"
                onClick={() => setSearchTerm(item)}
              >
                {item}
              </button>
            ))}
          </div>

          <div className="mt-5 grid gap-3 sm:grid-cols-2">
            <Button type="button" onClick={requestLocation} disabled={locationState.status === "loading"}>
              {locationState.status === "loading" ? "Locating..." : t.useLocation}
            </Button>
            <Button type="button" variant="secondary" onClick={() => setSearchTerm("")}>
              Clear search
            </Button>
          </div>

          <div className="mt-5">
            <LocationNotice locationState={locationState} />
          </div>
        </Card>

        {recommendedDoctor ? (
          <Card className="p-5 sm:p-6">
            <p className="text-sm font-semibold uppercase tracking-[0.16em] text-[var(--brand-primary)]">
              Best match
            </p>
            <div className="mt-4 flex gap-4">
              <div className="relative h-20 w-20 shrink-0 overflow-hidden rounded-2xl bg-[var(--brand-soft)]">
                <Image
                  src={recommendedDoctor.doctor.image}
                  alt={recommendedDoctor.doctor.name}
                  fill
                  sizes="80px"
                  className="object-cover"
                />
              </div>
              <div className="min-w-0">
                <h2 className="text-2xl font-semibold text-[var(--brand-heading)]">
                  {recommendedDoctor.doctor.name}
                </h2>
                <p className="mt-1 text-sm text-[var(--brand-muted)]">
                  {recommendedDoctor.doctor.specialty} in {recommendedDoctor.doctor.area}
                </p>
                <div className="mt-3 flex flex-wrap gap-2 text-xs font-semibold text-[var(--brand-muted)]">
                  <span className="rounded-full border border-[var(--brand-border)] px-3 py-1">
                    {recommendedDoctor.doctor.rating.toFixed(1)} rating
                  </span>
                  <span className="rounded-full border border-[var(--brand-border)] px-3 py-1">
                    {formatPrice(recommendedDoctor.doctor.price)}
                  </span>
                  {browserLocation ? (
                    <span className="rounded-full border border-[var(--brand-border)] px-3 py-1">
                      {formatDistance(recommendedDoctor.distanceKm)}
                    </span>
                  ) : null}
                </div>
              </div>
            </div>
          </Card>
        ) : null}
      </div>

      <div className="space-y-4">
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div>
            <p className="text-sm font-semibold uppercase tracking-[0.16em] text-[var(--brand-primary)]">
              Nearby network
            </p>
            <h2 className="mt-2 text-2xl font-semibold text-[var(--brand-heading)]">Available doctors</h2>
          </div>
          <span className="rounded-full border border-[var(--brand-border)] bg-[var(--brand-surface)] px-3 py-1 text-xs font-semibold text-[var(--brand-muted)]">
            {rankedDoctors.length} results
          </span>
        </div>

        {isLoadingDoctors ? (
          <Card className="p-6 text-sm font-medium text-[var(--brand-muted)]">Loading doctors...</Card>
        ) : null}

        {doctorError ? (
          <Card className="border-red-200 bg-red-50 p-6 text-sm font-medium text-red-800">{doctorError}</Card>
        ) : null}

        {!isLoadingDoctors && !doctorError && rankedDoctors.length === 0 ? (
          <Card className="p-6 text-sm font-medium text-[var(--brand-muted)]">
            No doctors match &quot;{searchTerm}&quot;. Try another name, specialty, or area.
          </Card>
        ) : null}

        {!isLoadingDoctors && !doctorError
          ? rankedDoctors.map((rankedDoctor) => (
              <DoctorCard
                key={rankedDoctor.doctor.id}
                rankedDoctor={rankedDoctor}
                origin={browserLocation}
                showDistance={Boolean(browserLocation)}
              />
            ))
          : null}
      </div>
    </div>
  );
}
