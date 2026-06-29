import type { StaticImageData } from "next/image";

export type Coordinates = {
  latitude: number;
  longitude: number;
};

export type Doctor = Coordinates & {
  id: number;
  name: string;
  specialty: string;
  rating: number;
  price: number;
  address: string;
  area: string;
  phone: string;
  image: StaticImageData;
  availableToday: boolean;
  experienceYears: number;
  searchAliases?: string[];
};

export type RankedDoctor = {
  doctor: Doctor;
  distanceKm?: number;
  score: number;
};

const normalize = (value: string) => value.trim().toLowerCase();

export function calculateDistance(from: Coordinates, to: Coordinates) {
  const earthRadiusKm = 6371;
  const latitudeDelta = ((to.latitude - from.latitude) * Math.PI) / 180;
  const longitudeDelta = ((to.longitude - from.longitude) * Math.PI) / 180;
  const startLatitude = (from.latitude * Math.PI) / 180;
  const endLatitude = (to.latitude * Math.PI) / 180;

  const haversine =
    Math.sin(latitudeDelta / 2) ** 2 +
    Math.sin(longitudeDelta / 2) ** 2 * Math.cos(startLatitude) * Math.cos(endLatitude);

  return 2 * earthRadiusKm * Math.asin(Math.sqrt(haversine));
}

export function filterDoctors(doctors: Doctor[], searchTerm: string) {
  const query = normalize(searchTerm);

  if (!query) {
    return doctors;
  }

  return doctors.filter((doctor) =>
    [doctor.name, doctor.specialty, doctor.address, doctor.area, ...(doctor.searchAliases ?? [])]
      .map(normalize)
      .some((value) => value.includes(query)),
  );
}

export function rankDoctors(doctors: Doctor[], userLocation?: Coordinates): RankedDoctor[] {
  if (doctors.length === 0) {
    return [];
  }

  const withDistance = doctors.map((doctor) => ({
    doctor,
    distanceKm: userLocation ? calculateDistance(userLocation, doctor) : undefined,
  }));

  const distances = withDistance
    .map((item) => item.distanceKm)
    .filter((distance): distance is number => typeof distance === "number");
  const maxDistance = Math.max(...distances, 1);
  const prices = doctors.map((doctor) => doctor.price);
  const minPrice = Math.min(...prices);
  const maxPrice = Math.max(...prices);
  const priceRange = Math.max(maxPrice - minPrice, 1);

  return withDistance
    .map(({ doctor, distanceKm }) => {
      const distanceScore =
        typeof distanceKm === "number" ? Math.max(0, 1 - distanceKm / maxDistance) : 0;
      const ratingScore = doctor.rating / 5;
      const priceScore = 1 - (doctor.price - minPrice) / priceRange;

      return {
        doctor,
        distanceKm,
        score: distanceScore * 0.6 + ratingScore * 0.3 + priceScore * 0.1,
      };
    })
    .sort((left, right) => {
      if (right.score !== left.score) {
        return right.score - left.score;
      }

      if (typeof left.distanceKm === "number" && typeof right.distanceKm === "number") {
        return left.distanceKm - right.distanceKm;
      }

      if (right.doctor.rating !== left.doctor.rating) {
        return right.doctor.rating - left.doctor.rating;
      }

      return left.doctor.price - right.doctor.price;
    });
}

export function getGoogleMapsUrl(doctor: Doctor, origin?: Coordinates) {
  const destination = `${doctor.latitude},${doctor.longitude}`;

  if (origin) {
    const params = new URLSearchParams({
      api: "1",
      origin: `${origin.latitude},${origin.longitude}`,
      destination,
      travelmode: "driving",
    });

    return `https://www.google.com/maps/dir/?${params.toString()}`;
  }

  const params = new URLSearchParams({
    api: "1",
    query: `${doctor.name} ${doctor.address} ${destination}`,
  });

  return `https://www.google.com/maps/search/?${params.toString()}`;
}

export const haversineDistanceKm = calculateDistance;
export const buildGoogleMapsDirectionsUrl = getGoogleMapsUrl;
