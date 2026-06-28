export type Doctor = {
  id: number;
  name: string;
  specialty: string;
  clinicName: string;
  address: string;
  city: string;
  phone: string;
  latitude: number;
  longitude: number;
  rating: number;
  waitTime: string;
};

export type Coordinates = {
  latitude: number;
  longitude: number;
};

export const fakeDoctors: Doctor[] = [
  {
    id: 1,
    name: "Dr. Omar Hassan",
    specialty: "Internal Medicine",
    clinicName: "BlueCare Medical Center",
    address: "12 Nile Corniche, Garden City",
    city: "Cairo",
    phone: "+20 100 123 4567",
    latitude: 30.0439,
    longitude: 31.2357,
    rating: 4.9,
    waitTime: "15 min",
  },
  {
    id: 2,
    name: "Dr. Sara Khaled",
    specialty: "Pediatrics",
    clinicName: "Astra Family Clinic",
    address: "45 Tahrir Street, Downtown",
    city: "Cairo",
    phone: "+20 100 222 3344",
    latitude: 30.0488,
    longitude: 31.2389,
    rating: 4.8,
    waitTime: "20 min",
  },
  {
    id: 3,
    name: "Dr. Ahmed Mostafa",
    specialty: "Orthopedics",
    clinicName: "North Point Clinic",
    address: "8 El-Thawra Avenue, Heliopolis",
    city: "Cairo",
    phone: "+20 100 555 6677",
    latitude: 30.0928,
    longitude: 31.3235,
    rating: 4.7,
    waitTime: "30 min",
  },
  {
    id: 4,
    name: "Dr. Mona Youssef",
    specialty: "Dermatology",
    clinicName: "Riverfront Health",
    address: "3 Corniche El-Nil, Zamalek",
    city: "Cairo",
    phone: "+20 100 777 8899",
    latitude: 30.0589,
    longitude: 31.2201,
    rating: 4.85,
    waitTime: "10 min",
  },
];

export function haversineDistanceKm(from: Coordinates, to: Coordinates) {
  const earthRadiusKm = 6371;
  const latitudeDelta = ((to.latitude - from.latitude) * Math.PI) / 180;
  const longitudeDelta = ((to.longitude - from.longitude) * Math.PI) / 180;

  const startLatitude = (from.latitude * Math.PI) / 180;
  const endLatitude = (to.latitude * Math.PI) / 180;

  const haversine =
    Math.sin(latitudeDelta / 2) * Math.sin(latitudeDelta / 2) +
    Math.sin(longitudeDelta / 2) * Math.sin(longitudeDelta / 2) * Math.cos(startLatitude) * Math.cos(endLatitude);

  return 2 * earthRadiusKm * Math.asin(Math.sqrt(haversine));
}

export function findNearestDoctor(location: Coordinates) {
  const scoredDoctors = fakeDoctors.map((doctor) => ({
    doctor,
    distanceKm: haversineDistanceKm(location, doctor),
  }));

  scoredDoctors.sort((left, right) => left.distanceKm - right.distanceKm);
  return scoredDoctors[0];
}

export function buildGoogleMapsDirectionsUrl(destination: Coordinates, origin?: Coordinates) {
  const params = new URLSearchParams({
    api: "1",
    destination: `${destination.latitude},${destination.longitude}`,
  });

  if (origin) {
    params.set("origin", `${origin.latitude},${origin.longitude}`);
  }

  return `https://www.google.com/maps/dir/?${params.toString()}`;
}

export function buildGoogleMapsDoctorSearchUrl(location: Coordinates, specialty = "doctor") {
  const query = `${specialty || "doctor"} near ${location.latitude},${location.longitude}`;
  return `https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(query)}`;
}
