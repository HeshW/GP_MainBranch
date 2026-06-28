import Image, { type StaticImageData } from "next/image";
import { Card } from "@/components/ui/card";
import doctorOne from "../../../doctors/dr1.png";
import doctorTwo from "../../../doctors/dr2.png";
import doctorThree from "../../../doctors/dr3.png";
import doctorFour from "../../../doctors/dr4.png";

type Doctor = {
  image: StaticImageData;
  name: string;
  specialty: string;
};

export const doctors: Doctor[] = [
  {
    image: doctorOne,
    name: "Dr. Jason Kovalsky",
    specialty: "Cardiologist",
  },
  {
    image: doctorTwo,
    name: "Patricia Mcneel",
    specialty: "Pediatrician",
  },
  {
    image: doctorThree,
    name: "William Khanna",
    specialty: "Throat Specialist",
  },
  
];

type DoctorTeamProps = {
  eyebrow: string;
  title: string;
  body?: string;
  limit?: number;
};

export function DoctorTeam({ eyebrow, title, body, limit }: DoctorTeamProps) {
  const shownDoctors = typeof limit === "number" ? doctors.slice(0, limit) : doctors;

  return (
    <section>
      <div className="text-center">
        <p className="text-sm font-bold uppercase tracking-wide text-[var(--brand-primary)]">{eyebrow}</p>
        <h2 className="mx-auto mt-3 max-w-2xl text-3xl font-bold tracking-tight text-[var(--brand-heading)] sm:text-4xl">
          {title}
        </h2>
        {body ? <p className="mx-auto mt-4 max-w-2xl text-sm leading-7 text-[var(--brand-muted)]">{body}</p> : null}
      </div>

      <div className="mt-10 grid gap-6 sm:grid-cols-2 xl:grid-cols-4">
        {shownDoctors.map((doctor) => (
          <Card
            key={doctor.name}
            className="group overflow-hidden p-0 text-center"
          >
            <div className="relative m-3 aspect-[4/5] overflow-hidden rounded-2xl bg-[var(--brand-soft)]">
              <Image
                src={doctor.image}
                alt={doctor.name}
                fill
                sizes="(min-width: 1280px) 25vw, (min-width: 640px) 50vw, 100vw"
                className="object-cover transition duration-500 group-hover:scale-105"
              />
            </div>
            <div className="px-4 py-5">
              <h3 className="text-xl font-semibold text-[var(--brand-heading)]">{doctor.name}</h3>
              <p className="mt-2 text-sm font-medium text-[var(--brand-primary)]">{doctor.specialty}</p>
            </div>
          </Card>
        ))}
      </div>
    </section>
  );
}
