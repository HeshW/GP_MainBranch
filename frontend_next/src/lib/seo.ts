import type { Metadata } from "next";

export const siteConfig = {
  name: "Nabda",
  arabicName: "نبضة",
  fullName: "نبضة - Nabda",
  tagline: "AI Care, Human Touch",
  defaultTitle: "نبضة - Nabda | AI Care, Human Touch",
  description:
    "Nabda is a bilingual AI medical assistant for symptom triage, medical report OCR, lab analysis, doctor search, and follow-up care guidance.",
  url: (process.env.NEXT_PUBLIC_SITE_URL ?? "https://nabda-care.com").replace(/\/$/, ""),
  logo: "/nabda-logo.svg",
  icon: "/nabda-icon.svg",
  locale: "en_US",
  alternateLocale: "ar_EG",
  contactEmail: "support@nabda-care.com",
  phone: "+20 100 123 4567",
  sameAs: [
    "https://www.facebook.com/nabda-care",
    "https://www.linkedin.com/company/nabda-care",
  ],
};

export const seoKeywords = [
  "Nabda",
  "نبضة",
  "AI medical assistant",
  "medical chatbot",
  "symptom checker",
  "doctor finder",
  "medical report OCR",
  "lab analysis",
  "Google Maps doctor search",
  "تحليل الأعراض",
  "مساعد طبي",
  "تحليل التقارير الطبية",
  "البحث عن طبيب",
];

export const indexableRoutes = [
  {
    path: "",
    title: siteConfig.defaultTitle,
    description: siteConfig.description,
    priority: 1,
    changeFrequency: "weekly" as const,
  },
  {
    path: "/chatbot",
    title: "Nabda AI Medical Assistant",
    description:
      "Chat with Nabda for symptom triage, report image analysis, lab interpretation, and follow-up medical guidance.",
    priority: 0.9,
    changeFrequency: "weekly" as const,
  },
  {
    path: "/doctors",
    title: "Find Doctors Near You",
    description:
      "Use Nabda to search nearby doctors with Google Maps and move from AI guidance to real care.",
    priority: 0.85,
    changeFrequency: "monthly" as const,
  },
  {
    path: "/services",
    title: "Nabda Medical AI Services",
    description:
      "Explore Nabda services for symptom triage, medical report OCR, lab analysis, and doctor handoff.",
    priority: 0.8,
    changeFrequency: "monthly" as const,
  },
  {
    path: "/contact",
    title: "Contact Nabda Care Team",
    description:
      "Contact the Nabda care team for support, escalation, and follow-up around the medical assistant workflow.",
    priority: 0.7,
    changeFrequency: "monthly" as const,
  },
];

export function absoluteUrl(path = "") {
  if (!path) return siteConfig.url;
  return `${siteConfig.url}${path.startsWith("/") ? path : `/${path}`}`;
}

export function buildPageMetadata({
  title,
  description,
  path = "",
  image = siteConfig.logo,
  noIndex = false,
}: {
  title: string;
  description: string;
  path?: string;
  image?: string;
  noIndex?: boolean;
}): Metadata {
  const canonical = absoluteUrl(path);

  return {
    title,
    description,
    keywords: seoKeywords,
    alternates: {
      canonical,
      languages: {
        en: `${canonical}?lang=en`,
        ar: `${canonical}?lang=ar`,
      },
    },
    openGraph: {
      title,
      description,
      url: canonical,
      siteName: siteConfig.fullName,
      locale: siteConfig.locale,
      alternateLocale: [siteConfig.alternateLocale],
      type: "website",
      images: [
        {
          url: image,
          width: 1536,
          height: 1024,
          alt: `${siteConfig.fullName} logo`,
        },
      ],
    },
    twitter: {
      card: "summary_large_image",
      title,
      description,
      images: [image],
    },
    robots: noIndex
      ? {
          index: false,
          follow: false,
          googleBot: {
            index: false,
            follow: false,
          },
        }
      : {
          index: true,
          follow: true,
          googleBot: {
            index: true,
            follow: true,
            "max-image-preview": "large",
            "max-snippet": -1,
            "max-video-preview": -1,
          },
        },
  };
}

export function organizationJsonLd() {
  return {
    "@context": "https://schema.org",
    "@type": ["MedicalOrganization", "Organization"],
    "@id": `${siteConfig.url}/#organization`,
    name: siteConfig.fullName,
    alternateName: [siteConfig.name, siteConfig.arabicName],
    url: siteConfig.url,
    logo: absoluteUrl(siteConfig.logo),
    slogan: siteConfig.tagline,
    email: siteConfig.contactEmail,
    telephone: siteConfig.phone,
    medicalSpecialty: [
      "PrimaryCare",
      "Cardiovascular",
      "Pediatric",
      "Dermatology",
      "Therapy",
    ],
    areaServed: [
      {
        "@type": "Country",
        name: "Egypt",
      },
      {
        "@type": "AdministrativeArea",
        name: "Middle East and North Africa",
      },
    ],
    sameAs: siteConfig.sameAs,
  };
}

export function websiteJsonLd() {
  return {
    "@context": "https://schema.org",
    "@type": "WebSite",
    "@id": `${siteConfig.url}/#website`,
    name: siteConfig.fullName,
    url: siteConfig.url,
    inLanguage: ["en", "ar"],
    publisher: {
      "@id": `${siteConfig.url}/#organization`,
    },
  };
}

export function softwareApplicationJsonLd() {
  return {
    "@context": "https://schema.org",
    "@type": "SoftwareApplication",
    "@id": `${siteConfig.url}/#assistant`,
    name: "Nabda Medical Assistant",
    applicationCategory: "HealthApplication",
    operatingSystem: "Web",
    url: absoluteUrl("/chatbot"),
    description:
      "A bilingual AI medical assistant for symptom triage, report OCR, lab analysis, and follow-up guidance.",
    offers: {
      "@type": "Offer",
      price: "0",
      priceCurrency: "USD",
    },
  };
}

export function faqJsonLd() {
  return {
    "@context": "https://schema.org",
    "@type": "FAQPage",
    mainEntity: [
      {
        "@type": "Question",
        name: "What does Nabda do?",
        acceptedAnswer: {
          "@type": "Answer",
          text: "Nabda helps users with symptom triage, medical report OCR, lab analysis, doctor search, and follow-up guidance in English and Arabic.",
        },
      },
      {
        "@type": "Question",
        name: "هل نبضة بديل للطبيب؟",
        acceptedAnswer: {
          "@type": "Answer",
          text: "نبضة نموذج إرشادي وتعليمي يساعد في تنظيم المعلومات الطبية، لكنه ليس بديلا عن الطبيب أو الرعاية الطارئة.",
        },
      },
      {
        "@type": "Question",
        name: "Can Nabda find nearby doctors?",
        acceptedAnswer: {
          "@type": "Answer",
          text: "Yes. Nabda can use browser location to open nearby doctor searches and directions through Google Maps.",
        },
      },
    ],
  };
}
