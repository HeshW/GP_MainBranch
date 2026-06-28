# Nabda SEO Strategy

This frontend treats SEO as a product feature, not just page titles.

## Technical SEO

- Centralized SEO config in `src/lib/seo.ts`.
- Page-level metadata for public routes: `/`, `/chatbot`, `/doctors`, `/services`, `/contact`.
- Private/demo routes are marked `noindex`: `/cart`, `/checkout`, `/orders`, `/login`.
- `robots.txt` allows public medical pages and blocks API/private flows.
- `sitemap.xml` includes canonical URLs and English/Arabic alternates.
- Canonical base URL comes from `NEXT_PUBLIC_SITE_URL`, with `https://nabda-care.com` as the default production placeholder.

## Structured Data

The root layout injects JSON-LD for:

- `MedicalOrganization`
- `Organization`
- `WebSite`
- `SoftwareApplication`
- `FAQPage`

This helps search engines understand Nabda as a bilingual healthcare-oriented AI assistant with doctor search, symptom triage, report OCR, and lab analysis.

## Bilingual Discoverability

- Metadata includes English and Arabic keywords.
- Sitemap alternates expose `?lang=en` and `?lang=ar`.
- HTML language and direction are updated client-side from the language preference.
- Arabic and English brand names are present in metadata and page copy.

## Social Sharing

- Open Graph and Twitter cards use the Nabda logo.
- Each indexable page has its own title and description.

## Discussion Point

The SEO implementation can be presented as a strengthened point because it covers:

- discoverability,
- bilingual access,
- medical structured data,
- share previews,
- crawl control,
- and a maintainable configuration layer.
