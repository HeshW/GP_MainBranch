# Nabda Next Frontend

Next.js frontend for نبضة - Nabda. The app uses the existing FastAPI backend contract from the old Vite frontend.

## Run Locally

Start the backend first:

```bash
uvicorn app.main:app --reload --app-dir backend
```

Then run the Next frontend:

```bash
npm run dev
```

Open http://localhost:3000.

## Backend Connection

In development, the browser client calls FastAPI directly by default:

```bash
http://127.0.0.1:8000
```

This avoids the Next.js development rewrite proxy timing out on long-running AI
pipeline requests. The same-origin `/api/*` rewrite in `next.config.ts` is
disabled during `next dev` unless `ENABLE_NEXT_BACKEND_PROXY=true` is set, and
remains available for production deployments that intentionally leave
`NEXT_PUBLIC_API_URL` empty.

Use one of these env vars if the backend runs elsewhere:

```bash
BACKEND_API_URL=http://127.0.0.1:8000
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
NEXT_PUBLIC_API_KEY=your-service-api-key
```

For production SEO URLs, set:

```bash
NEXT_PUBLIC_SITE_URL=https://your-domain.com
```

Supported backend flows:

- `POST /api/v1/pipeline/symptoms`
- `POST /api/v1/pipeline/labs`
- `POST /api/v1/pipeline/image`
- `POST /api/v1/pipeline/diagnosis/clarify`
- `POST /api/v1/chat`
- `POST /api/v1/chat/stream`

## Pages

- `/` professional bilingual landing page
- `/chatbot` GPT-style Nabda assistant page
- Floating Nabda chat on non-chatbot pages
- `/doctors`, `/services`, and `/contact` use one Nabda blue-and-white visual identity
- `/doctors` includes browser geolocation plus Google Maps doctor search and directions
- `robots.ts`, `sitemap.ts`, metadata, Open Graph, and branded icons are configured for SEO

## SEO

See `docs/SEO_STRATEGY.md` for the implementation details and discussion points.
