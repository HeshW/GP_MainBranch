# GP Medical Analysis — Web UI

React + Vite + TypeScript. Development traffic to `/api/*` is proxied to the FastAPI server on port 8000.

## Setup

```bash
cd frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173` with the backend running (see `backend/README.md`).

## Production build

```bash
npm run build
```

Serve `dist/` behind any static host; set `VITE_API_URL` to the public API base URL if it is not same-origin.
