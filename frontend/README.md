# Local RAG Frontend

React 19 + Vite + TypeScript frontend for the Local RAG backend.

## Stack

- React 19, Vite, TypeScript
- React Router, TanStack Query, Axios
- Tailwind CSS, shadcn/ui
- React Hook Form, Zod (for upcoming forms)

## Development

1. Start the backend from `backend/`:

```bash
uvicorn app.main:app --reload
```

2. Install and run the frontend:

```bash
cd frontend
npm install
npm run dev
```

The Vite dev server proxies `/api` to `http://127.0.0.1:8000`, so no backend CORS changes are required.

## Environment

Optional `.env`:

```env
VITE_API_BASE_URL=/api
```

## Current milestone

- Project setup, routing, shared layout
- Dashboard with live API data (documents + chat sessions)
