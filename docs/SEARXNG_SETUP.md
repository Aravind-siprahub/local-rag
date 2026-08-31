# Self-Hosted SearXNG Setup Guide (Free Web Search)

This Local RAG application supports real-time web search using **SearXNG**, an open-source, privacy-respecting metasearch engine. **No paid API keys (Tavily, Serper, Bing, Google) are required.**

---

## 1. Quick Start with Docker (Recommended)

Run SearXNG locally using Docker Compose:

```bash
docker compose -f docker-compose.searxng.yml up -d
```

Verify SearXNG is running:
- Open http://localhost:8080 in your browser.
- Test JSON API endpoint: `http://localhost:8080/search?q=test&format=json`

---

## 2. Backend Environment Variables

In your `backend/.env` file:

```env
WEB_SEARCH_ENABLED=true
WEB_SEARCH_PROVIDER=searxng
SEARXNG_URL=http://localhost:8080
WEB_SEARCH_MAX_RESULTS=5
WEB_SEARCH_TIMEOUT=10
WEB_SEARCH_MAX_CONTENT_LENGTH=50000
```

> **Note for Docker-based deployments**:
> If running the backend inside a Docker container, set:
> `SEARXNG_URL=http://searxng:8080`

---

## 3. Zero-Config Fallback (DuckDuckGo)

If SearXNG is temporarily stopped or unconfigured, the application automatically falls back to **DuckDuckGo Instant Answer & HTML Search**, ensuring real-time web search continues working out of the box without requiring any API key.
