# Agent Router v1 Design

**Goal:** Route `POST /api/chat` questions to RAG / WEB / CALCULATOR / DIRECT without changing ChatRequest/ChatResponse or the RAG retrieval pipeline.

**Architecture:** Insert deterministic `intent_router.classify(question)` in `RAGService.ask()` immediately after USER message persist. Only `Route.RAG` runs embedding → hybrid search → rerank → PromptBuilder. Other routes return `RAGResponse` with empty citations.

**Tech:** FastAPI backend, httpx DuckDuckGo Instant Answer, AST calculator, Ollama for DIRECT.

## Route priority

1. CALCULATOR — arithmetic expressions / percent-of
2. RAG — document/file/knowledge-base cues
3. WEB — current/external info cues
4. DIRECT — default (general knowledge via Ollama, `num_predict=128`)

## Components

- `app/rag/intent_router.py` — `Route` enum + `classify()`
- `app/tools/web_search.py` — `WebSearchProvider` protocol, DuckDuckGo + Stub, factory from `WEB_SEARCH_PROVIDER`
- `app/tools/calculator.py` — safe arithmetic
- `RAGService` — inject `web_search` + `calculator`; branch after persist

## Config

- `WEB_SEARCH_PROVIDER` default `duckduckgo` (`stub` when explicit)
- `WEB_SEARCH_TIMEOUT_SECONDS` default `8.0`

## Testing

Unit tests inject fake providers; no network. Existing RAG tests use document-cue questions so they still enter the retrieval pipeline.
