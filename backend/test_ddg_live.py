"""Standalone verification script for DuckDuckGo live web search."""
import asyncio
import logging

import httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def run_live_test():
    from app.tools.web_search import DuckDuckGoWebSearchProvider

    provider = DuckDuckGoWebSearchProvider(timeout_seconds=10.0)
    questions = [
        "When is Good Friday in 2026?",
        "What is the current Python version?",
        "What happened in today's news?",
    ]

    for q in questions:
        print(f"\n==========================================")
        print(f"QUERY: {q}")
        print(f"==========================================")
        try:
            result = await provider.search(q)
            print(f"PROVIDER: {result.provider}")
            print(f"HITS COUNT: {len(result.hits)}")
            for idx, hit in enumerate(result.hits[:3], 1):
                print(f"  Hit {idx}: {hit.title}")
                print(f"  URL  : {hit.url}")
                print(f"  Snippet: {hit.snippet[:120]}...")
            print("\nCONCISE ANSWER:")
            print(result.concise_answer())
        except Exception as exc:
            print(f"ERROR: {exc}")

    await provider.close()

if __name__ == "__main__":
    asyncio.run(run_live_test())
