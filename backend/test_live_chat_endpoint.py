"""Verification script for live WEB route search."""
import asyncio
import sys

from app.tools.web_search import DuckDuckGoWebSearchProvider

async def main():
    print("Testing DuckDuckGo keyless web search provider...")
    provider = DuckDuckGoWebSearchProvider(timeout_seconds=10.0)

    test_queries = [
        "When is Good Friday in 2026?",
        "What is the current Python version?",
        "What happened in today's news?",
    ]

    for idx, q in enumerate(test_queries, 1):
        print(f"\n--- TEST #{idx}: '{q}' ---")
        try:
            result = await provider.search(q)
            print(f"Provider: {result.provider}")
            print(f"Hits count: {len(result.hits)}")
            for h_idx, hit in enumerate(result.hits[:3], 1):
                print(f"  Hit #{h_idx}: {hit.title}")
                print(f"  URL: {hit.url}")
                print(f"  Snippet: {hit.snippet[:140]}...")
            print("\nFormatted Answer:")
            print(result.concise_answer())
        except Exception as exc:
            print(f"ERROR: {exc}")

    await provider.close()

if __name__ == "__main__":
    asyncio.run(main())
