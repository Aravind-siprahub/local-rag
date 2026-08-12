"""Verification script for live WEB route search."""
import asyncio
import sys

from app.tools.web_search import DuckDuckGoWebSearchProvider

async def main():
    print("Testing DuckDuckGo keyless web search provider...")
    provider = DuckDuckGoWebSearchProvider(timeout_seconds=10.0)

    test_queries = [
        "when is Pongal",
        "When is Deepawali in 2026?",
        "latest Python release",
        "What is today's date?",
    ]

    all_passed = True
    for idx, q in enumerate(test_queries, 1):
        print(f"\n--- TEST #{idx}: '{q}' ---")
        try:
            result = await provider.search(q, request_id=f"test-req-{idx}")
            print(f"Provider: {result.provider}")
            print(f"Hits count: {len(result.hits)}")
            assert len(result.hits) > 0, f"Expected hits > 0 for query: {q}"
            for h_idx, hit in enumerate(result.hits[:3], 1):
                print(f"  Hit #{h_idx}: {hit.title}")
                print(f"  URL: {hit.url}")
                print(f"  Snippet: {hit.snippet[:140]}...")
            print("\nFormatted Answer:")
            print(result.concise_answer())
            print(f"TEST #{idx} PASSED")
        except Exception as exc:
            print(f"TEST #{idx} FAILED: {exc}")
            all_passed = False

    await provider.close()
    if not all_passed:
        sys.exit(1)
    print("\nALL WEB SEARCH PROVIDER TESTS PASSED!")

if __name__ == "__main__":
    asyncio.run(main())
