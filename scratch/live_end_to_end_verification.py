"""Live End-to-End Verification script testing real DuckDuckGo network searches, LLM integration, and source citations."""
from __future__ import annotations

import sys
import os
import asyncio
import uuid
import logging

script_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.abspath(os.path.join(script_dir, "..", "backend"))
if os.path.exists(backend_dir) and backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)
elif os.path.exists(os.path.join(script_dir, "app")) and script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Configure logging to capture [WEB SEARCH] tags
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("LIVE_VERIFICATION")

from app.tools.web_search import DuckDuckGoWebSearchProvider, WebSearchError
from app.rag.intent_router import Route, classify
from app.prompting.builder import PromptBuilder
from app.llm.ollama_client import get_global_ollama_client
from app.rag.service import RAGService, _validate_web_answer


async def test_live_duckduckgo_query(query: str):
    print(f"\n==================================================")
    print(f"🔍 TESTING LIVE QUERY: '{query}'")
    print(f"==================================================")

    # 1. Test Intent Classification
    route = classify(query)
    print(f"1. Router Selected Route: {route.name} ({route.value})")

    # 2. Test Real DuckDuckGo Search Execution
    provider = DuckDuckGoWebSearchProvider()
    try:
        start_time = asyncio.get_event_loop().time()
        result = await provider.search(query)
        elapsed = asyncio.get_event_loop().time() - start_time
        
        print(f"2. DuckDuckGo Execution: SUCCESS ({elapsed:.2f}s)")
        print(f"   Provider: {result.provider}")
        print(f"   Hits Count: {len(result.hits)}")
        
        if result.hits:
            first_hit = result.hits[0]
            print(f"   First Hit Title:  '{first_hit.title}'")
            print(f"   First Hit URL:    '{first_hit.url}'")
            print(f"   First Hit Source: '{first_hit.source}'")
            print(f"   Snippet Preview:  '{first_hit.snippet[:100]}...'")
        
        # 3. Test LLM Context Integration
        llm = get_global_ollama_client()
        web_context = "\n\n".join(
            f"Source {i}: {h.title} ({h.url})\nSnippet: {h.snippet}"
            for i, h in enumerate(result.hits[:5], start=1)
        )
        
        from app.core.config import get_settings
        sys_prompt = get_settings().WEB_SEARCH_SYSTEM_PROMPT
        user_prompt = (
            f"WEB SEARCH RESULTS (retrieved live via DuckDuckGo):\n\n{web_context}\n\n"
            f"User Question:\n{query}\n\n"
            f"Instructions: Answer using the web search results above. Include relevant source URLs."
        )

        print("3. Sending Web Search Context to Ollama (Qwen 3 8B)...")
        resp = await llm.generate(sys_prompt, user_prompt, num_predict=512)
        answer = resp.answer.strip()
        print(f"4. Raw LLM Response Preview:\n{answer[:250]}...\n")

        # 4. Verify No Internet Disclaimer
        disclaimer_phrases = ["cannot access the internet", "cannot access github", "cannot perform external"]
        has_disclaimer = any(p in answer.lower() for p in disclaimer_phrases)
        if has_disclaimer:
            print("❌ FAILURE: LLM returned false 'cannot access internet' disclaimer!")
        else:
            print("✔ SUCCESS: LLM used search results without false disclaimer.")

        # 5. Verify URL Preservation
        urls_in_context = [h.url for h in result.hits if h.url]
        urls_in_answer = [h.url for h in result.hits if h.url and (h.url in answer or h.title in answer)]
        print(f"5. Sources Preserved in Answer: {len(urls_in_answer)} / {len(urls_in_context)}")

        return {
            "query": query,
            "route": route.value,
            "hits_count": len(result.hits),
            "first_title": result.hits[0].title if result.hits else "N/A",
            "first_url": result.hits[0].url if result.hits else "N/A",
            "disclaimer_free": not has_disclaimer,
            "answer_preview": answer[:150],
        }

    except WebSearchError as exc:
        print(f"❌ Web Search Error: {exc}")
        return {"query": query, "error": str(exc)}
    finally:
        await provider.close()


async def run_all_live_verifications():
    print("==================================================")
    print("🚀 STARTING REAL-TIME PUBLIC WEB SEARCH VERIFICATION")
    print("==================================================")

    queries = [
        "Search the internet for the latest FastAPI release.",
        "Look up PostHog and tell me what it is.",
        "Search GitHub for FastAPI authentication examples.",
        "Search Reddit for Qwen3 8B user experiences.",
        "Find the latest Python release.",
        "Search the web for current AI news.",
    ]

    results = []
    for q in queries:
        res = await test_live_duckduckgo_query(q)
        results.append(res)

    print("\n==================================================")
    print("📊 SUMMARY OF REAL-TIME LIVE NETWORK SEARCH RESULTS")
    print("==================================================")
    for i, r in enumerate(results, 1):
        if "error" in r:
            print(f"{i}. {r['query']} -> ERROR: {r['error']}")
        else:
            print(f"{i}. Query: '{r['query']}'")
            print(f"   Route: {r['route']} | Hits: {r['hits_count']} | First Source: {r['first_title']} ({r['first_url']})")
            print(f"   Disclaimer-Free: {r['disclaimer_free']}")

    print("\n==================================================")
    print("🎉 REAL-TIME LIVE VERIFICATION COMPLETE")
    print("==================================================")

if __name__ == "__main__":
    asyncio.run(run_all_live_verifications())
