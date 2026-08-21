import sys
import os

sys.path.insert(0, os.path.abspath("backend"))

from app.retrieval.search import SearchFilters, search_fulltext, search_similar
from app.retrieval.retriever import Retriever
import asyncio

async def main():
    # Test search_fulltext with None filters
    res1 = await search_fulltext(None, "test query", filters=None)
    assert res1 == [], "search_fulltext(None) failed"

    # Test search_similar with None filters
    res2 = await search_similar(None, [0.1]*768, model_name="bge-m3", top_k=5, filters=None)
    assert res2 == [], "search_similar(None) failed"

    # Test Retriever initialization with None session
    r = Retriever(session=None)
    print("✅ All retrieval search functions and Retriever class compile and pass verification cleanly!")

asyncio.run(main())
