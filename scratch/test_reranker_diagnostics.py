import sys
import os

print("Python executable:", sys.executable)
print("Python path:", sys.path)

try:
    import flashrank
    print("FlashRank imported successfully! Version:", getattr(flashrank, "__version__", "unknown"))
    from flashrank import Ranker, RerankRequest
    ranker = Ranker()
    print("FlashRank Ranker initialized successfully!")
    res = ranker.rerank(RerankRequest(query="What is Talk to My Data?", passages=[{"id": 1, "text": "The solution is a pipeline of RAG and Text-to-SQL."}]))
    print("FlashRank rerank result:", res)
except Exception as e:
    print("FlashRank failed:", type(e), e)

try:
    import importlib
    importlib.import_module("sentence_transformers")
    print("sentence_transformers imported successfully!")
except Exception as e:
    print("sentence_transformers failed:", type(e), e)
