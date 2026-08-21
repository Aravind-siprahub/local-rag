import sys
import os

sys.path.insert(0, os.path.abspath("backend"))

from app.rag.query_normalizer import normalize_query

raw = "what fronted and backend are using talk to my data"
orig, norm, ret = normalize_query(raw)

print(f"Original query:  {orig!r}")
print(f"Normalized query:{norm!r}")
print(f"Retrieval query: {ret!r}")

assert norm == "what frontend and backend are using talk to my data"
assert ret == "what frontend and backend are using talk to my data"

print("\n✅ TYPO CORRECTION RETRIEVAL QUERY TEST PASSED PERFECTLY!")
