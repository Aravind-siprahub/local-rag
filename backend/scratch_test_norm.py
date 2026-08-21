import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from app.rag.query_normalizer import normalize_query

query = "what fronted and backend are using talk to my data"
raw, norm, ret = normalize_query(query)
print(f"RAW: {raw}")
print(f"NORM: {norm}")
print(f"RET: {ret}")
