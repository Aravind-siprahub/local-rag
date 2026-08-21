import sys
import os

sys.path.insert(0, os.path.abspath("backend"))

import py_compile

try:
    py_compile.compile("backend/app/retrieval/retriever.py", doraise=True)
    print("✅ backend/app/retrieval/retriever.py compiled with ZERO syntax errors on disk!")
except Exception as e:
    print(f"❌ Syntax Error: {e}")
