"""Syntax and Import Check Script."""
import sys

try:
    from app.rag.intent_router import classify, Route
    print("[SUCCESS] app.rag.intent_router imported cleanly!")
except Exception as e:
    print(f"[ERROR] Import failed: {e}")
    sys.exit(1)

try:
    from app.main import app
    print("[SUCCESS] app.main imported cleanly!")
except Exception as e:
    print(f"[ERROR] app.main import failed: {e}")
    sys.exit(1)
