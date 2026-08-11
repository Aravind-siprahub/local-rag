"""Hit live /api/chat for regression acceptance queries and print routing metadata."""
from __future__ import annotations

import json
import urllib.request

SESSION_ID = "44b29404-d5cb-408f-87fd-16845736aafa"
BASE = "http://127.0.0.1:8000/api/chat"

QUERIES = [
    "what tech stack were using for talk to my data",
    "tell frontend and backend what using for talk to my data",
    "AIRIS what tech stack were using tell",
    "what is PM2?",
    "earth is 2 planet or 3 planet",
]


def ask(question: str) -> dict:
    payload = json.dumps({"session_id": SESSION_ID, "question": question}).encode()
    req = urllib.request.Request(
        BASE,
        data=payload,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read().decode())


def main() -> None:
    for q in QUERIES:
        print("=" * 72)
        print("Q:", q)
        try:
            data = ask(q)
        except Exception as exc:
            print("ERROR:", exc)
            continue
        answer = (data.get("answer") or "")[:400]
        sources = data.get("citations") or data.get("sources") or []
        print("answer:", answer.replace("\n", " ")[:400])
        print("sources_count:", len(sources))
        if sources:
            top = sources[0]
            print(
                "top_source:",
                top.get("document_id") or top.get("document_title"),
                "sim=",
                top.get("similarity_score"),
            )


if __name__ == "__main__":
    main()
