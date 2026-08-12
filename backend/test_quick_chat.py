"""Quick test script to invoke POST /api/chat directly with timeout handling."""
import time
import httpx

url = "http://localhost:8000/api/chat"

queries = [
    ("WEB", "when is Pongal"),
    ("WEB", "When is Deepawali in 2026?"),
    ("WEB", "latest Python release"),
    ("RAG", "What backend is used in Talk to My Data?"),
    ("RAG", "What is the Problem Statement?"),
    ("CALCULATOR", "50+50+100-20+20"),
]

print(f"Connecting to live backend at {url}...\n")
for cat, q in queries:
    print(f"==================================================")
    print(f"CATEGORY: {cat} | QUERY: '{q}'")
    print(f"==================================================")
    start = time.time()
    try:
        res = httpx.post(url, json={"question": q}, timeout=60)
        elapsed = time.time() - start
        print(f"Status: {res.status_code} (took {elapsed:.2f}s)")
        if res.status_code == 200:
            data = res.json()
            answer = data.get("answer", "")
            print("ANSWER:")
            print(answer)
            print(f"Model used: {data.get('model_used', 'N/A')}")
            print(f"Citations count: {len(data.get('citations', []))}")
        else:
            print("Error response:", res.text)
    except Exception as err:
        print("Request failed:", err)
    print("\n")
