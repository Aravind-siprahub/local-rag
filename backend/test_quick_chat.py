"""Quick test script to invoke POST /api/chat directly with timeout handling."""
import time
import requests

url = "http://localhost:8000/api/chat"
payload = {"question": "What is Talk to My Data?"}

print(f"Sending request to {url}...")
start = time.time()
try:
    res = requests.post(url, json=payload, timeout=60)
    elapsed = time.time() - start
    print(f"\nStatus: {res.status_code} (took {elapsed:.2f}s)")
    if res.status_code == 200:
        data = res.json()
        print("\n=================== REAL ANSWER ===================")
        print(data.get("answer", ""))
        print("===================================================")
        print(f"\nCitations count: {len(data.get('citations', []))}")
        print(f"Processing time: {data.get('processing_time_ms')} ms")
    else:
        print("Error response:", res.text)
except Exception as err:
    print("Request failed:", err)
