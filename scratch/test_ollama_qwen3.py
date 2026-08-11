import asyncio
import json
import httpx

async def test_qwen3():
    url = "http://localhost:11434/api/chat"
    
    # Test 1: With think=False
    payload_no_think = {
        "model": "qwen3:4b",
        "messages": [
            {"role": "system", "content": "You are a concise general knowledge assistant. Answer directly."},
            {"role": "user", "content": "Is Earth the 2nd or 3rd planet from the Sun?"}
        ],
        "stream": False,
        "think": False,
        "options": {"num_predict": 512, "temperature": 0.0}
    }
    
    # Test 2: Without think parameter
    payload_normal = {
        "model": "qwen3:4b",
        "messages": [
            {"role": "system", "content": "You are a concise general knowledge assistant. Answer directly."},
            {"role": "user", "content": "Is Earth the 2nd or 3rd planet from the Sun?"}
        ],
        "stream": False,
        "options": {"num_predict": 512, "temperature": 0.0}
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        print("--- TEST 1: think=False ---")
        try:
            r1 = await client.post(url, json=payload_no_think)
            data1 = r1.json()
            print("KEYS in response:", data1.keys())
            print("MESSAGE keys:", data1.get("message", {}).keys())
            print("CONTENT:", repr(data1.get("message", {}).get("content")))
            print("THINKING:", repr(data1.get("message", {}).get("thinking")))
        except Exception as e:
            print("Test 1 Error:", e)

        print("\n--- TEST 2: Without think parameter ---")
        try:
            r2 = await client.post(url, json=payload_normal)
            data2 = r2.json()
            print("KEYS in response:", data2.keys())
            print("MESSAGE keys:", data2.get("message", {}).keys())
            print("CONTENT:", repr(data2.get("message", {}).get("content")))
            print("THINKING:", repr(data2.get("message", {}).get("thinking")))
        except Exception as e:
            print("Test 2 Error:", e)

if __name__ == "__main__":
    asyncio.run(test_qwen3())
