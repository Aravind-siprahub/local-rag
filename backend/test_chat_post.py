import asyncio
import httpx
import json

async def test_live_chat():
    url = "http://127.0.0.1:8000/api/chat"
    payload = {
        "question": "What is inside PRD_Talk_to_My_Data.docx?"
    }

    print("=== SENDING POST REQUEST TO /api/chat ===")
    print(f"URL: {url}")
    print(f"Payload: {json.dumps(payload, indent=2)}")

    async with httpx.AsyncClient(timeout=300.0) as client:
        res = await client.post(url, json=payload)
        print(f"\nResponse Status: {res.status_code}")
        data = res.json()
        print("\n=== FINAL RESPONSE RETURNED TO FRONTEND ===")
        print(json.dumps(data, indent=2))

if __name__ == "__main__":
    asyncio.run(test_live_chat())
