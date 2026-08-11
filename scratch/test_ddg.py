import asyncio
import httpx
from bs4 import BeautifulSoup

async def test_ddg_api():
    q = "When is Good Friday in 2026?"
    
    # 1. Test Instant Answer API
    async with httpx.AsyncClient() as client:
        r1 = await client.get("https://api.duckduckgo.com/", params={"q": q, "format": "json"})
        print("=== Instant Answer API ===")
        print("Status:", r1.status_code)
        data = r1.json()
        print("AbstractText:", data.get("AbstractText"))
        print("Answer:", data.get("Answer"))
        print("RelatedTopics count:", len(data.get("RelatedTopics", [])))
        
    # 2. Test HTML endpoint (https://html.duckduckgo.com/html/)
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
    async with httpx.AsyncClient(headers=headers, follow_redirects=True) as client:
        r2 = await client.post("https://html.duckduckgo.com/html/", data={"q": q})
        print("\n=== DuckDuckGo HTML endpoint ===")
        print("Status:", r2.status_code)
        soup = BeautifulSoup(r2.text, "html.parser")
        results = soup.find_all("a", class_="result__snippet")
        for i, res in enumerate(results[:5], 1):
            print(f"Hit {i}:", res.get_text(strip=True))

if __name__ == "__main__":
    asyncio.run(test_ddg_api())
