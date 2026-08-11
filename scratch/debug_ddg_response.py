import asyncio
import httpx
import urllib.parse
import re
import html

async def debug_search():
    q = "when is good friday in 2026"
    log = []
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    
    async with httpx.AsyncClient(follow_redirects=True) as client:
        # 1. Test POST html.duckduckgo.com
        try:
            r1 = await client.post("https://html.duckduckgo.com/html/", data={"q": q}, headers=headers)
            log.append(f"=== POST html.duckduckgo.com ===")
            log.append(f"Status: {r1.status_code}")
            log.append(f"Body length: {len(r1.text)}")
            log.append(f"Body snippet:\n{r1.text[:1000]}\n")
        except Exception as e:
            log.append(f"POST html error: {e}")

        # 2. Test GET html.duckduckgo.com
        try:
            r2 = await client.get(f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(q)}", headers=headers)
            log.append(f"=== GET html.duckduckgo.com ===")
            log.append(f"Status: {r2.status_code}")
            log.append(f"Body length: {len(r2.text)}")
            log.append(f"Body snippet:\n{r2.text[:1000]}\n")
        except Exception as e:
            log.append(f"GET html error: {e}")

        # 3. Test POST lite.duckduckgo.com
        try:
            r3 = await client.post("https://lite.duckduckgo.com/lite/", data={"q": q}, headers=headers)
            log.append(f"=== POST lite.duckduckgo.com ===")
            log.append(f"Status: {r3.status_code}")
            log.append(f"Body length: {len(r3.text)}")
            log.append(f"Body snippet:\n{r3.text[:1000]}\n")
        except Exception as e:
            log.append(f"POST lite error: {e}")
            
    with open("scratch/ddg_debug_output.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(log))
    print("Debug output saved to scratch/ddg_debug_output.txt")

if __name__ == "__main__":
    asyncio.run(debug_search())
