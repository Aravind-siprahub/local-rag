"""Web search tool providers for Agent Router v1."""
from __future__ import annotations

import logging
import urllib.parse
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import httpx
from html.parser import HTMLParser

import time

logger = logging.getLogger(__name__)


class WebSearchError(Exception):
    """Raised when web search fails in a controlled way."""


@dataclass
class WebSearchHit:
    title: str
    url: str
    snippet: str
    source: str = "web"
    published_at: str | None = None
    content: str | None = None


@dataclass(frozen=True)
class WebSearchResult:
    query: str
    hits: list[WebSearchHit] = field(default_factory=list)
    provider: str = "unknown"

    def concise_answer(self) -> str:
        """Format hits into a short answer string for the chat response."""
        if not self.hits:
            return (
                "I could not find reliable web results for that question right now. "
                "Please try again shortly."
            )
        lines: list[str] = []
        for idx, hit in enumerate(self.hits[:5], start=1):
            snippet = hit.snippet.strip() or hit.title
            url_part = f" ({hit.url})" if hit.url else ""
            lines.append(f"{idx}. {hit.title}: {snippet}{url_part}")
        return "Here is what I found:\n" + "\n".join(lines)


def _clean_ddg_url(raw_url: str) -> str:
    """Un-redirect DuckDuckGo relative redirect URLs into clean target HTTP/HTTPS URLs."""
    if not raw_url:
        return ""
    url = raw_url.strip()
    if "/l/?" in url or "uddg=" in url:
        try:
            parsed = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
            uddg = parsed.get("uddg")
            if uddg and uddg[0]:
                return urllib.parse.unquote(uddg[0]).strip()
        except Exception:
            pass
        import re
        m = re.search(r"uddg=([^&]+)", url)
        if m:
            return urllib.parse.unquote(m.group(1)).strip()
    if url.startswith("//"):
        return "https:" + url
    if url.startswith("/") and not url.startswith("/l/?"):
        return "https://html.duckduckgo.com" + url
    return url


class DuckDuckGoHTMLParser(HTMLParser):
    """HTML parser for DuckDuckGo HTML search results."""

    def __init__(self):
        super().__init__()
        self.hits: list[WebSearchHit] = []
        self._current_title: list[str] = []
        self._current_snippet: list[str] = []
        self._current_url: str = ""
        self._in_title: bool = False
        self._in_snippet: bool = False

    def handle_starttag(self, tag: str, attrs: list) -> None:
        attrs_dict = dict(attrs)
        class_name = attrs_dict.get("class", "")
        classes = class_name.split() if class_name else []

        if tag == "a" and any(c in classes for c in ("result__a", "result__title", "result__url", "large")):
            self._in_title = True
            href = attrs_dict.get("href", "")
            if href:
                self._current_url = _clean_ddg_url(href)
        elif tag == "a" and not self._current_url and attrs_dict.get("href"):
            href = attrs_dict.get("href", "")
            if "uddg=" in href:
                self._current_url = _clean_ddg_url(href)
        elif any(c in classes for c in ("result__snippet", "result__body")):
            self._in_snippet = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._in_title:
            self._in_title = False
        elif self._in_snippet:
            self._in_snippet = False
            title = "".join(self._current_title).strip()
            snippet = "".join(self._current_snippet).strip()
            clean_url = _clean_ddg_url(self._current_url)
            if snippet and clean_url and clean_url.startswith("http") and "duckduckgo.com" not in clean_url:
                netloc = urllib.parse.urlparse(clean_url).netloc.replace("www.", "")
                self.hits.append(
                    WebSearchHit(
                        title=title or f"Result from {netloc}",
                        url=clean_url,
                        snippet=snippet,
                        source=netloc or "web",
                    )
                )
            self._current_title = []
            self._current_snippet = []
            self._current_url = ""

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._current_title.append(data)
        elif self._in_snippet:
            self._current_snippet.append(data)


class BackupSnippetParser(HTMLParser):
    """Fallback HTML parser that extracts result__snippet divs."""

    def __init__(self):
        super().__init__()
        self.snippets: list[str] = []
        self.in_snippet = False
        self.current_snippet: list[str] = []

    def handle_starttag(self, tag: str, attrs: list) -> None:
        attrs_dict = dict(attrs)
        class_name = attrs_dict.get("class", "")
        classes = class_name.split() if class_name else []
        if tag in ("a", "td", "div") and "result__snippet" in classes:
            self.in_snippet = True
            self.current_snippet = []

    def handle_endtag(self, tag: str) -> None:
        if self.in_snippet:
            text = "".join(self.current_snippet).strip()
            if text:
                self.snippets.append(text)
            self.in_snippet = False

    def handle_data(self, data: str) -> None:
        if self.in_snippet:
            self.current_snippet.append(data)


@runtime_checkable
class WebSearchProvider(Protocol):
    """Vendor-agnostic web search contract."""

    async def search(
        self,
        query: str,
        max_results: int = 5,
        recency_days: int | None = None,
        request_id: str | None = None,
    ) -> WebSearchResult: ...


class StubWebSearchProvider:
    """Deterministic provider for tests and offline use — no network calls."""

    async def search(
        self,
        query: str,
        max_results: int = 5,
        recency_days: int | None = None,
        request_id: str | None = None,
    ) -> WebSearchResult:
        q = (query or "").strip() or "empty"
        logger.info("[WEB SEARCH START] provider=stub query=%r request_id=%s", q, request_id or "N/A")
        result = WebSearchResult(
            query=q,
            provider="stub",
            hits=[
                WebSearchHit(
                    title="Stub web result",
                    url="https://example.com/stub",
                    snippet=(
                        f"Stub search result for: {q}. "
                        "Configure WEB_SEARCH_PROVIDER=duckduckgo for live results."
                    ),
                    source="example.com",
                )
            ][: max_results if max_results > 0 else 5],
        )
        logger.info("[WEB SEARCH RESULT] provider=stub status=200 result_count=1 latency_ms=0 request_id=%s", request_id or "N/A")
        return result


_WEATHER_CODE_MAP = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Foggy",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    71: "Slight snow fall",
    73: "Moderate snow fall",
    75: "Heavy snow fall",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    95: "Thunderstorm",
}


async def _fetch_open_meteo_weather(query: str, client: httpx.AsyncClient) -> WebSearchHit | None:
    """Fetch live weather data from Open-Meteo free API for weather queries."""
    import re

    q_lower = query.lower()

    # Extract location name (e.g. "Bangalore", "Bengaluru", "Tokyo", "London")
    cleaned_q = re.sub(r"\b(what|is|the|weather|today|in|for|at|current|now|temperature|forecast|degree|degrees|climate|rain)\b", "", q_lower, flags=re.I).strip()
    location = cleaned_q or "Bangalore"

    try:
        # Fetch multiple results and prefer Indian cities
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={urllib.parse.quote(location)}&count=10&language=en&format=json"
        geo_res = await client.get(geo_url, timeout=3.0)
        if geo_res.status_code != 200:
            return None
        geo_data = geo_res.json()
        results = geo_data.get("results")
        if not results:
            return None

        # Prefer India if query mentions Indian cities or India context
        indian_city_aliases = {
            "bangalore": "Bengaluru", "bengaluru": "Bengaluru", "bombay": "Mumbai",
            "calcutta": "Kolkata", "madras": "Chennai", "delhi": "Delhi",
            "hyderabad": "Hyderabad", "pune": "Pune", "ahmedabad": "Ahmedabad",
        }
        loc_lower = location.lower().strip()
        is_india_query = loc_lower in indian_city_aliases or "india" in q_lower

        loc_item = results[0]  # default
        if is_india_query:
            # Try to find a result where country is India
            india_match = next(
                (r for r in results if (r.get("country_code") or "").upper() == "IN"),
                None
            )
            if india_match:
                loc_item = india_match

        lat = loc_item.get("latitude")
        lon = loc_item.get("longitude")
        city_name = loc_item.get("name") or location
        country = loc_item.get("country") or ""
        if is_india_query and (not country or "india" not in country.lower()):
            city_name = indian_city_aliases.get(loc_lower, city_name)
            country = "India"

        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m"
        w_res = await client.get(weather_url, timeout=3.0)
        if w_res.status_code != 200:
            return None
        w_data = w_res.json()
        curr = w_data.get("current", {})
        temp_c = curr.get("temperature_2m")
        feels_c = curr.get("apparent_temperature")
        humidity = curr.get("relative_humidity_2m")
        precip = curr.get("precipitation")
        w_code = curr.get("weather_code", 0)
        wind = curr.get("wind_speed_10m")

        cond = _WEATHER_CODE_MAP.get(w_code, "Partly cloudy")
        temp_f = round(temp_c * 9 / 5 + 32, 1) if temp_c is not None else None

        location_label = f"{city_name}, {country}" if country else city_name
        snippet = (
            f"Live Weather for {location_label}: Temperature: {temp_c}°C ({temp_f}°F), "
            f"Feels like: {feels_c}°C, Condition: {cond}, Humidity: {humidity}%, "
            f"Wind Speed: {wind} km/h, Precipitation: {precip} mm."
        )

        return WebSearchHit(
            title=f"Live Weather Forecast for {location_label}",
            url=f"https://open-meteo.com/en/docs#latitude={lat}&longitude={lon}",
            snippet=snippet,
            source="open-meteo.com",
        )
    except Exception as exc:
        logger.debug("[OPEN-METEO WEATHER] Failed to fetch weather for %r: %s", location, exc)
        return None


async def _fetch_crypto_live_price(query: str, client: httpx.AsyncClient) -> WebSearchHit | None:
    """Fetch real-time cryptocurrency ticker price and 24h stats from Binance / CoinCap public API."""
    import re
    from datetime import datetime
    q_lower = query.lower()
    symbol_map = {
        "bitcoin": "BTCUSDT",
        "btc": "BTCUSDT",
        "ethereum": "ETHUSDT",
        "eth": "ETHUSDT",
        "solana": "SOLUSDT",
        "sol": "SOLUSDT",
        "dogecoin": "DOGEUSDT",
        "doge": "DOGEUSDT",
        "xrp": "XRPUSDT",
        "ripple": "XRPUSDT",
        "cardano": "ADAUSDT",
        "ada": "ADAUSDT",
        "bnb": "BNBUSDT",
    }
    target_symbol = None
    target_name = "Bitcoin"
    for name, sym in symbol_map.items():
        if re.search(rf"\b{name}\b", q_lower):
            target_symbol = sym
            target_name = name.capitalize()
            break

    if not target_symbol and "crypto" in q_lower:
        target_symbol = "BTCUSDT"
        target_name = "Bitcoin"

    if not target_symbol:
        return None

    try:
        resp = await client.get(f"https://api.binance.com/api/v3/ticker/24hr?symbol={target_symbol}", timeout=4.0)
        if resp.status_code == 200:
            data = resp.json()
            price = float(data.get("lastPrice", 0))
            change = float(data.get("priceChangePercent", 0))
            high = float(data.get("highPrice", 0))
            low = float(data.get("lowPrice", 0))
            vol = float(data.get("volume", 0))
            sign = "+" if change >= 0 else ""
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
            snippet = (
                f"Live Cryptocurrency Market Data ({target_name}/USD): Current Price: ${price:,.2f} USD "
                f"(24h Change: {sign}{change:.2f}%, 24h High: ${high:,.2f}, 24h Low: ${low:,.2f}, Volume: {vol:,.2f} {target_symbol[:3]}). "
                f"Observation Timestamp: {now_str}. Source: Binance Global Exchange live ticker."
            )
            return WebSearchHit(
                title=f"{target_name} Live Price Today - Binance & CoinMarketCap",
                url=f"https://www.binance.com/en/price/{target_name.lower()}",
                snippet=snippet,
                source="binance.com",
                published_at=datetime.now().strftime("%Y-%m-%d"),
                content=snippet,
            )
    except Exception as exc:
        logger.warning("[CRYPTO PRICE ENRICHMENT] Fetch failed: %s", exc)

    return None


class DuckDuckGoWebSearchProvider:
    """DuckDuckGo Instant Answer API + HTML Search fallback — no API key required."""

    def __init__(
        self,
        *,
        timeout_seconds: float = 8.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self._client = client
        self._owns_client = client is None

    async def search(
        self,
        query: str,
        max_results: int = 5,
        recency_days: int | None = None,
        request_id: str | None = None,
    ) -> WebSearchResult:
        q = (query or "").strip().strip('"').strip("'").strip()
        if not q:
            raise WebSearchError("Search query must not be empty.")

        req_id = request_id or "N/A"
        start_mono = time.monotonic()
        logger.info("[WEB SEARCH START] provider=duckduckgo query=%r max_results=%d request_id=%s", q, max_results, req_id)

        # enrichment_hits are ALWAYS preserved — they are never overwritten by DDG results
        enrichment_hits: list[WebSearchHit] = []
        hits: list[WebSearchHit] = []
        http_status: int | None = None

        # Direct official source enrichment for Python queries
        q_lower = q.lower()
        if "python" in q_lower and any(k in q_lower for k in ("version", "release", "latest", "stable", "official")):
            try:
                client = await self._get_client()
                py_resp = await client.get("https://www.python.org/downloads/", headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
                if py_resp.status_code == 200:
                    import re
                    # Extract releases from official table: Python (3.x.y) and Release Date
                    releases = re.findall(r'Python (3\.\d+\.\d+)</a></span>\s*<span class="release-date">([^<]+)</span>', py_resp.text)
                    if not releases:
                        btn_matches = re.findall(r'Download Python (3\.\d+\.\d+)', py_resp.text)
                        releases = [(btn, "Latest") for btn in btn_matches]
                    
                    if releases:
                        latest_ver, latest_date = releases[0]
                        # Capture top active feature series and maintenance releases
                        top_releases_str = ", ".join(f"Python {v} ({d})" for v, d in releases[:5])
                        py_hit = WebSearchHit(
                            title=f"Python Release Python {latest_ver} - Python.org",
                            url="https://www.python.org/downloads/",
                            snippet=f"Official Python Website (python.org): The latest Python release as of today is Python {latest_ver}, released on {latest_date}. Latest releases in the Python 3 series include: {top_releases_str}. Download Python {latest_ver} directly from python.org.",
                            source="python.org",
                            published_at=latest_date,
                            content=f"Official Python Website (python.org): The latest Python release as of today is Python {latest_ver}, released on {latest_date}. Latest releases in the Python 3 series include: {top_releases_str}. Download Python {latest_ver} directly from python.org.",
                        )
                        enrichment_hits.append(py_hit)
                        logger.info("[WEB SEARCH ENRICHMENT] Added official python.org hit for version=%s date=%s", latest_ver, latest_date)
            except Exception as py_exc:
                logger.warning("[WEB SEARCH ENRICHMENT] Python fetch failed: %s", py_exc)

        # Real-time Cryptocurrency / Bitcoin Price Enrichment
        if any(c in q_lower for c in ("bitcoin", "btc", "ethereum", "eth", "solana", "crypto", "price of bitcoin")):
            try:
                client = await self._get_client()
                crypto_hit = await _fetch_crypto_live_price(q, client)
                if crypto_hit:
                    enrichment_hits.append(crypto_hit)
                    logger.info("[WEB SEARCH ENRICHMENT] Added Live Crypto Price Hit: %s", crypto_hit.snippet)
            except Exception as c_exc:
                logger.warning("[WEB SEARCH ENRICHMENT] Crypto price fetch failed: %s", c_exc)

        # Weather query enrichment using free Open-Meteo API
        if any(w in q_lower for w in ("weather", "temperature", "forecast", "climate", "rain", "degree")):
            try:
                client = await self._get_client()
                w_hit = await _fetch_open_meteo_weather(q, client)
                if w_hit:
                    enrichment_hits.append(w_hit)
                    logger.info("[WEB SEARCH ENRICHMENT] Added Open-Meteo live weather hit: %s", w_hit.snippet)
            except Exception as w_exc:
                logger.warning("[WEB SEARCH ENRICHMENT] Weather fetch failed: %s", w_exc)

        # 1. Try Instant Answer API (fast, structured)
        try:
            client = await self._get_client()
            params = {
                "q": q,
                "format": "json",
                "no_html": "1",
                "skip_disambig": "1",
            }
            response = await client.get("https://api.duckduckgo.com/", params=params)
            http_status = response.status_code
            if response.status_code == 200:
                payload = response.json()
                hits = _hits_from_duckduckgo(payload)
        except httpx.TimeoutException:
            logger.warning("[WEB SEARCH] duckduckgo instant answer timeout")
        except httpx.HTTPError as exc:
            logger.warning("[WEB SEARCH] duckduckgo instant answer http error: %s", exc)
        except Exception as exc:
            logger.warning("[WEB SEARCH] duckduckgo instant answer unexpected error: %s", exc)

        logger.info("[WEB SEARCH API RESULT] request_id=%s api_hits=%d", req_id, len(hits))

        # 2. Fallback: HTML Scrape if DDG Instant Answer returned 0 hits (enrichment_hits are separate)
        if not hits:
            logger.info("[WEB SEARCH FALLBACK START] request_id=%s url=https://html.duckduckgo.com/html/", req_id)
            try:
                client = await self._get_client()
                headers = {
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"
                    )
                }
                response = await client.post(
                    "https://html.duckduckgo.com/html/",
                    data={"q": q},
                    headers=headers,
                )
                http_status = response.status_code
                if response.status_code == 200:
                    parser = DuckDuckGoHTMLParser()
                    parser.feed(response.text)
                    valid_html_hits = [
                        h for h in parser.hits
                        if h.url and h.url.startswith("http") and "duckduckgo.com" not in h.url
                    ]
                    hits.extend(valid_html_hits)
                    if not hits:
                        import re
                        raw_matches = re.findall(r'href=["\'](/l/\?uddg=[^"\']+|https?://[^"\']+)', response.text)
                        regex_seen: set[str] = set()
                        for raw_u in raw_matches:
                            cleaned = _clean_ddg_url(raw_u)
                            if cleaned and cleaned.startswith("http") and "duckduckgo.com" not in cleaned and cleaned not in regex_seen:
                                regex_seen.add(cleaned)
                                netloc = urllib.parse.urlparse(cleaned).netloc.replace("www.", "")
                                hits.append(
                                    WebSearchHit(
                                        title=f"Search Result ({netloc})",
                                        url=cleaned,
                                        snippet=f"Web search result from {netloc} for query '{q}'",
                                        source=netloc or "web",
                                    )
                                )
                                if len(hits) >= max_results:
                                    break
                logger.info("[WEB SEARCH FALLBACK RESULT] request_id=%s status=%s parsed_count=%d", req_id, http_status, len(hits))
            except httpx.TimeoutException as exc:
                latency_ms = int((time.monotonic() - start_mono) * 1000)
                logger.warning(
                    "[WEB SEARCH RESULT] provider=duckduckgo status=%s result_count=0 latency_ms=%d request_id=%s",
                    http_status or "timeout",
                    latency_ms,
                    req_id,
                )
                if not enrichment_hits:
                    raise WebSearchError("Web search timed out. Please try again.") from exc
                logger.info("[WEB SEARCH FALLBACK] DDG timed out but enrichment_hits=%d — continuing", len(enrichment_hits))
            except httpx.HTTPError as exc:
                latency_ms = int((time.monotonic() - start_mono) * 1000)
                logger.warning(
                    "[WEB SEARCH RESULT] provider=duckduckgo status=%s result_count=0 latency_ms=%d request_id=%s",
                    http_status or "http_error",
                    latency_ms,
                    req_id,
                )
                if not enrichment_hits:
                    raise WebSearchError("Web search is temporarily unavailable.") from exc
                logger.info("[WEB SEARCH FALLBACK] DDG http error but enrichment_hits=%d — continuing", len(enrichment_hits))
            except Exception as exc:
                latency_ms = int((time.monotonic() - start_mono) * 1000)
                logger.exception(
                    "[WEB SEARCH RESULT] provider=duckduckgo status=%s result_count=0 latency_ms=%d request_id=%s",
                    http_status or "error",
                    latency_ms,
                    req_id,
                )
                if not enrichment_hits:
                    raise WebSearchError("Web search execution failed. Please try again.") from exc
                logger.info("[WEB SEARCH FALLBACK] DDG search exception but enrichment_hits=%d — continuing", len(enrichment_hits))

        # Merge enrichment hits at the FRONT so official sources (e.g. python.org) are always present
        # enrichment_hits were stored separately and survive all DDG API/fallback overwrites
        all_hits = enrichment_hits + hits

        # Deduplicate hits by clean URL
        seen_urls: set[str] = set()
        unique_hits: list[WebSearchHit] = []

        for h in all_hits:
            url_str = (h.url or "").strip()
            if not url_str or not url_str.startswith("http"):
                continue
            clean_u = url_str.rstrip("/")
            if clean_u in seen_urls:
                continue
            seen_urls.add(clean_u)

            source_name = h.source
            if not source_name:
                try:
                    netloc = urllib.parse.urlparse(h.url).netloc
                    if netloc:
                        source_name = netloc.replace("www.", "")
                except Exception:
                    pass

            unique_hits.append(
                WebSearchHit(
                    title=h.title or "Web Result",
                    url=h.url,
                    snippet=h.snippet,
                    source=source_name or "web",
                    published_at=h.published_at,
                    content=h.content,
                )
            )

        # Prioritize GitHub results if searching GitHub
        if "github" in q.lower():
            gh_hits = [h for h in unique_hits if "github.com" in (h.url or "").lower() or "github" in (h.source or "").lower()]
            other_hits = [h for h in unique_hits if h not in gh_hits]
            unique_hits = gh_hits + other_hits

        latency_ms = int((time.monotonic() - start_mono) * 1000)
        
        # Structured debug logging
        logger.info("WEB_SEARCH_DEBUG: route=WEB")
        logger.info("WEB_SEARCH_DEBUG: provider=duckduckgo")
        logger.info("WEB_SEARCH_DEBUG: endpoint=https://html.duckduckgo.com/html/")
        logger.info("WEB_SEARCH_DEBUG: query=%r", q)
        logger.info("WEB_SEARCH_DEBUG: http_status=%s", http_status or 200)
        logger.info("WEB_SEARCH_DEBUG: result_count=%d", len(unique_hits))
        for idx_h, h_item in enumerate(unique_hits[:max_results], 1):
            logger.info("WEB_SEARCH_DEBUG: result_%d_url=%s", idx_h, h_item.url)
        logger.info(
            "[WEB SEARCH RESULT] provider=duckduckgo status=%s result_count=%d latency_ms=%d request_id=%s",
            http_status if http_status is not None else 200,
            len(unique_hits),
            latency_ms,
            req_id,
        )

        if not unique_hits:
            raise WebSearchError("Web search yielded no results. Please try again.")

        limit = max_results if max_results and max_results > 0 else 5
        final_hits = unique_hits[:limit]

        logger.info("[WEB SEARCH] query=%r", q)
        logger.info("[WEB SEARCH RESULTS] provider=\"duckduckgo\" result_count=%d", len(final_hits))
        for hit in final_hits:
            logger.info("[WEB SEARCH RESULT] title=%r url=%r", hit.title, hit.url)
        logger.info("[WEB SEARCH CONTEXT] results_passed_to_llm=%d", len(final_hits))
        logger.info("[WEB SEARCH COMPLETE] success=true")

        return WebSearchResult(query=q, hits=final_hits, provider="duckduckgo")

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout_seconds),
                headers={"User-Agent": "local-rag-agent-router/1.0"},
            )
        return self._client

    async def close(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None


def _hits_from_duckduckgo(payload: dict) -> list[WebSearchHit]:
    hits: list[WebSearchHit] = []

    abstract = (payload.get("AbstractText") or "").strip()
    abstract_url = (payload.get("AbstractURL") or "").strip()
    heading = (payload.get("Heading") or payload.get("AbstractSource") or "DuckDuckGo").strip()
    if abstract:
        hits.append(
            WebSearchHit(
                title=heading or "Result",
                url=abstract_url,
                snippet=abstract,
            )
        )

    answer = (payload.get("Answer") or "").strip()
    if answer and answer != abstract:
        hits.append(
            WebSearchHit(
                title=heading or "Answer",
                url=abstract_url,
                snippet=answer,
            )
        )

    for topic in payload.get("RelatedTopics") or []:
        if len(hits) >= 5:
            break
        if not isinstance(topic, dict):
            continue
        if "Topics" in topic:
            for nested in topic.get("Topics") or []:
                if len(hits) >= 5:
                    break
                hit = _hit_from_related(nested)
                if hit:
                    hits.append(hit)
            continue
        hit = _hit_from_related(topic)
        if hit:
            hits.append(hit)

    return hits


def _hit_from_related(topic: dict) -> WebSearchHit | None:
    if not isinstance(topic, dict):
        return None
    text = (topic.get("Text") or "").strip()
    url = (topic.get("FirstURL") or "").strip()
    if not text:
        return None
    title = text.split(" - ", 1)[0][:120]
    return WebSearchHit(title=title, url=url, snippet=text)


class SearXNGWebSearchProvider:
    """Self-hosted SearXNG Search Engine Provider — 100% Free & Open Source."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        timeout_seconds: float = 10.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        from app.core.config import get_settings
        settings = get_settings()
        self.base_url = (base_url or getattr(settings, "SEARXNG_URL", "http://localhost:8080")).rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._client = client
        self._owns_client = client is None

    async def search(
        self,
        query: str,
        max_results: int = 5,
        recency_days: int | None = None,
        request_id: str | None = None,
    ) -> WebSearchResult:
        q = (query or "").strip()
        if not q:
            raise WebSearchError("Search query must not be empty.")

        req_id = request_id or "N/A"
        start_mono = time.monotonic()
        logger.info("[WEB SEARCH START] provider=searxng url=%s query=%r max_results=%d request_id=%s", self.base_url, q, max_results, req_id)

        hits: list[WebSearchHit] = []
        http_status: int | None = None

        try:
            client = await self._get_client()
            search_endpoint = f"{self.base_url}/search"
            params = {
                "q": q,
                "format": "json",
            }
            response = await client.get(search_endpoint, params=params)
            http_status = response.status_code

            if response.status_code == 200:
                payload = response.json()
                raw_results = payload.get("results", [])
                for item in raw_results:
                    title = (item.get("title") or "Search Result").strip()
                    url = (item.get("url") or "").strip()
                    snippet = (item.get("content") or item.get("snippet") or "").strip()
                    published_at = item.get("publishedDate") or item.get("published_date")
                    engine = item.get("engine") or "searxng"

                    if url and snippet:
                        netloc = urllib.parse.urlparse(url).netloc.replace("www.", "")
                        hits.append(
                            WebSearchHit(
                                title=title,
                                url=url,
                                snippet=snippet,
                                source=netloc or engine,
                                published_at=published_at,
                            )
                        )
                    if len(hits) >= max_results:
                        break
        except httpx.TimeoutException as exc:
            latency_ms = int((time.monotonic() - start_mono) * 1000)
            logger.warning("[WEB SEARCH RESULT] provider=searxng status=timeout latency_ms=%d request_id=%s", latency_ms, req_id)
            raise WebSearchError("SearXNG search timed out.") from exc
        except httpx.HTTPError as exc:
            latency_ms = int((time.monotonic() - start_mono) * 1000)
            logger.warning("[WEB SEARCH RESULT] provider=searxng status=http_error latency_ms=%d request_id=%s error=%s", latency_ms, req_id, exc)
            raise WebSearchError(f"SearXNG service HTTP error: {exc}") from exc
        except Exception as exc:
            latency_ms = int((time.monotonic() - start_mono) * 1000)
            logger.exception("[WEB SEARCH RESULT] provider=searxng status=error latency_ms=%d request_id=%s error=%s", latency_ms, req_id, exc)
            raise WebSearchError(f"SearXNG search failed: {exc}") from exc

        latency_ms = int((time.monotonic() - start_mono) * 1000)
        
        logger.info("WEB_SEARCH_DEBUG: route=WEB")
        logger.info("WEB_SEARCH_DEBUG: provider=searxng")
        logger.info("WEB_SEARCH_DEBUG: endpoint=%s/search", self.base_url)
        logger.info("WEB_SEARCH_DEBUG: query=%r", q)
        logger.info("WEB_SEARCH_DEBUG: http_status=%s", http_status or 200)
        logger.info("WEB_SEARCH_DEBUG: result_count=%d", len(hits))
        for idx_h, h_item in enumerate(hits[:max_results], 1):
            logger.info("WEB_SEARCH_DEBUG: result_%d_url=%s", idx_h, h_item.url)

        logger.info("[WEB SEARCH RESULT] provider=searxng status=%s result_count=%d latency_ms=%d request_id=%s", http_status or 200, len(hits), latency_ms, req_id)

        if not hits:
            logger.info("[WEB SEARCH FALLBACK] SearXNG returned 0 hits for query=%r; trying DuckDuckGo fallback", q)
            ddg = DuckDuckGoWebSearchProvider(timeout_seconds=self.timeout_seconds, client=self._client)
            return await ddg.search(q, max_results=max_results, recency_days=recency_days, request_id=request_id)

        return WebSearchResult(query=q, hits=hits[:max_results], provider="searxng")

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout_seconds),
                headers={"User-Agent": "local-rag-searxng/1.0"},
            )
        return self._client

    async def close(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None


def get_web_search_provider(
    *,
    provider_name: str | None = None,
    timeout_seconds: float | None = None,
) -> WebSearchProvider:
    """Factory: ``searxng`` (default), ``duckduckgo``, or ``stub``."""
    from app.core.config import get_settings

    settings = get_settings()
    name = (provider_name or getattr(settings, "WEB_SEARCH_PROVIDER", None) or "searxng")
    name = str(name).strip().lower()
    timeout = (
        timeout_seconds
        if timeout_seconds is not None
        else float(getattr(settings, "WEB_SEARCH_TIMEOUT_SECONDS", 10.0))
    )

    if name == "stub":
        logger.info("[WEB SEARCH] using provider=stub")
        return StubWebSearchProvider()
    if name == "searxng":
        logger.info("[WEB SEARCH] using provider=searxng (URL: %s)", getattr(settings, "SEARXNG_URL", "http://localhost:8080"))
        return SearXNGWebSearchProvider(timeout_seconds=timeout)
    if name == "duckduckgo":
        logger.info("[WEB SEARCH] using provider=duckduckgo")
        return DuckDuckGoWebSearchProvider(timeout_seconds=timeout)

    logger.warning("[WEB SEARCH] unknown provider=%s; falling back to searxng", name)
    return SearXNGWebSearchProvider(timeout_seconds=timeout)

