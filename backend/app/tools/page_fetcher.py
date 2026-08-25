"""SSRF-safe Page Fetcher and HTML Content Extractor for Real-Time Web Search."""
from __future__ import annotations

import ipaddress
import logging
import re
import socket
import urllib.parse
from html.parser import HTMLParser
from typing import Any

import httpx

logger = logging.getLogger(__name__)


def is_safe_url(url: str) -> tuple[bool, str]:
    """Validate URL scheme and resolve target host IP to prevent SSRF attacks.

    Blocks:
    - Non http/https schemes
    - Loopback addresses (127.0.0.0/8, ::1)
    - Private IP networks (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16)
    - Link-local addresses (169.254.0.0/16, fe80::/10) - e.g. AWS metadata 169.254.169.254
    - Unspecified / Multicast / Reserved addresses
    - Localhost hostnames
    """
    if not url or not isinstance(url, str):
        return False, "URL must be a non-empty string"

    url_clean = url.strip()
    try:
        parsed = urllib.parse.urlparse(url_clean)
    except Exception as exc:
        return False, f"Malformed URL: {exc}"

    if parsed.scheme.lower() not in ("http", "https"):
        return False, f"Unsupported scheme: {parsed.scheme}"

    hostname = parsed.hostname
    if not hostname:
        return False, "Missing hostname in URL"

    hostname_lower = hostname.lower()
    if hostname_lower in ("localhost", "localhost.localdomain", "local"):
        return False, "Localhost target blocked"

    # Try resolving hostname to IP addresses via DNS
    try:
        addr_info = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
    except socket.gaierror as err:
        return False, f"DNS resolution failed for hostname {hostname}: {err}"
    except Exception as exc:
        return False, f"IP resolution error for {hostname}: {exc}"

    if not addr_info:
        return False, f"No IP addresses found for hostname {hostname}"

    for family, _, _, _, sockaddr in addr_info:
        ip_str = sockaddr[0]
        try:
            ip_obj = ipaddress.ip_address(ip_str)
        except ValueError:
            return False, f"Invalid IP address resolved: {ip_str}"

        if (
            ip_obj.is_loopback
            or ip_obj.is_private
            or ip_obj.is_link_local
            or ip_obj.is_multicast
            or ip_obj.is_reserved
            or ip_obj.is_unspecified
        ):
            logger.warning("[SSRF BLOCKED] Host %s resolved to private/forbidden IP %s", hostname, ip_str)
            return False, f"Access to private or restricted IP ({ip_str}) is forbidden"

    return True, "URL is safe"


class HTMLContentCleaner(HTMLParser):
    """HTML Parser that strips boilerplate elements (scripts, style, nav, headers, footers)

    and extracts readable plain text content.
    """

    _BOILERPLATE_TAGS = {
        "script", "style", "nav", "header", "footer", "aside", "form",
        "svg", "iframe", "noscript", "button", "head", "style"
    }

    def __init__(self) -> None:
        super().__init__()
        self._text_chunks: list[str] = []
        self._ignore_stack: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag_lower = tag.lower()
        if tag_lower in self._BOILERPLATE_TAGS:
            self._ignore_stack.append(tag_lower)

    def handle_endtag(self, tag: str) -> None:
        tag_lower = tag.lower()
        if self._ignore_stack and self._ignore_stack[-1] == tag_lower:
            self._ignore_stack.pop()

    def handle_data(self, data: str) -> None:
        if not self._ignore_stack:
            text = data.strip()
            if text:
                self._text_chunks.append(text)

    def get_clean_text(self) -> str:
        raw_text = " ".join(self._text_chunks)
        # Collapse multiple spaces and line breaks into clean paragraph text
        cleaned = re.sub(r"\s+", " ", raw_text).strip()
        return cleaned


def extract_readable_content(html: str, max_chars: int = 2500) -> str:
    """Extract readable text content from raw HTML string."""
    if not html or not html.strip():
        return ""

    try:
        cleaner = HTMLContentCleaner()
        cleaner.feed(html)
        cleaner.close()
        text = cleaner.get_clean_text()
    except Exception as exc:
        logger.debug("[CONTENT EXTRACTOR] Parser error: %s. Using regex fallback.", exc)
        # Regex fallback: strip scripts/styles and html tags
        no_scripts = re.sub(r"<(script|style|nav|header|footer).*?>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)
        no_tags = re.sub(r"<[^>]+>", " ", no_scripts)
        text = re.sub(r"\s+", " ", no_tags).strip()

    if len(text) > max_chars:
        return text[:max_chars].rsplit(" ", 1)[0] + "..."
    return text


class PageFetcher:
    """Secure, resilient page fetcher with SSRF protection and content extraction."""

    def __init__(
        self,
        *,
        timeout_seconds: float = 5.0,
        max_bytes: int = 500_000,
        user_agent: str | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.max_bytes = max_bytes
        self.user_agent = user_agent or (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        )
        self._client = client
        self._owns_client = client is None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout_seconds),
                headers={"User-Agent": self.user_agent},
                follow_redirects=True,
            )
        return self._client

    async def fetch_and_extract(
        self,
        url: str,
        *,
        max_chars: int = 2500,
        request_id: str | None = None,
    ) -> str | None:
        """Fetch URL with SSRF protection, size limits, and extract clean text."""
        req_id = request_id or "N/A"
        is_safe, reason = is_safe_url(url)
        if not is_safe:
            logger.warning("[PAGE FETCH BLOCKED] url=%r reason=%s request_id=%s", url, reason, req_id)
            return None

        try:
            client = await self._get_client()
            logger.info("[PAGE FETCH START] url=%r request_id=%s", url, req_id)
            response = await client.get(url)
            
            if response.status_code != 200:
                logger.warning("[PAGE FETCH HTTP ERROR] url=%r status=%d request_id=%s", url, response.status_code, req_id)
                return None

            content_type = response.headers.get("content-type", "").lower()
            if content_type and "text/html" not in content_type and "text/plain" not in content_type:
                logger.info("[PAGE FETCH SKIP] url=%r non-text content_type=%s request_id=%s", url, content_type, req_id)
                return None

            raw_body = response.text[: self.max_bytes]
            extracted_text = extract_readable_content(raw_body, max_chars=max_chars)
            logger.info("[PAGE FETCH SUCCESS] url=%r extracted_len=%d request_id=%s", url, len(extracted_text), req_id)
            return extracted_text if extracted_text else None

        except httpx.TimeoutException:
            logger.warning("[PAGE FETCH TIMEOUT] url=%r timeout=%.1fs request_id=%s", url, self.timeout_seconds, req_id)
            return None
        except httpx.HTTPError as err:
            logger.warning("[PAGE FETCH HTTP FAILED] url=%r error=%s request_id=%s", url, err, req_id)
            return None
        except Exception as exc:
            logger.exception("[PAGE FETCH UNEXPECTED ERROR] url=%r error=%s request_id=%s", url, exc, req_id)
            return None

    async def close(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None
