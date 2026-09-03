import pytest
from app.tools.page_fetcher import extract_publication_date, extract_readable_content, extract_headline_title
from app.services.web_search_service import format_web_context, is_article_url
from app.tools.web_search import WebSearchHit


def test_is_article_url():
    assert is_article_url("https://reuters.com/technology/openai-launches-new-model-2026-08-31/") is True
    assert is_article_url("https://thehackernews.com/2026/08/new-ai-threat.html") is True
    
    # Reject category, topic, tag, search, and newsroom pages
    assert is_article_url("https://openai.com/newsroom/") is False
    assert is_article_url("https://thehackernews.com/search/label/artificial%20intelligence") is False
    assert is_article_url("https://www.forbes.com/topics/ai-cybersecurity/") is False
    assert is_article_url("https://news.google.com/topics/CAAqJ...") is False
    assert is_article_url("https://openai.com/") is False


def test_extract_headline_title():
    html_og = '<html><head><meta property="og:title" content="OpenAI Unveils Enterprise Agent Framework"></head></html>'
    assert extract_headline_title(html_og) == "OpenAI Unveils Enterprise Agent Framework"

    html_title = '<html><head><title>Cybersecurity Firm Detects AI Exploit</title></head></html>'
    assert extract_headline_title(html_title) == "Cybersecurity Firm Detects AI Exploit"


def test_extract_publication_date():
    html_meta = '<html><head><meta property="article:published_time" content="2026-08-31T12:00:00Z"></head><body>Text</body></html>'
    date_val = extract_publication_date(html_meta)
    assert date_val == "2026-08-31"

    json_ld = '<html><head><script type="application/ld+json">{"datePublished": "2026-08-30"}</script></head><body>Text</body></html>'
    date_val_ld = extract_publication_date(json_ld)
    assert date_val_ld == "2026-08-30"


def test_format_web_context_with_published_date():
    hit = WebSearchHit(
        title="Test OpenAI News",
        url="https://example.com/news/openai-update",
        snippet="OpenAI releases new model update.",
        source="example.com",
        published_at="2026-08-31",
        content="OpenAI announced a new update today."
    )
    formatted = format_web_context([hit])
    assert "Published: 2026-08-31" in formatted
    assert "https://example.com/news/openai-update" in formatted
    assert "OpenAI announced a new update today." in formatted
