"""Query understanding, entity extraction, and attribute classification layer.

Analyzes raw user queries to extract:
1. Target project/entity (e.g., "Talk to My Data", "SipraOne", "AIRIS")
2. Requested attributes (e.g., "frontend", "backend", "port", "database")
3. Attribute category ("technology", "configuration", "deployment", "general")
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class AttributeCategory(str, Enum):
    TECHNOLOGY = "technology"
    CONFIGURATION = "configuration"
    DEPLOYMENT = "deployment"
    GENERAL = "general"


@dataclass(frozen=True)
class QueryIntent:
    """Structured intent representation extracted from user query."""

    raw_query: str
    normalized_query: str
    entity: str | None = None
    attributes: set[str] = field(default_factory=set)
    category: AttributeCategory = AttributeCategory.GENERAL


def extract_query_intent(raw_query: str) -> QueryIntent:
    """Parse user query into a structured QueryIntent."""
    q_clean = (raw_query or "").strip()
    if not q_clean:
        return QueryIntent(raw_query="", normalized_query="")

    q_lower = q_clean.lower()
    # Normalize outer quotes and common typos
    q_norm = re.sub(r"^[\"'\s]+|[\"'\s]+$", "", q_lower)
    q_norm = re.sub(r"\bfronted\b", "frontend", q_norm)
    q_norm = re.sub(r"\b(?:tell|please|me|show|explain|give)\b\s*$", "", q_norm).strip()

    # 1. Entity / Project Detection
    entity: str | None = None
    if "talk to my data" in q_norm or "talktomydata" in q_norm:
        entity = "Talk to My Data"
    elif "sipraone" in q_norm or "sipra one" in q_norm:
        entity = "SipraOne"
    elif "siprahub" in q_norm or "sipra hub" in q_norm:
        entity = "SipraHub"
    elif "airis" in q_norm:
        entity = "AIRIS"
    elif "sipra" in q_norm:
        entity = "Sipra"
    else:
        # Fallback entity match via preposition
        m_proj = re.search(r"\b(?:in|for|of|using|used|with|about)\s+([a-zA-Z0-9_\-\.]+)\b", q_norm)
        if m_proj and len(m_proj.group(1).strip()) >= 3 and m_proj.group(1).strip() not in {"the", "this", "what", "which", "how"}:
            entity = m_proj.group(1).strip().capitalize()

    # 2. Attribute & Category Detection
    attributes: set[str] = set()
    category: AttributeCategory = AttributeCategory.GENERAL

    # Port / Configuration check
    has_port = bool(re.search(r"\b(?:ports?|port\s+numbers?|listening\s+port|which\s+port|what\s+port)\b", q_norm))
    if has_port:
        category = AttributeCategory.CONFIGURATION
        if "frontend" in q_norm:
            attributes.add("frontend port")
        if "backend" in q_norm:
            attributes.add("backend port")
        if not attributes:
            attributes.add("port")

    # Framework / Tech Stack check
    has_tech_kw = any(kw in q_norm for kw in ("frontend", "backend", "tech stack", "technology stack", "software framework", "web framework", "built with"))
    if has_tech_kw and not has_port:
        category = AttributeCategory.TECHNOLOGY
        if "frontend" in q_norm:
            attributes.add("frontend")
        if "backend" in q_norm:
            attributes.add("backend")
        if "tech stack" in q_norm or "technology" in q_norm:
            attributes.add("tech stack")
        if not attributes:
            attributes.add("technology")

    # Deployment process check
    if any(kw in q_norm for kw in ("pm2", "nginx", "docker", "deployment process", "process manager")) and category == AttributeCategory.GENERAL:
        category = AttributeCategory.DEPLOYMENT
        attributes.add("deployment")

    # Construct clean normalized query for vector search
    # Only rewrite query when explicitly asking about frontend/backend tech stack or ports
    is_explicit_tech_stack_query = any(kw in q_norm for kw in ("frontend", "backend", "tech stack", "technology stack", "built with"))
    if entity and category == AttributeCategory.TECHNOLOGY and is_explicit_tech_stack_query:
        normalized_q = f"What frontend and backend technologies and frameworks are used in {entity}?"
    elif entity and category == AttributeCategory.CONFIGURATION and has_port:
        normalized_q = f"What ports do frontend and backend use in {entity}?"
    else:
        normalized_q = q_clean

    return QueryIntent(
        raw_query=q_clean,
        normalized_query=normalized_q,
        entity=entity,
        attributes=attributes,
        category=category,
    )
