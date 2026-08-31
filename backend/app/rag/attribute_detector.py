"""Attribute detection layer for attribute-specific RAG queries.

Identifies exact requested attributes (tech stack, port numbers, deployment, etc.)
so retrieval, reranking, and context filtering target ONLY the requested fact type.
"""
from __future__ import annotations

import re
from enum import Enum, auto


class RequestedAttribute(Enum):
    FRAMEWORK_TECH_STACK = auto()
    PORT_NETWORKING = auto()
    DEPLOYMENT_PROCESS = auto()
    GENERAL_ATTRIBUTE = auto()


def detect_requested_attributes(query: str) -> set[RequestedAttribute]:
    """Detect which specific attributes are requested by the user query."""
    q_low = (query or "").lower().strip()
    attrs: set[RequestedAttribute] = set()

    # Explicit port / networking attribute check
    if re.search(r"\b(?:ports?|port\s+numbers?|listening\s+port|which\s+port|what\s+port)\b", q_low):
        attrs.add(RequestedAttribute.PORT_NETWORKING)

    # Framework / tech stack attribute check (when ports are not explicitly requested)
    if any(kw in q_low for kw in ("frontend", "backend", "fronted", "tech stack", "technology stack", "software framework", "web framework", "library", "built with")) and RequestedAttribute.PORT_NETWORKING not in attrs:
        attrs.add(RequestedAttribute.FRAMEWORK_TECH_STACK)

    # Deployment process attribute check
    if any(kw in q_low for kw in ("pm2", "nginx", "docker", "deployment process", "process manager", "reverse proxy", "proxy", "deployment environment", "deployment")) and not attrs:
        attrs.add(RequestedAttribute.DEPLOYMENT_PROCESS)

    if not attrs:
        attrs.add(RequestedAttribute.GENERAL_ATTRIBUTE)

    return attrs
