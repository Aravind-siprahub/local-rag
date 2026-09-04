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
    CULTURE_VALUES = "culture_values"
    POLICY_LEAVE = "policy_leave"
    POLICY_WFH = "policy_wfh"
    POLICY_POSH = "policy_posh"
    POLICY_GRIEVANCE = "policy_grievance"
    POLICY_PERFORMANCE = "policy_performance"
    POLICY_EXIT = "policy_exit"
    POLICY_IT_SECURITY = "policy_it_security"
    WORKING_HOURS = "working_hours"
    POLICY_GENERAL = "policy_general"
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
    q_norm = re.sub(r"\b(polcies|policie|policys|polices|plocies)\b", "policies", q_norm)
    q_norm = re.sub(r"\b(?:what about|tell me about|how about|explain|give details on|tell|please|me|show|give)\b\s*", "", q_norm).strip()

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

    # WFH / Remote & Hybrid Work check (checked first to avoid confusion with general policies)
    has_wfh_policy = any(
        kw in q_norm
        for kw in (
            "wfh", "work from home", "remote work", "remote working", "hybrid work",
            "work remotely", "working remotely", "wfh policy", "remote policy",
            "hybrid policy", "remote and hybrid"
        )
    )
    if has_wfh_policy:
        category = AttributeCategory.POLICY_WFH
        attributes.add("wfh_policy")

    # Working Hours / Shift Schedule check
    has_working_hours = any(
        kw in q_norm
        for kw in (
            "working hours", "work hours", "shift timings", "shift hours",
            "office timings", "work timings", "daily hours", "hours per day",
            "weekly off", "weekly offs", "standard working hours"
        )
    )
    if has_working_hours and category == AttributeCategory.GENERAL:
        category = AttributeCategory.WORKING_HOURS
        attributes.add("working_hours")

    # Leave Policy / Entitlement / Types check
    has_leave_policy = any(
        kw in q_norm
        for kw in (
            "leave policy", "leave policies", "leave entitlement", "casual leave", "casual leaves",
            "sick leave", "earned leave", "maternity leave", "leave balance",
            "how many days are allowed", "how many days allowed", "days allowed",
            "how many leave", "how many leaves", "leave days", "carry forward of leave",
            "carry forward", "leave carry forward", "leave carry-forward", "unused leave",
            "leave per month", "leaves per month", "leave can take", "probation leave",
            "leaves available", "types of leave", "types of leaves", "types leave", "leave types", "what leaves",
            "which leave", "which leaves", "leave categories", "leave without pay", "lwp",
            "public holidays", "holiday calendar", "company holidays", "leave rules", "leave rule",
            "leave polices", "leave guidelines", "leave procedure", "leaves", "leave"
        )
    )
    if has_leave_policy and category == AttributeCategory.GENERAL:
        category = AttributeCategory.POLICY_LEAVE
        attributes.add("leave_policy")
        if "casual leave" in q_norm or "casual leaves" in q_norm:
            attributes.add("casual_leave")
        if "days" in q_norm or "how many" in q_norm:
            attributes.add("leave_days")

    # General Policies list check (e.g. "what are policies available in siprahub")
    has_policies_general = any(
        kw in q_norm
        for kw in (
            "polcies available", "policies available", "what are policies",
            "what are the policies", "list of policies", "company policies", "all policies",
            "overview of policies"
        )
    )
    if has_policies_general and category == AttributeCategory.GENERAL:
        category = AttributeCategory.POLICY_GENERAL
        attributes.add("policies_general")

    # POSH Policy check
    has_posh = any(
        kw in q_norm
        for kw in (
            "posh", "sexual harassment", "icc", "internal complaints committee",
            "prevention of sexual harassment"
        )
    )
    if has_posh and category == AttributeCategory.GENERAL:
        category = AttributeCategory.POLICY_POSH
        attributes.add("posh_policy")

    # Grievance Redressal check
    has_grievance = any(
        kw in q_norm
        for kw in (
            "grievance", "grievances", "grievance redressal", "dispute resolution",
            "anti-retaliation", "retaliation policy", "employee complaint"
        )
    )
    if has_grievance and category == AttributeCategory.GENERAL:
        category = AttributeCategory.POLICY_GRIEVANCE
        attributes.add("grievance_policy")

    # Performance Management check
    has_performance = any(
        kw in q_norm
        for kw in (
            "performance management", "performance appraisal", "review cycle",
            "performance evaluation", "pip", "performance improvement plan"
        )
    )
    if has_performance and category == AttributeCategory.GENERAL:
        category = AttributeCategory.POLICY_PERFORMANCE
        attributes.add("performance_policy")

    # Exit & Termination check
    has_exit = any(
        kw in q_norm
        for kw in (
            "resignation", "resign", "notice period", "exit policy", "termination",
            "separation process", "relieving letter", "handover process"
        )
    )
    if has_exit and category == AttributeCategory.GENERAL:
        category = AttributeCategory.POLICY_EXIT
        attributes.add("exit_policy")

    # IT & Security check
    has_it_sec = any(
        kw in q_norm
        for kw in (
            "it security", "security policy", "it policy", "data security",
            "acceptable use", "device policy", "password policy", "cybersecurity"
        )
    )
    if has_it_sec and category == AttributeCategory.GENERAL:
        category = AttributeCategory.POLICY_IT_SECURITY
        attributes.add("it_security_policy")

    # Culture / Values / Conduct / Ethics check
    has_culture_values = any(
        kw in q_norm
        for kw in (
            "core value", "core values", "values", "company value", "company values",
            "culture", "code of conduct", "principles", "standards of behavior",
            "conduct", "ethics", "ethical standards", "workplace culture"
        )
    )
    if has_culture_values and category == AttributeCategory.GENERAL:
        category = AttributeCategory.CULTURE_VALUES
        if "value" in q_norm:
            attributes.add("values")
        if "culture" in q_norm:
            attributes.add("culture")
        if "conduct" in q_norm or "code of conduct" in q_norm:
            attributes.add("code_of_conduct")
        if "ethics" in q_norm or "ethical" in q_norm:
            attributes.add("ethics")
        if "principle" in q_norm:
            attributes.add("principles")
        if not attributes:
            attributes.add("culture_values")

    # Port / Configuration check
    has_port = bool(re.search(r"\b(?:ports?|port\s+numbers?|listening\s+port|which\s+port|what\s+port)\b", q_norm))
    if has_port and category == AttributeCategory.GENERAL:
        category = AttributeCategory.CONFIGURATION
        if "frontend" in q_norm:
            attributes.add("frontend port")
        if "backend" in q_norm:
            attributes.add("backend port")
        if not attributes:
            attributes.add("port")

    # Framework / Tech Stack check
    has_tech_kw = any(kw in q_norm for kw in ("frontend", "backend", "tech stack", "technology stack", "software framework", "web framework", "built with"))
    if has_tech_kw and not has_port and category == AttributeCategory.GENERAL:
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

    # Construct clean normalized query for vector search without erasing user's specific requested sub-topics
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


def decompose_query_topics(raw_query: str) -> list[str]:
    """Decompose a natural language query into distinct requested sub-topics.

    Examples:
        "What are the rules for casual leave and sick leave?"
        -> ["casual leave", "sick leave"]

        "What is the probation period and notice period?"
        -> ["probation period", "notice period"]

        "What are Our Core Values of Siprahub?"
        -> ["Our Core Values"]

        "What are the working hours, shift timings, and attendance rules?"
        -> ["working hours", "shift timings", "attendance rules"]
    """
    if not raw_query or not raw_query.strip():
        return []

    q = raw_query.strip()
    q_clean = re.sub(r"[?\"'`]", " ", q).strip()

    # Strip common conversational question prefixes
    clean_core = re.sub(
        r"(?i)^(?:what\s+(?:are|is|was|were|do|does|did)?\s*(?:the\s+)?(?:rules\s+(?:for|about)\s+|policies\s+(?:for|about)\s+|guidelines\s+(?:for|about)\s+|details\s+(?:for|on|about)\s+)?|"
        r"can\s+you\s+(?:tell|explain|provide)\s+(?:me\s+)?(?:about\s+)?|"
        r"tell\s+(?:me\s+)?(?:about\s+)?|"
        r"explain\s+(?:to\s+me\s+)?(?:about\s+)?|"
        r"give\s+(?:me\s+)?(?:details\s+(?:on|about)\s+)?|"
        r"describe\s+(?:the\s+)?)\s*",
        "",
        q_clean,
    ).strip()

    # Strip standalone 'what <topic>' or 'which <topic>' prefix
    clean_core = re.sub(r"(?i)^(?:what|which|how)\s+(?:is|are|was|were|do|does|did)?\s*", "", clean_core).strip()

    # Strip company/project prepositional suffixes like "in SipraHub", "of SipraHub", "for Talk to My Data"
    clean_core = re.sub(
        r"(?i)\s+(?:in|of|for|at|about)\s+(?:siprahub|sipraone|sipra|airis|talk\s+to\s+my\s+data|the\s+company|our\s+company|the\s+organization|the\s+project|this\s+project|the\s+handbook|the\s+document)\s*$",
        "",
        clean_core,
    ).strip()

    # Strip trailing conversational verbs (e.g. "using", "used", "implemented")
    clean_core = re.sub(r"(?i)\s+(?:is\s+|are\s+)?(?:using|used|utilized|implemented|applied|available)\s*$", "", clean_core).strip()

    # Replace conjunctions and list separators with standard delimiter
    normalized_delim = re.sub(
        r"(?i)\s*(?:,\s*and\s+|,\s*|\s+and\s+|\s+as\s+well\s+as\s+|\s+along\s+with\s+|\s+versus\s+|\s+vs\.?\s+)\s*",
        " || ",
        clean_core,
    )

    parts = [p.strip() for p in normalized_delim.split("||") if p.strip()]

    stopwords = {"the", "a", "an", "our", "their", "its", "rules", "policy", "policies"}
    valid_topics: list[str] = []
    for p in parts:
        clean_p = re.sub(r"^(?:the|a|an|our|their|its)\s+", "", p, flags=re.IGNORECASE).strip()
        words = [w for w in clean_p.split() if w.lower() not in stopwords]
        if words:
            valid_topics.append(clean_p)

    if not valid_topics:
        return [clean_core] if clean_core else [q]

    return valid_topics


