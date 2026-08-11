"""Strip internal LLM reasoning from user-facing chat responses."""
from __future__ import annotations

import re

# Explicit tag literals (built via concatenation to avoid tooling corruption).
_TAG_THINK_OPEN = "<" + "think" + ">"
_TAG_THINK_CLOSE = "</" + "think" + ">"
_TAG_RED_OPEN = "<" + "redacted_thinking" + ">"
_TAG_RED_CLOSE = "</" + "redacted_thinking" + ">"
_TAG_DS_OPEN = "`" + "think" + "`"
_TAG_DS_CLOSE = "`" + "/" + "think" + "`"
_TAG_QWEN_OPEN = "`" + "think" + "`"
_TAG_QWEN_CLOSE = "\\" + "`" + "think" + "`"
_TAG_XML_OPEN = "<" + "thinking" + ">"
_TAG_XML_CLOSE = "</" + "thinking" + ">"

_QWEN_THINK_BLOCK = re.escape(_TAG_QWEN_OPEN) + r"[\s\S]*?" + re.escape(_TAG_QWEN_CLOSE)
_DS_THINK_BLOCK = re.escape(_TAG_DS_OPEN) + r"[\s\S]*?" + re.escape(_TAG_DS_CLOSE)
_THINK_BLOCK = re.escape(_TAG_THINK_OPEN) + r"[\s\S]*?" + re.escape(_TAG_THINK_CLOSE)

_THINKING_BLOCK_PATTERNS: tuple[str, ...] = (
    _THINK_BLOCK,
    re.escape(_TAG_RED_OPEN) + r".*?" + re.escape(_TAG_RED_CLOSE),
    _QWEN_THINK_BLOCK,
    _DS_THINK_BLOCK,
    re.escape(_TAG_XML_OPEN) + r".*?" + re.escape(_TAG_XML_CLOSE),
)

# Common chain-of-thought monologue prefixes (paragraph or sentence-level).
# Each pattern anchors at the start of the string (^) and is applied
# iteratively until no more matches remain.
# Sentence-boundary patterns stop at [.!?]+\s+(?=[A-Z0-9]) when an answer
# follows on the same line, or fall back to \n\n or \Z for full-paragraph CoT.
_REASONING_PREFIX_PATTERNS: tuple[str, ...] = (
    # --- "It doesn't / This passage" openers ---
    r"(?is)^it doesn(?:'|')?t (?:directly )?(?:mention|state|define|contain).*?(?=\n\n|\Z)",
    r"(?is)^this passage (?:discusses|mentions|gives|states|provides|has|is).*?(?=\n\n|\Z)",
    r"(?is)^according to passage \d+[:.]?\s*",
    # --- "Okay / Hmm" openers ---
    r"(?is)^okay[,.]?\s+the user(?:.*?[.!?]+\s+(?=[A-Z0-9])|.*?(?=\n\n|\Z))",
    r"(?is)^okay[,.]?\s+let me(?:.*?[.!?]+\s+(?=[A-Z0-9])|.*?(?=\n\n|\Z))",
    r"(?is)^okay[,.]?\s+let'?s\s+(?:.*?[.!?]+\s+(?=[A-Z0-9])|.*?(?=\n\n|\Z))",
    r"(?is)^okay[,.]?\s+so\s+i\s+need(?:.*?[.!?]+\s+(?=[A-Z0-9])|.*?(?=\n\n|\Z))",
    r"(?is)^hmm[,.]?\s*(?:.*?[.!?]+\s+(?=[A-Z0-9])|.*?(?=\n\n|\Z))",
    # --- "Let me" openers ---
    r"(?is)^let me think(?:.*?[.!?]+\s+(?=[A-Z0-9])|.*?(?=\n\n|\Z))",
    r"(?is)^let me analyze(?:.*?[.!?]+\s+(?=[A-Z0-9])|.*?(?=\n\n|\Z))",
    r"(?is)^let me check(?:.*?[.!?]+\s+(?=[A-Z0-9])|.*?(?=\n\n|\Z))",
    r"(?is)^let me review(?:.*?[.!?]+\s+(?=[A-Z0-9])|.*?(?=\n\n|\Z))",
    r"(?is)^let me look(?:.*?[.!?]+\s+(?=[A-Z0-9])|.*?(?=\n\n|\Z))",
    r"(?is)^let me see(?:.*?[.!?]+\s+(?=[A-Z0-9])|.*?(?=\n\n|\Z))",
    # --- Chunk/passage/context reference openers ---
    r"(?is)^based on (?:the )?(?:chunks?|passages?|excerpts?|context|information|documents?)(?: (?:provided|above|below))?[,.]?\s*",
    r"(?is)^from (?:the )?(?:chunks?|passages?|excerpts?|context)(?: (?:provided|above|below))?[,.]?\s*",
    r"(?is)^from (?:the )?above[,.]?\s*",
    r"(?is)^looking at (?:the )?(?:document|documents|excerpts?|passages?|chunks?|sections?|\s+)+[s\d]*[,.:]?\s*(?:i\s+can\s+see\s+that\s*)?",
    r"(?is)^looking at (?:the )?(?:document|documents|excerpts?|passages?|chunks?|sections?|\s+)+(?:.*?[.!?]+\s+(?=[A-Z0-9])|.*?(?=\n\n|\Z))",
    r"(?is)^(?:the )?(?:chunks?|passages?|excerpts?|documents?) (?:above|below|provided|show|indicate)(?:.*?[.!?]+\s+(?=[A-Z0-9])|.*?(?=\n\n|\Z))",
    # --- Passage/Chunk bullet-list reasoning monologues ---
    r"(?is)^(?:[-*]\s+(?:passage|chunk|excerpt|document)\s*\d+.*?(?:\n|\Z))+",
    # --- "To answer / To respond" openers ---
    r"(?is)^to answer this(?:.*?[.!?]+\s+(?=[A-Z0-9])|.*?(?=\n\n|\Z))",
    r"(?is)^to respond to(?:.*?[.!?]+\s+(?=[A-Z0-9])|.*?(?=\n\n|\Z))",
    r"(?is)^in order to answer(?:.*?[.!?]+\s+(?=[A-Z0-9])|.*?(?=\n\n|\Z))",
    # --- "Important: Must not invent..." / Meta-commentary ---
    r"(?is)^important[,:]?\s+(?:must not|i must|we must|the document|never).*?(?=\n\n|\Z)",
    # --- "I" subject openers ---
    r"(?is)^i(?:'|')?ll formulate(?:.*?[.!?]+\s+(?=[A-Z0-9])|.*?(?=\n\n|\Z))",
    r"(?is)^i(?:'|')?ll (?:now |start |begin |try |attempt |use )(?:.*?[.!?]+\s+(?=[A-Z0-9])|.*?(?=\n\n|\Z))",
    r"(?is)^i need to(?:.*?[.!?]+\s+(?=[A-Z0-9])|.*?(?=\n\n|\Z))",
    r"(?is)^i should(?:.*?[.!?]+\s+(?=[A-Z0-9])|.*?(?=\n\n|\Z))",
    r"(?is)^i think(?:.*?[.!?]+\s+(?=[A-Z0-9])|.*?(?=\n\n|\Z))",
    r"(?is)^i can see(?:.*?[.!?]+\s+(?=[A-Z0-9])|.*?(?=\n\n|\Z))",
    r"(?is)^i will (?:now |start |begin |try |analyze |use )(?:.*?[.!?]+\s+(?=[A-Z0-9])|.*?(?=\n\n|\Z))",
    # --- "First, I'll" / "First, I need" (only when followed by reasoning verb) ---
    r"(?is)^first[,.]?\s+i(?:'|')?ll(?:.*?[.!?]+\s+(?=[A-Z0-9])|.*?(?=\n\n|\Z))",
    r"(?is)^first[,.]?\s+i need to(?:.*?[.!?]+\s+(?=[A-Z0-9])|.*?(?=\n\n|\Z))",
    r"(?is)^first[,.]?\s+let me(?:.*?[.!?]+\s+(?=[A-Z0-9])|.*?(?=\n\n|\Z))",
    # --- "The user" openers ---
    r"(?is)^the user is asking(?:.*?[.!?]+\s+(?=[A-Z0-9])|.*?(?=\n\n|\Z))",
    r"(?is)^the user wants(?:.*?[.!?]+\s+(?=[A-Z0-9])|.*?(?=\n\n|\Z))",
    r"(?is)^the user has asked(?:.*?[.!?]+\s+(?=[A-Z0-9])|.*?(?=\n\n|\Z))",
    # --- Generic LLM Greetings / Filler Intros ---
    r"(?is)^hello!?\s+i'?m\s+ready\s+to\s+help.*?(?=\n\n|\Z)",
    r"(?is)^hello!?\s+(?:how\s+can\s+i\s+help|just\s+let\s+me\s+know).*?(?=\n\n|\Z)",
    r"(?is)^sure[,!]?\s+(?:i\s+can\s+help|i'll\s+help).*?(?=\n\n|\Z)",
    r"(?is)^let\s+me\s+help\s+you\s+with.*?(?=\n\n|\Z)",
    r"(?is)^i\s+found\s+the\s+following\s+information.*?(?=\n\n|\Z)",
    r"(?is)^here\s+is\s+the\s+information\s+(?:from|about).*?(?=\n\n|\Z)",
    r"(?is)^(?:based\s+on|according\s+to)\s+(?:your\s+)?uploaded\s+documents?.*?(?=\n\n|\Z)",
    # --- Query analysis / CoT meta-commentary openers ---
    r"(?is)^okay[,.]?\s+.*?(?=\n\n|\Z)",
    r"(?is)^(?:that|the)\s+(?:phrasing|wording|statement|query|question|user's?\s+input|core\s+question)\s+(?:is|seems|might|could).*?(?=\n\n|\Z)",
    r"(?is)^the user\s+(?:seems|might|probably|specifically|appears|is\s+asking|wants|has\s+asked).*?(?=\n\n|\Z)",
    r"(?is)^i recall\s+that.*?(?=\n\n|\Z)",
    r"(?is)^wait[,.]?\s+.*?(?=\n\n|\Z)",
    r"(?is)^checks?\s+reliable\s+knowledge.*?(?=\n\n|\Z)",
    r"(?is)^(?:checking|recalling|analyzing|unpacking|evaluating)\s+.*?(?=\n\n|\Z)",
    r"(?is)^since\s+(?:they|the\s+user|i|we)\s+.*?(?=\n\n|\Z)",
    r"(?is)^so\s+the\s+(?:cleanest|best|final|direct)\s+(?:response|answer)\s+is[:.]?\s*",
    r"(?is)^the\s+(?:cleanest|best|final|direct)\s+(?:response|answer)\s+is[:.]?\s*",
    r"(?is)^in\s+astronomy[,.]?\s+.*?(?=\n\n|\Z)",
)

_REASONING_ONLY_RE = re.compile(
    r"(?is)^(?:"
    r"okay[,.]?\s+(?:the user|let me|let'?s|so\s+i\s+need)|"
    r"hmm[,.]?\s|"
    r"let me think|"
    r"first[,.]?\s+i(?:'|')?ll|"
    r"first[,.]?\s+i need to|"
    r"i need to|i should|the user is asking|the user wants|the user seems|the user might|"
    r"that phrasing is|the phrasing is|checks reliable knowledge|i recall that"
    r")[\s\S]*$"
)

_REASONING_START_RE = re.compile(
    r"(?is)^(?:okay|hmm|let me|i need to|i should|the user|that phrasing|the phrasing|checks reliable)\b",
)

# Truncated reasoning tail when num_predict cuts off mid-monologue.
_INCOMPLETE_REASONING_FRAGMENT_RE = re.compile(
    r"(?is)^(?:wait|so|but|hmm|okay|well|right|now|let me)(?:[,.!?]|\s)*$"
)

_OPEN_TAGS: tuple[str, ...] = (
    _TAG_THINK_OPEN,
    _TAG_RED_OPEN,
    _TAG_QWEN_OPEN,
    _TAG_DS_OPEN,
    _TAG_XML_OPEN,
)
_CLOSE_TAGS: tuple[str, ...] = (
    _TAG_THINK_CLOSE,
    _TAG_RED_CLOSE,
    _TAG_QWEN_CLOSE,
    _TAG_DS_CLOSE,
    _TAG_XML_CLOSE,
)

_REASONING_MODEL_MARKERS: tuple[str, ...] = (
    "r1",
    "qwq",
    "qwen3",
    "thinking",
    "reasoner",
    "reasoning",
    "deepseek-r1",
    "o1",
    "o3",
)

# Models known to accept Ollama's `think` request parameter (qwen3+).
_THINK_PARAM_MODEL_MARKERS: tuple[str, ...] = (
    "qwen3",
    "qwq",
    "deepseek-r1",
    "r1",
)


def is_reasoning_model(model_name: str | None) -> bool:
    """Return True when the model is known to emit chain-of-thought output."""
    if not model_name:
        return False
    lowered = model_name.lower()
    return any(marker in lowered for marker in _REASONING_MODEL_MARKERS)


def supports_think_parameter(model_name: str | None) -> bool:
    """Return True only for models that accept Ollama's `think` API parameter."""
    if not model_name:
        return False
    lowered = model_name.lower()
    return any(marker in lowered for marker in _THINK_PARAM_MODEL_MARKERS)


def detect_reasoning_leakage(text: str | None) -> bool:
    """Detect if unhandled internal thinking tags exist in output."""
    if not text or not isinstance(text, str) or not text.strip():
        return False

    lowered = text.lower().strip()
    tags = ("<think>", "</think>", "<thinking>", "</thinking>", "<redacted_thinking>")
    return any(tag in lowered for tag in tags)


def sanitize_response(text: str | None) -> str:
    """Remove explicit thinking blocks (<think>...</think>, <thinking>...</thinking>) and leading reasoning monologues; return clean answer."""
    if text is None or not isinstance(text, str) or not text.strip():
        return ""

    cleaned = text

    # Remove closed thinking blocks
    for pattern in _THINKING_BLOCK_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned, flags=re.DOTALL | re.IGNORECASE)

    # Handle unclosed thinking tags
    cleaned = _strip_unclosed_thinking_blocks(cleaned)

    # Strip leading reasoning monologues ("Let me analyze...", "Looking at Chunk...", "I'll formulate...", etc.)
    cleaned = _strip_reasoning_paragraphs(cleaned)

    # If the remaining text is entirely a reasoning monologue, clear it completely
    if _REASONING_ONLY_RE.match(cleaned):
        return ""

    # If the remaining text is just an incomplete reasoning fragment (e.g. "Wait"), clear it
    if _INCOMPLETE_REASONING_FRAGMENT_RE.match(cleaned):
        return ""

    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()




def _strip_reasoning_paragraphs(text: str) -> str:
    """Strip one or more leading reasoning paragraphs (iterative — order matters)."""
    cleaned = text
    previous = None
    while cleaned != previous:
        previous = cleaned
        cleaned = cleaned.lstrip()
        for pattern in _REASONING_PREFIX_PATTERNS:
            cleaned = re.sub(pattern, "", cleaned)
    return cleaned


def _strip_unclosed_thinking_blocks(text: str) -> str:
    """Handle thinking tags split across chunks or missing closing tags."""
    result = text
    for open_tag, close_tag in zip(_OPEN_TAGS, _CLOSE_TAGS, strict=True):
        if not open_tag:
            continue
        if close_tag and close_tag.lower() in result.lower():
            parts = re.split(re.escape(close_tag), result, flags=re.IGNORECASE)
            result = parts[-1]
        elif open_tag.lower() in result.lower():
            parts = re.split(re.escape(open_tag), result, flags=re.IGNORECASE, maxsplit=1)
            result = parts[0]
    return result


class ThinkingStreamFilter:
    """Incremental filter for SSE token streams — never emits reasoning tokens."""

    def __init__(self) -> None:
        self._buffer = ""
        self._in_thinking = False
        self._leading_hold = ""
        self._leading_resolved = False

    def feed(self, token: str) -> str:
        """Process one token delta; return only user-safe text (may be empty)."""
        if not token:
            return ""
        self._buffer += token
        return self._drain()

    def flush(self) -> str:
        """Emit any remaining safe buffered text at end of stream."""
        remaining = self._drain(flush=True)
        if not self._leading_resolved and self._leading_hold:
            remaining += self._emit_leading_hold(flush=True)
        tail = sanitize_response(self._buffer) if self._buffer else ""
        self._buffer = ""
        self._in_thinking = False
        self._leading_hold = ""
        return remaining + tail

    def _drain(self, *, flush: bool = False) -> str:
        emitted: list[str] = []

        while self._buffer:
            if self._in_thinking:
                closed = self._find_earliest_close(self._buffer)
                if closed is None:
                    if flush:
                        self._buffer = ""
                    break
                end_pos, _ = closed
                self._buffer = self._buffer[end_pos:]
                self._in_thinking = False
                continue

            opened = self._find_earliest_open(self._buffer)
            if opened is not None:
                idx, open_len = opened
                if idx > 0:
                    emitted.append(self._buffer[:idx])
                self._buffer = self._buffer[idx + open_len :]
                self._in_thinking = True
                continue

            if flush:
                emitted.append(self._buffer)
                self._buffer = ""
                break

            hold = self._partial_tag_suffix_len(self._buffer)
            if hold:
                safe = self._buffer[:-hold] if len(self._buffer) > hold else ""
                if safe:
                    emitted.append(safe)
                    self._buffer = self._buffer[-hold:]
                break

            emitted.append(self._buffer)
            self._buffer = ""

        raw = "".join(emitted)
        if not raw:
            return ""
        if not self._leading_resolved:
            return self._emit_leading_hold(raw)
        return sanitize_response(raw) if raw else ""

    def _emit_leading_hold(self, chunk: str = "", *, flush: bool = False) -> str:
        """Hold the first paragraph until we can tell reasoning from a real answer."""
        if chunk:
            self._leading_hold += chunk
        if not flush and "\n\n" not in self._leading_hold:
            hold_stripped = self._leading_hold.lstrip()
            if _REASONING_START_RE.match(hold_stripped):
                if len(self._leading_hold) < 400:
                    return ""
            elif len(hold_stripped) < 8:
                return ""

        candidate = self._leading_hold
        self._leading_hold = ""
        self._leading_resolved = True
        safe = sanitize_response(candidate)
        return safe if safe else ""

    @staticmethod
    def _find_earliest_open(text: str) -> tuple[int, int] | None:
        best: tuple[int, int] | None = None
        for tag in _OPEN_TAGS:
            if not tag:
                continue
            idx = text.lower().find(tag.lower())
            if idx != -1 and (best is None or idx < best[0]):
                best = (idx, len(tag))
        return best

    @staticmethod
    def _find_earliest_close(text: str) -> tuple[int, int] | None:
        best: tuple[int, int] | None = None
        for tag in _CLOSE_TAGS:
            if not tag:
                continue
            idx = text.lower().find(tag.lower())
            if idx != -1 and (best is None or idx < best[0]):
                best = (idx + len(tag), len(tag))
        return best

    @staticmethod
    def _partial_tag_suffix_len(text: str) -> int:
        lowered = text.lower()
        candidates = [t for t in _OPEN_TAGS + _CLOSE_TAGS if t]
        max_hold = 0
        for tag in candidates:
            tag_lower = tag.lower()
            for i in range(1, len(tag_lower)):
                if lowered.endswith(tag_lower[:i]):
                    max_hold = max(max_hold, i)
            if tag_lower.startswith(lowered) and len(lowered) < len(tag_lower):
                max_hold = max(max_hold, len(lowered))
        return max_hold
