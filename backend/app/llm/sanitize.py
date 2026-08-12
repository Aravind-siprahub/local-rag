"""Strip model thinking/reasoning leakage from LLM answers."""
from __future__ import annotations

import re

_THINKING_BLOCK_PATTERNS = (
    r"由于.*?结果",
    r"<thinking\b[^>]*>.*?</thinking>",
    r"<redacted_thinking\b[^>]*>.*?</redacted_thinking>",
)

_UNCLOSED_THINKING_RE = re.compile(
    r"<(?:think|thinking|redacted_thinking)\b[^>]*>.*\Z",
    re.IGNORECASE | re.DOTALL,
)

_UNOPENED_THINKING_RE = re.compile(
    r"^.*?</(?:think|thinking|redacted_thinking)\b[^>]*>\s*",
    re.IGNORECASE | re.DOTALL,
)

_REASONING_PREFIX_RE = re.compile(
    r"(?is)^\s*(?:"
    r"let me (?:think|analyze|check|look|see|consider|review|examine|re-read|reread|read)|"
    r"i(?:'ll| will) (?:think|analyze|check|look|formulate|consider|re-read|reread)|"
    r"first[,.]?\s+(?:i'll|i will|let me|the user|a simple)|"
    r"hmm[,.]?\s+(?:the user|i|let)|"
    r"looking at (?:the )?(?:passage|chunk|excerpt|document|context)|"
    r"okay[,.]?\s+(?:the user|let me)|"
    r"wait[,.]?\s+|"
    r"checks (?:requirements|instructions)|"
    r"the user (?:seems|asks|asked|is asking|just said|wants|wanted)|"
    r"how to (?:reconcile|resolve|handle|proceed)|"
    r"however[,.]?\s+(?:the instruction|the prompt|the system|the requirement)|"
    r"i need to respond|"
    r"the instruction (?:says|states|requires)|"
    r"reconcil(?:e|ing) (?:the|these)|"
    r"(?:re-read|reread|re-reading) (?:the|the prompt|the question)|"
    r"(?:let me|i should) (?:re-read|reread|review|check) (?:the|the prompt|the question)"
    r")[^\n]*\n+"
)

_REASONING_ONLY_RE = re.compile(
    r"(?is)^\s*(?:"
    r"let me (?:think|analyze|check|look)|"
    r"okay[,.]?\s+the user|"
    r"i need to|"
    r"the user (?:asks|is asking|just said|seems|asked|wants|wanted)|"
    r"first[,.]?\s+i'll|"
    r"hmm[,.]?\s+the user|"
    r"checks requirements|"
    r"wait\b|"
    r"how to (?:reconcile|resolve|handle|proceed)|"
    r"however[,.]?\s+(?:the instruction|the prompt|the system|the requirement)|"
    r"the instruction (?:says|states|requires)|"
    r"reconcil(?:e|ing) (?:the|these)|"
    r"(?:re-read|reread|re-reading) (?:the|the prompt|the question)|"
    r"(?:let me|i should) (?:re-read|reread|review|check) (?:the|the prompt|the question)"
    r").*\Z"
)

_INCOMPLETE_REASONING_FRAGMENT_RE = re.compile(
    r"(?is)^\s*(?:wait|hmm|okay|let me think)\.?\s*$"
)

_THINK_MODEL_MARKERS = ("qwen3", "qwen", "qwq", "deepseek-r1", "deepseek", "r1", "think", "coder")


def supports_think_parameter(model: str | None) -> bool:
    """Return True when the Ollama model accepts the ``think`` request flag."""
    if not model:
        return False
    lowered = model.lower()
    return any(marker in lowered for marker in _THINK_MODEL_MARKERS)


def detect_reasoning_leakage(text: str | None) -> bool:
    """Detect if unhandled internal thinking tags exist in output."""
    if not text or not isinstance(text, str) or not text.strip():
        return False

    lowered = text.lower().strip()
    # Support both U+7ED3 (结) and U+7EFE (结) variants of "结果"
    tags = ("由于", "结果", "结果", "<thinking>", "</thinking>", "<redacted_thinking>")
    return any(tag in lowered for tag in tags)


def _strip_common_monologue_prefixes(text: str) -> str:
    """Strip leading sentences that match reasoning/narration monologue."""
    cleaned = text.strip()
    
    # Prefix patterns to strip (case-insensitive)
    patterns = [
        r"^looking\s+at\s+the\s+(?:web\s+)?search\s+results[,.:]*\s*",
        r"^based\s+on\s+the\s+(?:web\s+)?search\s+results[,.:]*\s*",
        r"^according\s+to\s+the\s+(?:web\s+)?search\s+results[,.:]*\s*",
        r"^based\s+only\s+on\s+the\s+search\s+results[^.]*\s*",
        r"^i\s+need\s+to\s+answer\s+in\s+\d+-\d+\s+sentences[,.:]*\s*",
        r"^i\s+need\s+to\s+[^.]*sentences[,.:]*\s*",
        r"^i\s+shouldn'?t\s+add\s+any\s+additional\s+information[,.:]*\s*",
        r"^i\s+shouldn'?t\s+add\s+any\s+information[,.:]*\s*",
        r"^i\s+should\s+not\s+add\s+any\s+additional\s+information[,.:]*\s*",
        r"^i\s+should\s+not\s+add\s+any\s+information[,.:]*\s*",
        r"^first[,.]?\s+i\s+need\s+to\s+[^.]*\s*",
        r"^the\s+answer\s+is[,.:]*\s*",
        r"^here\s+is\s+the\s+answer[,.:]*\s*",
    ]
    
    modified = True
    while modified:
        modified = False
        for pat in patterns:
            new_text = re.sub(pat, "", cleaned, flags=re.IGNORECASE)
            if new_text != cleaned:
                cleaned = new_text.strip()
                modified = True
                break
    return cleaned


def sanitize_response(text: str | None) -> str:
    """Remove thinking blocks and leading reasoning monologues; return clean answer."""
    if text is None or not isinstance(text, str) or not text.strip():
        return ""

    cleaned = text

    for pattern in _THINKING_BLOCK_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned, flags=re.DOTALL | re.IGNORECASE)

    cleaned = _UNOPENED_THINKING_RE.sub("", cleaned)
    cleaned = _strip_unclosed_thinking_blocks(cleaned)
    cleaned = _strip_reasoning_paragraphs(cleaned)

    if _REASONING_ONLY_RE.match(cleaned):
        return ""

    if _INCOMPLETE_REASONING_FRAGMENT_RE.match(cleaned):
        return ""

    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _strip_unclosed_thinking_blocks(text: str) -> str:
    return _UNCLOSED_THINKING_RE.sub("", text)


def _strip_reasoning_paragraphs(text: str) -> str:
    cleaned = text
    # Strip a few leading reasoning-style paragraphs without looping forever.
    for _ in range(5):
        updated = _REASONING_PREFIX_RE.sub("", cleaned, count=1)
        if updated == cleaned:
            break
        cleaned = updated
    return cleaned


class ThinkingStreamFilter:
    """Filter streamed tokens that fall inside thinking tags."""

    def __init__(self) -> None:
        self._buffer = ""
        self._in_think = False

    def feed(self, token: str) -> str:
        if not token:
            return ""
        self._buffer += token
        out: list[str] = []

        while self._buffer:
            lowered = self._buffer.lower()
            if self._in_think:
                end_idx = -1
                end_tag = None
                for tag in ("结果", "结果", "</thinking>", "</redacted_thinking>"):
                    idx = lowered.find(tag)
                    if idx != -1 and (end_idx == -1 or idx < end_idx):
                        end_idx = idx
                        end_tag = tag
                if end_idx == -1:
                    # Keep a short tail in case a closing tag is split across chunks.
                    keep = min(len(self._buffer), 24)
                    self._buffer = self._buffer[-keep:]
                    return "".join(out)
                self._buffer = self._buffer[end_idx + len(end_tag or "") :]
                self._in_think = False
                continue

            end_idx = -1
            end_tag = None
            for tag in ("结果", "结果", "</thinking>", "</redacted_thinking>"):
                idx = lowered.find(tag)
                if idx != -1 and (end_idx == -1 or idx < end_idx):
                    end_idx = idx
                    end_tag = tag

            start_idx = -1
            start_tag = None
            for tag in ("由于", "<thinking>", "<redacted_thinking>"):
                idx = lowered.find(tag)
                if idx != -1 and (start_idx == -1 or idx < start_idx):
                    start_idx = idx
                    start_tag = tag

            if end_idx != -1 and (start_idx == -1 or end_idx < start_idx):
                out.clear()
                self._buffer = self._buffer[end_idx + len(end_tag or "") :]
                self._in_think = False
                continue

            if start_idx == -1:
                out.append(self._buffer)
                self._buffer = ""
                break
            if start_idx > 0:
                out.append(self._buffer[:start_idx])
            self._buffer = self._buffer[start_idx + len(start_tag or "") :]
            self._in_think = True

        return "".join(out)

    def flush(self) -> str:
        if self._in_think:
            self._buffer = ""
            return ""
        leftover = self._buffer
        self._buffer = ""
        return leftover
