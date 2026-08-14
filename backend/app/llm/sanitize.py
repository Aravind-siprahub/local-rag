"""Strip model thinking/reasoning leakage from LLM answers."""
from __future__ import annotations

import re

_THINKING_BLOCK_PATTERNS = (
    r"由于.*?结果",
    r"<thinking\b[^>]*>.*?</thinking>",
    r"<redacted_thinking\b[^>]*>.*?</redacted_thinking>",
    r"<think\b[^>]*>.*?</think>",
    r"`think`.*?`/think`",
    r"`think`.*?\\`think`",
)

_UNCLOSED_THINKING_RE = re.compile(
    r"<(?:think|thinking|redacted_thinking)\b[^>]*>.*\Z|"
    r"`think`.*\Z",
    re.IGNORECASE | re.DOTALL,
)

_UNOPENED_THINKING_RE = re.compile(
    r"^.*?</(?:think|thinking|redacted_thinking)\b[^>]*>\s*|"
    r"^.*?`/(?:think|thinking|redacted_thinking)`\s*|"
    r"^.*?\\`think`\s*",
    re.IGNORECASE | re.DOTALL,
)

# Filler words that often precede reasoning
_FILLER = r"(?:(?:okay|hmm|wait|first|so|now|then|firstly)[,.]?\s*)*"

# Meta-subjects and verbs that indicate internal reasoning or narrative structure
_META_SUBJECT_VERBS = [
    r"let me (?:think|try|unpack|check|look|see|analyze|review|figure|acknowledge)",
    r"let's (?:tackle|unpack|extract|analyze|check|look|see|review)",
    r"i(?:'ll| will| need to| should| must| can) (?:look|review|calculate|answer|check|analyze|read|write|see|try|acknowledge|start|begin)",
    r"the user (?:is asking|asked|seems|wants|just said|has shared|has asked)",
    r"that phrasing is",
    r"this (?:query|question|prompt) (?:asks|requires|seems)",
    r"this seems like",
    r"to answer this question",
    r"checks (?:requirements|instructions)",
    r"how to (?:reconcile|resolve|handle|proceed)",
    r"the instruction (?:says|states|requires)",
    r"(?:re-read|reread|re-reading) (?:the|this)",
    r"it doesn't directly mention",
    r"important:\s*must not",
    r"we (?:are given|have to|need to|must|must answer)",
    r"which (?:one|passage|chunk) to",
    r"the (?:document|passage|chunk|context|excerpt)(?:\s+excerpts?)? (?:clearly )?(?:states|says|shows|mentions|gives)",
    r"[-*]?\s*(?:passage|chunk|document)\s*\d+[:\s]*(?:is about|is critical|discusses|has a key)",
    r"looking at the (?:document|passage|chunk) (?:excerpts|context):?",
    r"okay[,.]?\s+(?:passage|chunk|document)",
]

# Match the FULL reasoning sentence up to its terminating punctuation or newline.
_FULL_REASONING_SENTENCE_RE = re.compile(
    r"(?i)^" + _FILLER + r"(?:" + "|".join(_META_SUBJECT_VERBS) + r").*?(?:[.!?:]+(?:\s+|$)|\n+|$)"
)

# For backward compatibility with ThinkingStreamFilter
_REASONING_PREFIX_RE = _FULL_REASONING_SENTENCE_RE

_INLINE_SOURCE_PREFIX_RE = re.compile(
    r"(?i)^(?:"
    r"(?:looking at|based on|according to|from)\s+(?:the\s+)?(?:context|chunks?|passages?|excerpts?|documents?(?:\s+excerpts?)?|sources?|search results?|web search results?)(?:\s+\d+)?(?:\s+(?:above|provided))?(?:[,.:]*)\s*(?:(?:i can see|we can see|we can infer|we can find|it says|it states|it shows|we see|it is clear)(?:\s+that)?\s*)?|"
    r"(?:chunk|passage)\s+\d+\s+(?:says|states|shows|mentions|specifies|indicates|contains)(?:\s+that)?\s*"
    r")"
)

_REASONING_ONLY_RE = re.compile(
    r"(?is)^\s*(?:"
    r"let me (?:think|analyze|check|look)|"
    r"okay[,.]?\s+(?:the user|let's|so i need|i need)|"
    r"i need to|"
    r"the user (?:asks|is asking|just said|seems|asked|wants|wanted)|"
    r"first[,.]?\s+i'll|"
    r"hmm[,.]?\s+(?:the user|this seems)|"
    r"checks requirements|"
    r"wait\b|"
    r"how to (?:reconcile|resolve|handle|proceed)|"
    r"however[,.]?\s+(?:the instruction|the prompt|the system|the requirement)|"
    r"the instruction (?:says|states|requires)|"
    r"reconcil(?:e|ing) (?:the|these)|"
    r"(?:re-read|reread|re-reading) (?:the|the prompt|the question)|"
    r"(?:let me|i should) (?:re-read|reread|review|check) (?:the|the prompt|the question)"
    r"|\s"
    r")+\s*\Z"
)

_INCOMPLETE_REASONING_FRAGMENT_RE = re.compile(
    r"(?is)^\s*(?:wait|hmm|okay|let me think|first|so|now|then)[.!?:]*\s*$"
)

_THINK_MODEL_MARKERS = ("qwen3", "qwen", "qwq", "deepseek-r1", "deepseek", "r1", "think", "coder")


def supports_think_parameter(model: str | None) -> bool:
    """Return True when the Ollama model accepts the ``think`` request flag."""
    if not model:
        return False
    lowered = model.lower()
    return any(marker in lowered for marker in _THINK_MODEL_MARKERS)


def is_reasoning_model(model: str | None) -> bool:
    """Return True if the model is a reasoning/thinking model (e.g. qwen3, deepseek-r1, qwq, o1)."""
    if not model:
        return False
    lowered = model.lower()
    reasoning_markers = ("qwen3", "deepseek-r1", "qwq", "o1-", "o1/", "/o1", "thinking", "reasoning")
    if any(marker in lowered for marker in reasoning_markers):
        return True
    if lowered == "o1":
        return True
    return False


def detect_reasoning_leakage(text: str | None) -> bool:
    """Detect if unhandled internal thinking tags exist in output."""
    if not text or not isinstance(text, str) or not text.strip():
        return False

    lowered = text.lower().strip()
    # Support both U+7ED3 (结) and U+7EFE (结) variants of "结果"
    tags = ("由于", "结果", "结果", "<thinking>", "</thinking>", "<redacted_thinking>", "<think>", "</think>", "`think`", "`/think`")
    return any(tag in lowered for tag in tags)


def sanitize_response(text: str | None) -> str:
    """Remove thinking blocks and reasoning monologues structurally; return clean answer."""
    if text is None or not isinstance(text, str) or not text.strip():
        return ""

    cleaned = text

    # Strip XML blocks safely
    for _ in range(3):
        old_cleaned = cleaned
        for pattern in _THINKING_BLOCK_PATTERNS:
            cleaned = re.sub(pattern, "", cleaned, flags=re.DOTALL | re.IGNORECASE)
        cleaned = _UNOPENED_THINKING_RE.sub("", cleaned)
        cleaned = _strip_unclosed_thinking_blocks(cleaned)
        if cleaned == old_cleaned:
            break

    # Structural reasoning stripping line-by-line
    lines = cleaned.split("\n")
    last_reasoning_idx = -1

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
            
        orig_stripped = stripped
        while True:
            m = _FULL_REASONING_SENTENCE_RE.match(stripped)
            if m:
                stripped = stripped[m.end():].lstrip()
            else:
                break
                
        # If the line was entirely consumed by reasoning sentences, or just fragments remain
        if not stripped or _INCOMPLETE_REASONING_FRAGMENT_RE.match(stripped):
            last_reasoning_idx = idx
            lines[idx] = ""
            continue
            
        # If the line was partially consumed (reasoning prefix attached to an answer)
        if stripped != orig_stripped:
            lines[idx] = stripped
            last_reasoning_idx = idx - 1

    cleaned_lines = []
    for idx, line in enumerate(lines):
        if idx <= last_reasoning_idx:
            continue
            
        stripped_line = line.strip()
        if not stripped_line:
            if cleaned_lines:
                cleaned_lines.append("")
            continue
            
        # Strip inline source analysis prefixes (e.g. "Chunk 1 says ") from valid lines
        while True:
            m = _INLINE_SOURCE_PREFIX_RE.match(stripped_line)
            if m:
                stripped_line = stripped_line[m.end():].lstrip()
                if stripped_line and stripped_line[0].islower():
                    stripped_line = stripped_line[0].upper() + stripped_line[1:]
            else:
                break
                
        if not stripped_line:
            continue
            
        cleaned_lines.append(stripped_line)

    cleaned = "\n".join(cleaned_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _strip_unclosed_thinking_blocks(text: str) -> str:
    return _UNCLOSED_THINKING_RE.sub("", text)


def _strip_reasoning_paragraphs(text: str) -> str:
    cleaned = text
    for _ in range(50):
        updated = _REASONING_PREFIX_RE.sub("", cleaned, count=1)
        if updated == cleaned:
            break
        cleaned = updated
    return cleaned


def _get_potential_tag_prefix_len(text: str) -> int:
    text_lower = text.lower()
    max_len = 0
    for tag in ["<thinking>", "<redacted_thinking>", "<think>", "`think`", "由于"]:
        for i in range(1, len(tag)):
            prefix = tag[:i]
            if text_lower.endswith(prefix):
                max_len = max(max_len, len(prefix))
    return max_len


def _get_potential_close_tag_prefix_len(text: str) -> int:
    text_lower = text.lower()
    max_len = 0
    for tag in ["</thinking>", "</redacted_thinking>", "</think>", "`/think`", "\\`think`", "结果"]:
        for i in range(1, len(tag)):
            prefix = tag[:i]
            if text_lower.endswith(prefix):
                max_len = max(max_len, len(prefix))
    return max_len


REASONING_CUES = [
    "let me ", "i'll ", "i will ", "first", "hmm", "looking at", "okay",
    "wait", "checks ", "the user ", "how to ", "however", "i need to",
    "the instruction", "reconcil", "re-read", "reread", "that phrasing",
    "it doesn't", "passage ", "important", "- passage", "i'll write",
    "based on", "from the", "to answer", "we are given", "let's extract",
    "chunk ", "we have to", "which one to", "there are ", "this query", "we need to"
]


class ThinkingStreamFilter:
    """Filter streamed tokens that fall inside thinking tags."""

    def __init__(self) -> None:
        self._buffer = ""
        self._in_think = False
        self._emitted_any = False

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
                for tag in ("结果", "结果", "</thinking>", "</redacted_thinking>", "</think>", "`/think`", "\\`think`"):
                    idx = lowered.find(tag)
                    if idx != -1 and (end_idx == -1 or idx < end_idx):
                        end_idx = idx
                        end_tag = tag
                if end_idx == -1:
                    keep = _get_potential_close_tag_prefix_len(self._buffer)
                    if keep > 0:
                        self._buffer = self._buffer[-keep:]
                    else:
                        self._buffer = ""
                    return "".join(out)
                self._buffer = self._buffer[end_idx + len(end_tag or "") :]
                self._in_think = False
                continue

            end_idx = -1
            end_tag = None
            for tag in ("结果", "结果", "</thinking>", "</redacted_thinking>", "</think>", "`/think`", "\\`think`"):
                idx = lowered.find(tag)
                if idx != -1 and (end_idx == -1 or idx < end_idx):
                    end_idx = idx
                    end_tag = tag

            start_idx = -1
            start_tag = None
            for tag in ("由于", "<thinking>", "<redacted_thinking>", "<think>", "`think`"):
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
                keep = _get_potential_tag_prefix_len(self._buffer)
                if keep > 0:
                    emit_len = len(self._buffer) - keep
                    if emit_len > 0:
                        out.append(self._buffer[:emit_len])
                    self._buffer = self._buffer[-keep:]
                else:
                    out.append(self._buffer)
                    self._buffer = ""
                break

            if start_idx > 0:
                out.append(self._buffer[:start_idx])
            self._buffer = self._buffer[start_idx + len(start_tag or "") :]
            self._in_think = True

        output_str = "".join(out)
        if not self._emitted_any:
            lowered_hold = output_str.lower().lstrip()
            starts_with_cue = any(lowered_hold.startswith(cue) for cue in REASONING_CUES)
            if starts_with_cue:
                if "\n\n" in output_str:
                    parts = output_str.split("\n\n")
                    emitted_parts = []
                    for part in parts:
                        part_para = part + "\n\n"
                        if _REASONING_PREFIX_RE.match(part_para):
                            continue
                        else:
                            emitted_parts.append(part)
                    if emitted_parts:
                        output_str = "\n\n".join(emitted_parts)
                        output_str = output_str.lstrip()
                        self._emitted_any = True
                    else:
                        output_str = ""
                else:
                    self._buffer = output_str + self._buffer
                    return ""
            else:
                output_str = output_str.lstrip()
                if output_str:
                    self._emitted_any = True
        return output_str

    def flush(self) -> str:
        if self._in_think:
            self._buffer = ""
            return ""
        leftover = self._buffer
        self._buffer = ""
        if not self._emitted_any and leftover.strip():
            leftover_para = leftover + "\n\n"
            if _REASONING_PREFIX_RE.match(leftover_para) or _REASONING_ONLY_RE.match(leftover):
                return ""
            leftover = leftover.lstrip()
        return leftover
