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
_FILLER = r"(?:(?:okay|hmm|wait|first|so|now|then|firstly|also|note|but note|also note|please note|note that|however note)[,.:]?\s*|(?:\d+[\.)]\s*))*"

# Meta-subjects and verbs that indicate internal reasoning or narrative structure
_META_SUBJECT_VERBS = [
    r"let me (?:think|try|unpack|check|look|see|analyze|review|figure|acknowledge|tackle|answer|solve)",
    r"let's (?:tackle|unpack|extract|analyze|check|look|see|review|break down|solve|answer)",
    r"i(?:'ll| will| need to| should| must| can) (?:look|review|calculate|answer|check|analyze|read|write|see|try|acknowledge|start|begin|tackle|solve)",
    r"the user (?:is asking|asked|seems|wants|just said|has shared|has asked|wants to know|is saying)",
    r"that phrasing is",
    r"this (?:query|question|prompt) (?:asks|requires|seems)",
    r"this seems like",
    r"to answer this question",
    r"checks (?:requirements|instructions)",
    r"how to (?:reconcile|resolve|handle|proceed)",
    r"the instructions?\s+(?:say|says|state|states|require|requires)",
    r"(?:but\s+)?the instructions?\s+(?:say|says|state|states|require|requires)",
    r"we can (?:write|format|phrase|state) (?:it|this)",
    r"(?:re-read|reread|re-reading) (?:the|this)",
    r"it doesn't directly mention",
    r"important:\s*must not",
    r"we (?:are given|have to|need to|must|must answer|can synthesize|can say)",
    r"given the constraints",
    r"the question asks for",
    r"which (?:one|passage|chunk) to",
    r"(?:in|from|according to|based on)\s+[\"']?[a-zA-Z0-9_\-\.]+\.?[a-zA-Z0-9]*[\"']?\s*(?:section|page|document)?",
    r"this (?:tells us|confirms|shows|indicates)",
    r"putting it together",
    r"therefore[,.]?\s*(?:the\s+)?answer",
    r"the (?:key points|main points|summary):?",
    r"the [a-zA-Z0-9_\-\.]+\s+(?:guide|doc|document)\s+(?:doesn't|does not|gives|provides|mentions|has)",
    r"the (?:document|documents|doc|docs|passage|passages|chunk|chunks|context|excerpt|excerpts)(?:\s+excerpts?)?\s+(?:also|only)?\s*(?:clearly )?(?:states|state|says|say|shows|show|mentions|mention|gives|give|details|detail|discusses|discuss|contains|contain)",
    r"[-*]?\s*(?:passage|chunk|document)\s*\d+[:\s]*(?:is about|is critical|discusses|has a key)",
    r"looking at the (?:provided\s+)?(?:document|documents|passage|passages|chunk|chunks)(?:\s+excerpts|\s+context)?",
    r"okay[,.]?\s+(?:passage|chunk|document)",
    # Natural-language reasoning leakage from multi-document analysis
    r"but wait[,.]?",
    r"hmm[,.]?",
    r"so there(?:'s| is) a conflict",
    r"there(?:'s| is) a conflict",
    r"the (?:first|second|third|fourth|fifth|\d+(?:st|nd|rd|th)?) document (?:says|states|mentions|gives|has|lists|specifies)",
    r"the (?:first|second|third|fourth|fifth|\d+(?:st|nd|rd|th)?) doc (?:says|states|mentions|gives|has|lists|specifies)",
    r"\d+[\.)]\s+the (?:first|second|third|fourth|fifth|\d+(?:st|nd|rd|th)?) (?:document|doc)(?:\s*\([^)]+\))?\s*(?:again|also|only)?\s*(?:mentions|states|says|gives|has|lists|specifies|talks about|refers to|details|outlines|notes|discusses)",
    r"the (?:first|second|third|fourth|fifth|\d+(?:st|nd|rd|th)?) (?:document|doc)(?:\s*\([^)]+\))?\s*(?:again|also|only)?\s*(?:mentions|states|says|gives|has|lists|specifies|talks about|refers to|details|outlines|notes|discusses)",
    r"(?:document|doc) (?:one|two|three|four|five|1|2|3|4|5) (?:says|states|mentions|gives|has|specifies)",
    r"(?:document|doc) a (?:says|states|mentions|gives|has|specifies)",
    r"(?:document|doc) b (?:says|states|mentions|gives|has|specifies)",
    r"comparing (?:the|these) (?:documents|sources|passages)",
    r"cross-referencing",
    r"reconciling (?:the|these|this)",
    r"we are given a question:?",
    r"let's look at the context:?",
    r"\d+[\.)]\s+in\s+[\"']?[a-zA-Z0-9_\-\.]+\.(?:docx|pdf|txt|md|xlsx)[\"']?",
    r"\d+[\.)]\s+the question (?:asks|is|requires|states)",
    r"the term [\"'].*?[\"'] is (?:likely|probably)?\s*(?:referring|pointing) to",
    r"from the context,\s*we see that",
    r"we can interpret that",
    r"the question (?:asks|is|requires|states):?",
    r"the question is (?:specifically )?(?:about|asking for)",
    r"(?:but|however)[,.]?\s+the question is (?:specifically )?(?:about|asking for)",
    r"(?:therefore|so)[,.]?\s*(?:the\s+)?answer (?:should|must|might|will|is) be:?",
    r"(?:[^,\n()]+\)?\s*[,:]?\s*)?we (?:know|note|observe|see|find|gather|infer|learn|deduce|conclude)(?:[:\s]+that)?",
    r"the (?:context|document|documents|passage|passages) (?:tells us|shows|states|mentions|indicates|explains|says|includes|contains|has|provides)(?:\s+that)?:?",
    r"the context (?:includes|contains|has|lists):?",
    r"let's go through the context.*",
    r"\d+[\.)]\s+a section about.*",
    r"\d+[\.)]\s+another section about.*",
    r"\d+[\.)]\s+a systemd service file.*",
    r"\d+[\.)]\s+commands for.*",
    r"\d+[\.)]\s+a curl command.*",
    r"(?:however|but|note)[,.]?\s+the context (?:does not|doesn't) (?:explicitly )?(?:state|specify|mention|contain).*",
    r"the backend service is named.*",
    r"but the context (?:does not|doesn't) specify.*",
    r"the question:\s*.*",
    r"this (?:does not|doesn't) (?:specify|contain|mention)",
    r"this also (?:does not|doesn't) (?:specify|contain|mention)",
    r"\d+[\.)]\s*(?:first|second|third|fourth|fifth|\d+(?:st|nd|rd|th)?) document section",
    r"(?:first|second|third|fourth|fifth|\d+(?:st|nd|rd|th)?) document section",
    r"discrepancy between the (?:two|three|\d+|several)?\s*(?:documents|sources|chunks|passages)",
    r"so there(?:'s| is) a discrepancy.*",
    r"wait[,.]?\s+the\s+other\s+documents?.*",
]

# Match the FULL reasoning sentence up to its terminating punctuation or newline.
_FULL_REASONING_SENTENCE_RE = re.compile(
    r"(?i)^" + _FILLER + r"(?:" + "|".join(_META_SUBJECT_VERBS) + r").*?(?:[.!?:]+(?:\s+|$)|\n+|$)"
)

# For backward compatibility with ThinkingStreamFilter
_REASONING_PREFIX_RE = _FULL_REASONING_SENTENCE_RE

_INLINE_SOURCE_PREFIX_RE = re.compile(
    r"(?i)^(?:"
    r"(?:looking at|based on|according to|from)\s+(?:the\s+)?(?:provided\s+|retrieved\s+)?(?:context|chunks?|passages?|excerpts?|documents?(?:\s+excerpts?)?|sources?|search results?|web search results?)(?:\s+\d+)?(?:\s+(?:above|provided))?(?:[,.:]*)\s*(?:(?:i can see|we can see|we can infer|we can find|it says|it states|it shows|we see|it is clear)(?:\s+that)?\s*)?|"
    r"(?:chunk|passage|document|doc)\s+\d+\s+(?:says|states|shows|mentions|specifies|indicates|contains)(?:\s+that)?\s*"
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


def sanitize_response(text: str | None, question: str | None = None) -> str:
    """Remove thinking blocks and reasoning monologues structurally; return clean answer."""
    if text is None or not isinstance(text, str) or not text.strip():
        return ""

    cleaned = text.strip()
    if cleaned.startswith("{") and "answer" in cleaned:
        try:
            import json
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict) and "answer" in parsed and isinstance(parsed["answer"], str):
                cleaned = parsed["answer"].strip()
        except Exception:
            pass

    # Strip XML blocks safely
    for _ in range(3):
        old_cleaned = cleaned
        for pattern in _THINKING_BLOCK_PATTERNS:
            cleaned = re.sub(pattern, "", cleaned, flags=re.DOTALL | re.IGNORECASE)
        cleaned = _UNOPENED_THINKING_RE.sub("", cleaned)
        cleaned = _strip_unclosed_thinking_blocks(cleaned)
        if cleaned == old_cleaned:
            break

    # Aggressively extract the final answer if the model leaked a concluding phrase anywhere
    concluding_regex = re.compile(
        r"(?:final\s+answer|in\s+summary|in\s+conclusion)[:\s]*",
        re.IGNORECASE
    )
    matches = list(concluding_regex.finditer(cleaned))
    if matches:
        last_match = matches[-1]
        candidate = cleaned[last_match.end():].strip()
        if candidate:
            cleaned = candidate

    # Structural reasoning stripping line-by-line
    lines = cleaned.split("\n")
    cleaned_lines = []

    cleaned_items: list[tuple[str, bool]] = []

    for line in lines:
        stripped_line = line.strip()
        if not stripped_line:
            if cleaned_items and cleaned_items[-1][0] != "":
                cleaned_items.append(("", False))
            continue

        # 1. Strip inline source analysis prefixes first (e.g. "Chunk 1 says ", "From the chunks, we can infer that ")
        had_chunk_prefix = False
        while True:
            m = _INLINE_SOURCE_PREFIX_RE.match(stripped_line)
            if m:
                had_chunk_prefix = True
                stripped_line = stripped_line[m.end():].lstrip()
                if stripped_line and stripped_line[0].islower():
                    stripped_line = stripped_line[0].upper() + stripped_line[1:]
            else:
                break

        # 2. Strip full reasoning sentence prefixes from remaining line
        while True:
            m = _FULL_REASONING_SENTENCE_RE.match(stripped_line)
            if m:
                stripped_line = stripped_line[m.end():].lstrip()
            else:
                break

        # Discard lines that are empty or contain only reasoning monologue fragments
        if not stripped_line or _INCOMPLETE_REASONING_FRAGMENT_RE.match(stripped_line) or _REASONING_ONLY_RE.match(stripped_line):
            continue

        cleaned_items.append((stripped_line, had_chunk_prefix))

    # If some lines had chunk prefixes (working thoughts) and a subsequent line is a direct answer without a chunk prefix,
    # keep only the non-working-thought lines.
    has_direct_answer = any(not had_prefix for _, had_prefix in cleaned_items)
    if has_direct_answer and any(had_prefix for _, had_prefix in cleaned_items):
        final_lines = [text for text, had_prefix in cleaned_items if not had_prefix]
    else:
        final_lines = [text for text, _ in cleaned_items]

    # Trailing self-talk truncation: Stop at the first line that introduces self-reflection or meta-commentary
    truncated_lines = []
    self_talk_prefixes = (
        "let's write", "we are to be", "also, note", "note that", "the key is",
        "additionally, note", "therefore,", "document:", "1. document:", "2. document:",
        "3. document:", "4. document:", "5. document:", "let's go through",
        "the document says", "so the frontend", "so the backend", "the key is in"
    )
    for line in final_lines:
        lower_line = line.strip().lower()
        if truncated_lines and any(lower_line.startswith(prefix) for prefix in self_talk_prefixes):
            break
        truncated_lines.append(line)
    final_lines = truncated_lines

    cleaned = "\n".join(final_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    # Safety Guard: Handle meta monologues about missing information ONLY if no direct answer is present
    if cleaned:
        lowered_c = cleaned.lower()
        has_factual_content = any(
            kw in lowered_c
            for kw in (
                "frontend:", "backend:", "react", "fastapi", "next.js", "express",
                "django", "flask", "pm2", "nginx", "vue", "angular", "svelte",
                "python", "node", "frontend", "backend", "vite", "typescript",
                "javascript", "postgres", "postgresql", "framework", "technology"
            )
        )
        if not has_factual_content:
            missing_phrases = (
                "the requested information is not found",
                "the requested information isn't found",
                "aren't listed in any of the documents",
                "aren't mentioned anywhere in the context",
                "don't explicitly state what frontend and backend",
                "doesn't explicitly state what frontend and backend",
                "the documents don't explicitly state",
                "the documents do not explicitly state",
                "the context provided doesn't",
                "the context provided does not",
                "doesn't specify the actual technologies",
                "don't specify the actual technologies",
                "aren't specified in any of the documents",
            )
            if any(phrase in lowered_c for phrase in missing_phrases):
                return "The requested information is not found in the documents."

    # Echo prevention: if the output is just the user's question, discard it
    if question and isinstance(question, str) and question.strip():
        q_norm = re.sub(r"[^\w\s]", "", question.lower()).strip()
        c_norm = re.sub(r"[^\w\s]", "", cleaned.lower()).strip()
        if c_norm == q_norm or (len(c_norm) > 5 and c_norm in q_norm and len(c_norm) >= len(q_norm) * 0.8):
            return ""

    return cleaned


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
    "chunk ", "we have to", "which one to", "there are ", "this query", "we need to",
    # qwen3-specific leaked monologue patterns
    "this is under", "so i think", "we output", "alternatively", "but note",
    "so the answer", "the answer is correct", "the rule says", "the problem says",
    "the first word", "but that's okay", "so the first", "word must", "words.",
    "under 50", "so this", "final answer:", "note:", "answer:", "output:",
    "the third passage", "the second passage", "also, the",
    # Multi-document reasoning leakage patterns
    "but wait", "hmm,", "so there's a conflict", "there's a conflict",
    "the first document", "the second document", "the third document",
    "the fourth document", "the fifth document",
    "document one says", "document two says", "document a says", "document b says",
    "comparing the documents", "cross-referencing",
]

# Regex to detect standalone reasoning-only lines (lines with NO actual document content)
_REASONING_LINE_RE = re.compile(
    r"(?i)^\s*("
    r"\d+\s+words?\.?|"  # "14 words."
    r"this is under \d+\.?|"  # "This is under 50."
    r"so i think.*|"  # "So I think this is the answer."
    r"we output.*|"  # "We output:"
    r"but note.*|"  # "But note: ..."
    r"but wait[,.]?.*|"  # "But wait, the second document says..."
    r"wait,?\s*(?:the\s+)?(?:other\s+)?(?:documents?|chunks?|passages?|sources?|context).*|"  # "Wait, the other documents..."
    r"hmm[,.]?.*|"  # "Hmm, there's a conflict..."
    r"so there(?:'s| is) a conflict.*|"  # "So there's a conflict..."
    r"there(?:'s| is) a conflict.*|"  # "There's a conflict here."
    r"the (?:first|second|third|fourth|fifth) document (?:says|states|mentions|gives|has|lists|specifies).*|"  # "The second document says..."
    r"however,? the (instruction|problem|rule).*|"  # "However, the instruction says..."
    r"alternatively,?.*|"  # "Alternatively, we can write..."
    r"the (first|second|third|fourth|last) (word|passage|document|sentence).*|"  # "The first word must..."
    r"but that'?s okay.*|"  # "But that's okay because..."
    r"so (the first|the answer|this).*|"  # "So the first word is 'N'."
    r"(final answer|note|answer|output):\s*.*|"  # "Final answer: ..."
    r"also,? the (third|second|first|last) passage.*|"  # "Also, the third passage gives..."
    r"the (answer|rule|problem) (is|says|says that).*"  # "The answer starts with 'N'."
    r")\s*$"
)


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
        # Strip any lines that are pure reasoning/meta-commentary (e.g. "14 words.", "So I think...")
        if output_str:
            filtered_lines = [
                line for line in output_str.split("\n")
                if not _REASONING_LINE_RE.match(line)
            ]
            output_str = "\n".join(filtered_lines)
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
