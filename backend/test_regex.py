import re
import sys

_REASONING_INLINE_RE = re.compile(
    r"(?i)^\s*("
    r"let me (?:think|analyze|check|look|see|consider|review|examine|re-read|reread|read|unpack|try)(?:\s+(?:about\s+this(?:\s+carefully)?|this(?:\s+problem)?))?\s*[^.]*\.{0,3}\s*|"
    r"i(?:'ll| will) (?:think|analyze|check|look|formulate|consider|re-read|reread)[^.]*\.{0,3}\s*|"
    r"first[,.]?\s+(?:i'll|i will|let me|the user|a simple|i need)[^.]*\.{0,3}\s*|"
    r"hmm\b(?:[,.]?\s+(?:this seems(?: like a financial(?: question)?)?|the user|the)[^.]*\.{0,3})?\s*|"
    r"looking at (?:the )?(?:passage|chunk|excerpt|document|context|sources|excerpts)s?(?:\s+excerpts)?(?:[,.]\s*(?:i\s+can\s+see\s+that)?)?[^.]*\.{0,3}\s*|"
    r"okay[,.]?\s+(?:let me|let's|the user|so i need|i need)[^.]*\.{0,3}\s*|"
    r"let's (?:unpack|tackle) (?:this|the)(?:\s+problem)?[,.]?(?:\s+the\s+user\s+is\s+asking\s+for(?:\s+the)?(?:[^.]*))?\.{0,3}\s*|"
    r"wait\b[^.]*\.{0,3}\s*"
    r")(.*)$"
)

def sanitize_response(text: str) -> str:
    lines = text.split('\n')
    last_reasoning_idx = -1
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        m = _REASONING_INLINE_RE.match(stripped)
        if m:
            rest_of_line = m.group(2).strip()
            if not rest_of_line:
                last_reasoning_idx = idx
            else:
                lines[idx] = rest_of_line
                last_reasoning_idx = idx - 1
                
    cleaned_lines = []
    for idx, line in enumerate(lines):
        stripped_line = line.strip()
        if not stripped_line:
            if idx > last_reasoning_idx:
                cleaned_lines.append('')
            continue

        if idx <= last_reasoning_idx:
            continue
            
        modified = True
        while modified:
            modified = False
            
        if not stripped_line:
            continue
            
        cleaned_lines.append(stripped_line)
    return '\n'.join(cleaned_lines).strip()

tests = [
    ('Let me think about this carefully.\n\nThe answer is 42.', 'The answer is 42.'),
    ('Okay, the user asked about revenue.\n\nHmm, this seems like a financial question.\n\nRevenue was $5M.', 'Revenue was $5M.'),
    ('Okay, let\'s tackle this problem. The user is asking for the total revenue.', ''),
    ('That phrasing is a bit odd - they probably meant \'2 planets or 3 planets\'.\n\nOkay, let me unpack this. The user seems confused about planetary counts.\n\nEarth is the 3rd planet from the Sun.', 'Earth is the 3rd planet from the Sun.'),
    ('Looking at Chunk 1, I can see that Talk to My Data is a platform.', 'Talk to My Data is a platform.'),
    ('First, note that Talk to My Data is a platform with three key features.', 'First, note that Talk to My Data is a platform with three key features.')
]

if __name__ == "__main__":
    failed = False
    for t, expected in tests:
        res = sanitize_response(t)
        print(f'Input: {t}\nResult: {res}\nExpected: {expected}\nMatch: {res == expected}\n')
        if res != expected:
            failed = True
            
    sys.exit(1 if failed else 0)
