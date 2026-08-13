import re

_REASONING_INLINE_RE = re.compile(
    r"(?i)^\s*("
    r"let me (?:think|analyze|check|look|see|consider|review|examine|re-read|reread|read|unpack|try).*?|"
    r"i(?:'ll| will) (?:think|analyze|check|look|formulate|consider|re-read|reread).*?|"
    r"first[,.]?\s+(?:i'll|i will|let me|the user|a simple|note that|i need).*?|"
    r"hmm\b(?:[,.]?\s+(?:this seems|the user|the).*?)?|"
    r"looking at (?:the )?(?:passage|chunk|excerpt|document|context|sources|excerpts)s?(?:\s+excerpts)?.*?|"
    r"okay[,.]?\s+(?:let me|let's|the user|so i need|i need).*?|"
    r"let's (?:unpack|tackle) (?:this|the).*?|"
    r"wait\b.*?"
    r")[ \t]*\.{0,3}[ \t]*(.*)$"
)

tests = [
    ('Let me think about this. The answer is 42.', 'The answer is 42.'),
    ('Hmm, this seems like a financial analysis... Revenue was $5M.', 'Revenue was $5M.'),
    ('Okay, let\'s tackle this problem... Total revenue.', 'Total revenue.'),
    ('Let\'s unpack this. Earth is the third planet from the Sun.', 'Earth is the third planet from the Sun.'),
    ('Looking at Chunk 1, I can see... Talk to My Data is a platform.', 'Talk to My Data is a platform.'),
    ('First, I\'ll acknowledge the question... Hello! How can I assist you today?', 'Hello! How can I assist you today?')
]

for t, expected in tests:
    m = _REASONING_INLINE_RE.match(t)
    res = m.group(2).strip() if m else None
    print(f'Input: {t}\nResult: {res}\nExpected: {expected}\nMatch: {m is not None}\n')
