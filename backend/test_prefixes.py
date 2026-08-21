import re
import sys

# Test patterns with [^.!?]* instead of .*?
_MONOLOGUE_PREFIX_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"^first[,.]?\s+(?:i'll|i will|let me|the user|a simple)[^.!?]*[.!?]{0,3}\s*",
        r"^okay[,.]?\s+(?:let me|let's|the user|so i need|i need|let's tackle)[^.!?]*[.!?]{0,3}\s*",
        r"^hmm\b(?:[,.]?\s+(?:this seems|the user|the)[^.!?]*[.!?]{0,3})?\s*",
        r"^let's (?:unpack|tackle) (?:this|the)[^.!?]*[.!?]{0,3}\s*",
        r"^looking at (?:the )?(?:passage|chunk|excerpt|document|context|sources|excerpts)s?(?:\s+excerpts)?[^.!?]*[.!?]{0,3}\s*",
        r"^let me (?:think|analyze|check|look|see|consider|review|examine|re-read|reread|read|unpack|try)[^.!?]*[.!?]{0,3}\s*",
        r"^wait\b[^.!?]*[.!?]{0,3}\s*"
    ]
]

tests = [
    ("Let me think about this carefully. The answer is 42.", "The answer is 42."),
    ("Hmm, this seems like a financial question. Revenue was $5M.", "Revenue was $5M."),
    ("Okay, let's tackle this problem. Total revenue.", "Total revenue."),
    ("Let's unpack this. Earth is the third planet.", "Earth is the third planet."),
    ("Looking at chunk 1, I can see that Earth is the third planet.", "Earth is the third planet."),
    ("Wait, I need to check something. The answer is 5.", "The answer is 5.")
]

if __name__ == "__main__":
    failed = False
    for t, expected in tests:
        res = t
        for pat in _MONOLOGUE_PREFIX_PATTERNS:
            res = pat.sub("", res)
        print(f"Input: {t}\nResult: {res}\nExpected: {expected}\nMatch: {res == expected}\n")
        if res != expected:
            failed = True
            
    sys.exit(1 if failed else 0)
