import re

_REASONING_PREFIX_RE = re.compile(
    r"(?i)^\s*(?:"
    r"let me (?:think|analyze|check|look|see|consider|review|examine|re-read|reread|read)|"
    r"i(?:'ll| will) (?:think|analyze|check|look|formulate|consider|re-read|reread)|"
    r"first[,.]?\s+(?:i'll|i will|let me|the user|a simple)|"
    r"hmm\b|"
    r"looking at (?:the )?(?:passage|chunk|excerpt|document|context|sources|excerpts)s?(?:\s+excerpts)?\s*[:.!?]|"
    r"okay[,.]?\s+(?:the user|let me|let's|so i need|i need)|"
    r"wait\b|"
    r"checks (?:requirements|instructions)|"
    r"the user (?:seems|asks|asked|is asking|just said|wants|wanted)|"
    r"how to (?:reconcile|resolve|handle|proceed)|"
    r"however[,.]?\s+(?:the instruction|the prompt|the system|the requirement)|"
    r"i need to respond|"
    r"the instruction (?:says|states|requires)|"
    r"reconcil(?:e|ing) (?:the|these)|"
    r"(?:re-read|reread|re-reading) (?:the|the prompt|the question)|"
    r"(?:let me|i should) (?:re-read|reread|review|check) (?:the|the prompt|the question)|"
    r"that phrasing is a bit odd|"
    r"it doesn't directly mention|"
    r"passage \d+[:]?|"
    r"important:\s*must not|"
    r"-\s*passage\s*\d+|"
    r"i'll\s+write\s+this|"
    r"we (?:are|have|need|must) (?:given|to|need|answer|should)\b|"
    r"let's (?:extract|analyze|check|look|determine|see|use|think|review)\b|"
    r"chunk\s*\d+[:]?\s*$|"
    r"which\s+(?:one|passage|chunk|document)\s+to\s+(?:use|choose|select)\b|"
    r"there (?:are|is) \d+ (?:passages|chunks|documents)\b|"
    r"this (?:query|question|prompt) (?:asks|requires)\b"
    r")[^\n]*(?:\n+|$)"
)

lines = [
    "We are given a user query about SipraOne.",
    "Let's extract information from the chunks.",
    "Chunk 1 says SipraOne is deployed on Azure VM.",
    "We have to be careful.",
    "Which one to use?",
    "Wait...",
    "SipraOne was deployed on an Azure Ubuntu VM."
]

with open("c:/Users/ARAVIND/Desktop/local-rag/backend/scratch/re_test_out.txt", "w") as f:
    for i, line in enumerate(lines):
        m = _REASONING_PREFIX_RE.match(line)
        f.write(f"{i}: {bool(m)}\n")
