text = 'Here is the answer. \u7ed3\u679c'
lowered = text.lower().strip()
tags = ('\u7531\u4e8e', '\u7ed3\u679c', '<thinking>', '</thinking>', '<redacted_thinking>')
result = any(tag in lowered for tag in tags)
with open('debug_output.txt', 'w', encoding='utf-8') as f:
    f.write(f'lowered: {repr(lowered)}\n')
    f.write(f'tags: {[repr(t) for t in tags]}\n')
    f.write(f'any tag in lowered: {result}\n')
    for tag in tags:
        f.write(f'  {repr(tag)} in lowered: {tag in lowered}\n')