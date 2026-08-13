import subprocess
import sys

try:
    result = subprocess.run([sys.executable, "-m", "pytest", "tests/test_llm_sanitize.py", "tests/test_sanitize.py", "-v"], capture_output=True, text=True)
    with open("test_results.txt", "w", encoding="utf-8") as f:
        f.write(result.stdout)
        f.write("\nSTDERR:\n")
        f.write(result.stderr)
except Exception as e:
    with open("test_results.txt", "w", encoding="utf-8") as f:
        f.write(str(e))
