import subprocess
import sys

try:
    result = subprocess.run([sys.executable, "-m", "pytest", "tests/test_rag_service.py", "-v"], capture_output=True, text=True)
    with open("test_results.txt", "w", encoding="utf-8") as f:
        f.write("--- STDOUT ---\n")
        f.write(result.stdout)
        f.write("\n--- STDERR ---\n")
        f.write(result.stderr)
except Exception as e:
    with open("test_results.txt", "w", encoding="utf-8") as f:
        f.write(str(e))
