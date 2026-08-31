import subprocess
import sys

res = subprocess.run([sys.executable, "-m", "pytest", "tests", "-v"], capture_output=True, text=True)
print("STDOUT:")
print(res.stdout[-3000:])
print("STDERR:")
print(res.stderr[-2000:])
print("EXIT CODE:", res.returncode)
