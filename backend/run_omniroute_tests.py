import subprocess
import sys

def main():
    res = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_omniroute.py", "tests/test_llm_providers.py", "-v"],
        capture_output=True,
        text=True
    )
    print("RETURN CODE:", res.returncode)
    print("--- STDOUT ---")
    print(res.stdout)
    print("--- STDERR ---")
    print(res.stderr)
    with open("omniroute_test_results.txt", "w", encoding="utf-8") as f:
        f.write(f"RETURN CODE: {res.returncode}\n")
        f.write("--- STDOUT ---\n")
        f.write(res.stdout)
        f.write("\n--- STDERR ---\n")
        f.write(res.stderr)

if __name__ == "__main__":
    main()
