import subprocess
with open("c:/Users/ARAVIND/Desktop/local-rag/backend/scratch/test_out.txt", "w") as f:
    subprocess.run(["pytest", "tests/test_sanitize.py", "-v"], stdout=f, stderr=f, cwd="c:/Users/ARAVIND/Desktop/local-rag/backend")
