import sys
import os
import pytest

sys.path.insert(0, r"c:\Users\ARAVIND\Desktop\local-rag\backend")

if __name__ == "__main__":
    os.chdir(r"c:\Users\ARAVIND\Desktop\local-rag\backend")
    exit_code = pytest.main(["tests", "-v", "--tb=short"])
    print(f"\nPytest Exit Code: {exit_code}")
