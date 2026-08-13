import pytest
import sys

# Custom patch for devnull issue on Windows if needed, or just redirect stdout/stderr
import os
import io

os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"

try:
    exit_code = pytest.main(["-v", "tests/test_llm_sanitize.py", "tests/test_sanitize.py"])
except Exception as e:
    print(f"Error: {e}")
    exit_code = 1

print(f"EXIT_CODE: {exit_code}")
sys.exit(exit_code)
