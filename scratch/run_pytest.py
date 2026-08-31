import pytest
import sys

if __name__ == "__main__":
    sys.exit(pytest.main([
        "backend/tests/test_agent_router.py",
        "backend/tests/test_rag_accuracy_regression.py",
        "backend/tests/test_rag_enterprise_suite.py",
        "backend/tests/test_web_search_parser.py",
        "-v"
    ]))
