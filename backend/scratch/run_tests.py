import sys
import pytest

if __name__ == "__main__":
    ret = pytest.main([
        "tests/test_intent_router.py",
        "tests/test_document_parser_suite.py",
        "tests/test_rag_pipeline_e2e.py",
        "tests/test_memory_conversation.py",
        "tests/test_memory_store.py",
        "tests/test_memory_extractor.py",
        "tests/test_memory_context_builder.py",
        "tests/test_memory_rag_integration.py",
        "-v",
    ])
    sys.exit(ret)
