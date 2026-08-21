import sys
import pytest

if __name__ == "__main__":
    args = sys.argv[1:] if len(sys.argv) > 1 else ["tests/", "-v"]
    ret = pytest.main(args)
    sys.exit(ret)
