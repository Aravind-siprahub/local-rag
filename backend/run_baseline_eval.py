"""Run baseline evaluation runner and save baseline results."""
import os
import sys
import json

# Ensure backend directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval.eval_runner import run_evaluation

if __name__ == "__main__":
    run_evaluation()
