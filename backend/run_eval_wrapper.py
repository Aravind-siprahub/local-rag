import sys
import os

def run_eval():
    sys.argv = ['evaluate_accuracy.py']
    
    # We will just import the evaluate function and run it inside asyncio
    import asyncio
    from scripts.evaluate_accuracy import evaluate
    
    # Redirect stdout to a file to avoid the cortex execution error
    with open('eval_out.txt', 'w', encoding='utf-8') as f:
        original_stdout = sys.stdout
        sys.stdout = f
        try:
            asyncio.run(evaluate())
        finally:
            sys.stdout = original_stdout

if __name__ == '__main__':
    run_eval()
