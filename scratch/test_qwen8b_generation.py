"""Test actual Ollama HTTP generation call with qwen3:8b."""
import asyncio
import sys
from pathlib import Path

backend_path = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(backend_path))

from app.llm.ollama_client import OllamaLLMClient

async def test_qwen8b():
    client = OllamaLLMClient(model="qwen3:8b")
    print("[TEST] Sending test prompt to qwen3:8b...")
    resp = await client.generate(
        system_prompt="You are a helpful AI assistant.",
        user_prompt="Say 'Qwen3 8B model is fully functional and ready' in one short line.",
        num_predict=50,
    )
    print(f"[SUCCESS] Response from {resp.model_name}: {resp.answer.strip()}")

if __name__ == "__main__":
    asyncio.run(test_qwen8b())
