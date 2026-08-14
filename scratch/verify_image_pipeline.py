import asyncio
import logging
import sys
from pathlib import Path

import os
backend_dir = Path(__file__).resolve().parent.parent / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))
os.chdir(backend_dir)

from app.core.config import get_settings
from app.llm.ollama_client import OllamaLLMClient

# Setup logging to console
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
logger = logging.getLogger("verify_image_pipeline")

# 67-byte minimal transparent 1x1 PNG byte string (fallback)
MINIMAL_PNG_BYTES = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\rIDATx\x9cc`\x00\x01\x00\x00\xff\xff\x03\x00\x00\x06\x00\x05\x57-\x0f\xa0\x00\x00\x00\x00IEND\xaeB`\x82'

def get_test_image_bytes() -> bytes:
    try:
        from PIL import Image
        import io
        img = Image.new("RGB", (100, 100), color="red")
        output = io.BytesIO()
        img.save(output, format="PNG")
        return output.getvalue()
    except Exception as e:
        logger.warning("Could not generate standard PIL PNG, falling back to minimal 1x1 PNG: %s", e)
        return MINIMAL_PNG_BYTES

async def main():
    settings = get_settings()
    vision_model = settings.OLLAMA_VISION_MODEL
    print("\n=======================================================")
    print("IMAGE PIPELINE CONFIGURATION & END-TO-END VERIFICATION")
    print("=======================================================")
    print(f"Ollama Vision Model Configured: {vision_model}")
    print(f"Ollama Chat Model Configured: {settings.OLLAMA_MODEL}")
    print(f"Ollama Base URL: {settings.OLLAMA_BASE_URL}")
    print("=======================================================\n")

    # 1. Initialize Ollama Client
    client = OllamaLLMClient(model=vision_model)

    # 2. Test supports_vision function
    print("--- 1. Testing supports_vision() capability check ---")
    has_vision = await client.supports_vision(model=vision_model)
    print(f"Result for {vision_model}: {has_vision}")
    if not has_vision:
        print(f"❌ FAIL: {vision_model} was not detected as supporting vision.")
        return
    print(f"✅ PASS: {vision_model} correctly identified as supporting vision.\n")

    # 3. Test Direct Generate with Image
    print("--- 2. Testing direct Ollama client generation with 100x100 Red PNG ---")
    system_prompt = "You are a helpful assistant. Describe what you see in the provided image."
    user_prompt = "What color is this image? Keep it extremely short (1 sentence)."
    
    try:
        response = await client.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            images=[get_test_image_bytes()],
            model=vision_model
        )
        print("✅ PASS: Direct vision model generation succeeded!")
        print(f"Model used: {response.model_name}")
        print(f"Response: {response.answer.strip()}")
        print(f"Latency: {response.latency_ms if hasattr(response, 'latency_ms') else 'N/A'} ms\n")
    except Exception as e:
        print(f"❌ FAIL: Ollama direct generation failed: {e}")
        return

    print("Verification complete. All programmatic checks passed!\n")

if __name__ == "__main__":
    asyncio.run(main())
