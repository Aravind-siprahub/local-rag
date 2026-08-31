"""Script to verify ModelRouter role assignments against config and .env settings."""
import sys
from pathlib import Path

# Add backend directory to sys.path
backend_path = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(backend_path))

from app.core.config import get_settings
from app.agent.model_router import ModelRouter, TaskRole

def run_routing_test():
    settings = get_settings()
    print("=== EFFECTIVE MODEL ROUTING CONFIGURATION ===")
    print(f"MODEL_ROUTER_CLASSIFY:   {settings.MODEL_ROUTER_CLASSIFY}")
    print(f"MODEL_QUERY_REWRITE:     {settings.MODEL_QUERY_REWRITE}")
    print(f"MODEL_RAG_REASONING:     {settings.MODEL_RAG_REASONING}")
    print(f"MODEL_COMPLEX_REASONING: {settings.MODEL_COMPLEX_REASONING}")
    print(f"MODEL_FINAL_ANSWER:      {settings.MODEL_FINAL_ANSWER}")
    print(f"OLLAMA_VISION_MODEL:     {settings.OLLAMA_VISION_MODEL}")
    print(f"EMBEDDING_MODEL:         {settings.EMBEDDING_MODEL}")
    print("=============================================\n")

    print("=== MODEL ROUTER ASSIGNMENTS PER TASK ROLE ===")
    roles = [
        TaskRole.CLASSIFY,
        TaskRole.QUERY_REWRITE,
        TaskRole.RAG_REASONING,
        TaskRole.COMPLEX_REASONING,
        TaskRole.FINAL_ANSWER,
        TaskRole.VISION,
        TaskRole.EMBEDDING,
    ]

    for r in roles:
        selected = ModelRouter.get_model(r)
        print(f"Role: {r.value:<20} -> Selected Model: {selected}")

if __name__ == "__main__":
    run_routing_test()
