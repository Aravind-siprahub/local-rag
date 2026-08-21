import sys
import os
import json
import asyncio
from unittest.mock import patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.session import AsyncSessionLocal
from app.rag.service import RAGService
from app.rag.intent_router import classify
from app.repositories.user_repository import UserRepository
from app.services.chat_session_service import ChatSessionService
from app.llm.ollama_client import get_global_ollama_client

async def evaluate_correctness_with_llm(question: str, expected: str, actual: str) -> bool:
    """Use LLM-as-a-judge to evaluate if actual answer contains the expected semantic fact."""
    if expected.lower() in actual.lower():
        return True
        
    client = get_global_ollama_client()
    prompt = (
        "You are an impartial grader evaluating an AI assistant's answer.\n"
        f"Question: {question}\n"
        f"Expected Fact: {expected}\n"
        f"Actual Answer: {actual}\n\n"
        "Does the Actual Answer semantically state or contain the Expected Fact? "
        "Answer with exactly 'YES' or 'NO'."
    )
    
    resp = await client.generate(
        system_prompt="You are a strict boolean grader.",
        user_prompt=prompt,
        num_predict=10
    )
    
    return "yes" in resp.answer.lower()

async def evaluate():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "benchmark_data.json")
    
    try:
        with open(data_path, "r", encoding="utf-8") as f:
            benchmark_data = json.load(f)
    except Exception as e:
        print(f"Failed to load benchmark data: {e}")
        return

    print(f"Loaded {len(benchmark_data)} questions for deep evaluation.\n")
    correct = 0

    async with AsyncSessionLocal() as db_session:
        user_repo = UserRepository(db_session)
        users = await user_repo.list_active()
        if not users:
            print("No active users found.")
            return
        user_id = users[0].id

        session_service = ChatSessionService(db_session)
        chat_session = await session_service.create_session(user_id=user_id, title="Deep Eval Session")
        
        rag_service = RAGService(db_session)

        for idx, item in enumerate(benchmark_data, 1):
            q = item["question"]
            expected = item["expected"]
            
            print(f"[{idx}/{len(benchmark_data)}] Q: {q}")
            print(f"    Expected: {expected}")
            
            # --- Hooks for tracing ---
            trace = {}
            
            original_classify = classify
            def patched_classify(*args, **kwargs):
                res = original_classify(*args, **kwargs)
                trace["intent"] = res.name
                print(f"    [TRACE] Detected Intent: {res.name}")
                return res
            
            original_retrieve = rag_service.retriever.retrieve
            async def patched_retrieve(*args, **kwargs):
                res = await original_retrieve(*args, **kwargs)
                trace["retrieval_count"] = len(res)
                print(f"    [TRACE] Retrieval count: {len(res)}")
                for i, r in enumerate(res[:3]):
                    print(f"      -> Rank {i+1} [Score: {r.similarity_score:.3f}]: {r.chunk_text[:80]}...")
                return res
                
            original_generate = rag_service.llm_client.generate
            async def patched_generate(system_prompt, user_prompt, *args, **kwargs):
                trace["ollama_query"] = user_prompt
                print(f"    [TRACE] Final Context Sent to Ollama:")
                print("      " + repr(user_prompt[:300]) + " ... [truncated]")
                res = await original_generate(system_prompt, user_prompt, *args, **kwargs)
                return res

            with patch("app.rag.service.classify", side_effect=patched_classify), \
                 patch.object(rag_service.retriever, "retrieve", side_effect=patched_retrieve), \
                 patch.object(rag_service.llm_client, "generate", side_effect=patched_generate):
                 
                try:
                    resp = await rag_service.ask(session_id=chat_session.id, question=q)
                    answer = resp.answer
                except Exception as e:
                    answer = f"ERROR: {str(e)}"

            print(f"    Actual Answer: {answer}")
            
            is_correct = await evaluate_correctness_with_llm(q, expected, answer)
            if is_correct:
                print("    ✅ PASS")
                correct += 1
            else:
                print("    ❌ FAIL")
                
            print("-" * 60)

    score = (correct / len(benchmark_data)) * 100
    print(f"\nFinal Semantic Accuracy Score: {correct}/{len(benchmark_data)} ({score:.1f}%)")

if __name__ == "__main__":
    if sys.platform == "win32" and sys.version_info < (3, 14):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(evaluate())
