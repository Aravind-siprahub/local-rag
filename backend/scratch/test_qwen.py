import asyncio
import uuid
import sys
import os

sys.path.insert(0, os.path.abspath("."))
from app.llm.ollama_client import OllamaLLMClient
from app.prompting.builder import PromptBuilder
from app.retrieval.ranking import RankedResult

async def main():
    client = OllamaLLMClient()
    builder = PromptBuilder()
    
    chunks = [
        RankedResult(
            chunk_id=uuid.uuid4(),
            chunk_text="The SipraOne frontend uses port 4173.",
            document_id=uuid.uuid4(),
            document_version_id=uuid.uuid4(),
            similarity_score=0.9,
            rank=1,
            document_title="Deployment Guide",
            section_title="Frontend"
        )
    ]
    prompt = builder.build("What port does the Sipraone frontend use?", chunks)
    print("----- SYSTEM PROMPT -----")
    print(prompt.system_prompt)
    print("----- USER PROMPT -----")
    print(prompt.user_prompt)
    print("----- CALLING LLM -----")
    response = await client.generate(prompt.system_prompt, prompt.user_prompt)
    print("----- RAW RESPONSE -----")
    print(repr(response.answer))

if __name__ == "__main__":
    asyncio.run(main())
