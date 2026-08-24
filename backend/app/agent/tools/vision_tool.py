"""Vision Tool for multimodal image analysis, chart understanding, and visual evidence extraction."""
from __future__ import annotations

import logging
import time
from typing import Any

from app.agent.tools.base import Tool, ToolInput, ToolMetadata, ToolOutput
from app.core.config import get_settings
from app.llm.ollama_client import get_global_ollama_client

logger = logging.getLogger(__name__)


class VisionTool(Tool):
    """Modular tool for multimodal image analysis and visual evidence extraction."""

    def __init__(self) -> None:
        super().__init__(
            ToolMetadata(
                name="vision_analysis",
                description="Analyzes uploaded images, charts, screenshots, and diagrams using multimodal vision models.",
                version="1.0.0",
                requires_gpu=True,
            )
        )
        self.llm_client = get_global_ollama_client()

    async def execute(self, tool_input: ToolInput) -> ToolOutput:
        start_mono = time.monotonic()
        query = tool_input.query.strip() or "Describe and analyze the contents of this image in detail."
        params = tool_input.parameters

        image_bytes: bytes | None = params.get("image_bytes")
        image_name: str = params.get("image_name", "uploaded_image.png")

        if not image_bytes:
            return ToolOutput(
                success=False,
                data={"analysis": "", "evidence": []},
                error="No image binary payload provided to VisionTool.",
                execution_time_ms=0,
            )

        settings = get_settings()
        vision_model = getattr(settings, "OLLAMA_VISION_MODEL", "qwen3-vl:4b")

        system_prompt = (
            "You are an expert multimodal visual analyst. Analyze the provided image, chart, screenshot, "
            "or document image in clear detail. Extract factual evidence, numbers, labels, diagrams, "
            "and text present in the image relevant to the user query."
        )
        user_prompt = f"User Question: {query}\n\nAnalyze the image and provide clear, grounded findings."

        try:
            response = await self.llm_client.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=vision_model,
                images=[image_bytes],
                num_predict=300,
            )

            analysis_text = (response.answer or "").strip()
            evidence_item = {
                "source_type": "vision",
                "source_name": image_name,
                "content": analysis_text,
                "relevance_score": 0.90,
            }

            duration_ms = int((time.monotonic() - start_mono) * 1000)
            logger.info(
                "[VISION TOOL SUCCESS] model=%s image_size=%d duration_ms=%d",
                vision_model, len(image_bytes), duration_ms
            )

            return ToolOutput(
                success=True,
                data={
                    "analysis": analysis_text,
                    "evidence": [evidence_item],
                    "image_name": image_name,
                    "model_used": vision_model,
                },
                execution_time_ms=duration_ms,
            )

        except Exception as exc:
            duration_ms = int((time.monotonic() - start_mono) * 1000)
            logger.exception("[VISION TOOL FAILED] model=%s error=%s", vision_model, exc)
            return ToolOutput(
                success=False,
                data={"analysis": "", "evidence": []},
                error=str(exc),
                execution_time_ms=duration_ms,
            )
