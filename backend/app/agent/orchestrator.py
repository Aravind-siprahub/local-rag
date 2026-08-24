"""Central Agent Orchestrator managing execution loop, tool dispatch, self-correction, and telemetry."""
from __future__ import annotations

import logging
import time
import uuid
from typing import Any
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.state import AgentState, AgentStatus, EvidenceItem, ToolResult
from app.agent.planner import Planner
from app.agent.model_router import ModelRouter, TaskRole
from app.agent.verifier import VerificationAgent
from app.agent.tools.rag_tool import DocumentRAGTool
from app.agent.tools.web_tool import WebSearchTool
from app.agent.tools.vision_tool import VisionTool
from app.agent.tools.memory_tool import MemoryTool
from app.agent.tools.base import ToolInput, ToolOutput
from app.core.config import get_settings
from app.prompting.builder import PromptBuilder
from app.llm.ollama_client import get_global_ollama_client
from app.llm.sanitize import sanitize_response

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """Production Agent Orchestrator for multi-tool execution, verification, and self-correction."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.planner = Planner()
        self.verifier = VerificationAgent()
        self.prompt_builder = PromptBuilder()
        self.llm_client = get_global_ollama_client()

        # Initialize modular tools
        self.rag_tool = DocumentRAGTool(session)
        self.web_tool = WebSearchTool()
        self.vision_tool = VisionTool()
        self.memory_tool = MemoryTool(session)

    async def run(
        self,
        query: str,
        *,
        session_id: uuid.UUID | None = None,
        user_id: uuid.UUID | None = None,
        document_id: uuid.UUID | None = None,
        document_version_id: uuid.UUID | None = None,
        image_bytes: bytes | None = None,
        image_name: str | None = None,
        request_id: str | None = None,
        document_titles: list[str] | None = None,
    ) -> AgentState:
        """Run the complete Agent loop for a user query."""
        state = AgentState(user_query=query, trace_id=request_id or str(uuid.uuid4()))
        start_mono = time.monotonic()
        settings = get_settings()
        max_iterations = getattr(settings, "AGENT_MAX_ITERATIONS", 4)

        try:
            # 1. Memory retrieval step
            if session_id:
                mem_input = ToolInput(
                    query=query,
                    parameters={"session_id": session_id, "history_limit": 4},
                )
                mem_out = await self.memory_tool.execute(mem_input)
                if mem_out.success:
                    state.conversation_context = mem_out.data.get("history", [])
                    state.working_memory = mem_out.data.get("working_memory")

            # 2. Planning step
            state.transition_to(AgentStatus.PLANNING)
            plan_start = time.monotonic()
            await self.planner.create_plan(
                state,
                document_titles=document_titles,
                has_image=bool(image_bytes),
                has_doc_filter=bool(document_id or document_version_id),
            )
            state.metrics.planning_time_ms = int((time.monotonic() - plan_start) * 1000)

            # 3. Execution & Self-Correction Loop
            while state.retry_count < max_iterations and state.status not in (AgentStatus.COMPLETED, AgentStatus.FAILED):
                state.metrics.iterations += 1

                if state.current_step_index >= len(state.plan):
                    # No more planned steps, break to answer synthesis
                    break

                current_step = state.plan[state.current_step_index]
                target_tool = current_step.target_tool
                state.selected_tool = target_tool

                # Tool Execution
                tool_start = time.monotonic()
                tool_output: ToolOutput

                if target_tool == "vision_analysis" and image_bytes:
                    state.transition_to(AgentStatus.ANALYZING)
                    v_model = ModelRouter.get_model(TaskRole.VISION)
                    state.selected_model = v_model
                    state.add_model_usage(v_model)
                    v_input = ToolInput(
                        query=query,
                        parameters={"image_bytes": image_bytes, "image_name": image_name or "image.png"},
                    )
                    tool_output = await self.vision_tool.execute(v_input)

                elif target_tool == "document_rag":
                    state.transition_to(AgentStatus.RETRIEVING)
                    rag_model = ModelRouter.get_model(TaskRole.RAG_REASONING)
                    state.selected_model = rag_model
                    state.add_model_usage(rag_model)
                    rag_input = ToolInput(
                        query=query,
                        parameters={
                            "user_id": user_id,
                            "document_id": document_id,
                            "document_version_id": document_version_id,
                        },
                    )
                    tool_output = await self.rag_tool.execute(rag_input)

                elif target_tool == "web_search":
                    state.transition_to(AgentStatus.SEARCHING)
                    web_input = ToolInput(
                        query=query,
                        parameters={"request_id": state.trace_id},
                    )
                    tool_output = await self.web_tool.execute(web_input)
                else:
                    logger.warning("[AGENT] Unknown tool target %s. Skipping step.", target_tool)
                    state.current_step_index += 1
                    continue

                tool_time_ms = int((time.monotonic() - tool_start) * 1000)
                state.metrics.tool_execution_time_ms += tool_time_ms

                # Record Tool Output
                state.record_tool_result(
                    ToolResult(
                        tool_name=target_tool,
                        input_data={"query": query},
                        output_data=tool_output.data,
                        execution_time_ms=tool_time_ms,
                        success=tool_output.success,
                        error_message=tool_output.error,
                    )
                )

                if tool_output.success:
                    # Ingest Evidence
                    new_evidence = tool_output.data.get("evidence", [])
                    for ev in new_evidence:
                        state.evidence.append(
                            EvidenceItem(
                                source_type=ev.get("source_type", target_tool),
                                content=ev.get("content", ""),
                                source_name=ev.get("source_name", "Source"),
                                relevance_score=ev.get("relevance_score", 0.8),
                                metadata=ev,
                            )
                        )
                    if "chunks" in tool_output.data:
                        state.retrieved_documents.extend(tool_output.data["chunks"])

                    current_step.is_completed = True
                    state.current_step_index += 1

                else:
                    # Tool Failed -> Trigger Self-Correction
                    state.transition_to(AgentStatus.RETRYING)
                    state.retry_count += 1
                    logger.warning(
                        "[AGENT SELF-CORRECTION] Step %d tool=%s failed: %s. Retrying...",
                        current_step.step_number, target_tool, tool_output.error
                    )

                    # Fallback Strategy: If Document RAG failed/empty, try Web Search
                    if target_tool == "document_rag":
                        current_step.target_tool = "web_search"
                        current_step.description = "Fallback live web search after RAG tool retry."
                    else:
                        state.current_step_index += 1

            # 4. Verification Step
            state.transition_to(AgentStatus.VERIFYING)
            ver_start = time.monotonic()
            v_outcome = self.verifier.verify_evidence(state)
            state.verification_result = v_outcome
            state.metrics.verification_time_ms = int((time.monotonic() - ver_start) * 1000)

            # If evidence is rejected and we can retry, switch strategy
            if not v_outcome.is_valid and v_outcome.requires_retry and state.retry_count < max_iterations:
                state.transition_to(AgentStatus.RETRYING)
                state.retry_count += 1
                logger.info("[AGENT SELF-CORRECTION] Evidence rejected. Retrying with Web Search fallback.")
                web_input = ToolInput(query=query, parameters={"request_id": state.trace_id})
                web_out = await self.web_tool.execute(web_input)
                if web_out.success:
                    for ev in web_out.data.get("evidence", []):
                        state.evidence.append(
                            EvidenceItem(
                                source_type="web",
                                content=ev.get("content", ""),
                                source_name=ev.get("source_name", "Web"),
                                relevance_score=ev.get("relevance_score", 0.75),
                            )
                        )

            # 5. Answering Step
            state.transition_to(AgentStatus.ANSWERING)
            final_model = ModelRouter.get_model(TaskRole.FINAL_ANSWER)
            state.selected_model = final_model
            state.add_model_usage(final_model)

            llm_start = time.monotonic()

            if not state.evidence and not any(r.success for r in state.tool_results):
                state.final_answer = "I could not find this information in the uploaded documents."
            else:
                prompt = self.prompt_builder.build(
                    query,
                    state.retrieved_documents,
                    chat_history=state.conversation_context,
                    working_memory_summary=state.working_memory,
                    is_vision=bool(image_bytes),
                )

                _images_payload = [image_bytes] if image_bytes else None
                resp = await self.llm_client.generate(
                    prompt.system_prompt,
                    prompt.user_prompt,
                    num_predict=settings.OLLAMA_NUM_PREDICT,
                    images=_images_payload,
                    model=ModelRouter.get_model(TaskRole.VISION) if image_bytes else final_model,
                )

                clean_ans = sanitize_response(resp.answer, question=query)

                # Verify Final Answer
                ans_outcome = self.verifier.verify_final_answer(clean_ans, state)
                if not ans_outcome.is_valid and ans_outcome.requires_retry and state.retry_count < max_iterations:
                    logger.warning("[AGENT] Final answer verification failed: %s. Performing re-generation.", ans_outcome.reason)
                    corr_prompt = prompt.user_prompt + "\n\nCRITICAL MANDATE: Answer using ONLY verified facts directly present in the context."
                    resp_corr = await self.llm_client.generate(
                        prompt.system_prompt,
                        corr_prompt,
                        num_predict=settings.OLLAMA_NUM_PREDICT,
                        model=final_model,
                    )
                    clean_ans = sanitize_response(resp_corr.answer, question=query)

                state.final_answer = clean_ans

            state.metrics.llm_generation_time_ms = int((time.monotonic() - llm_start) * 1000)
            state.metrics.total_latency_ms = int((time.monotonic() - start_mono) * 1000)
            state.transition_to(AgentStatus.COMPLETED)

            logger.info(
                "[AGENT ORCHESTRATOR COMPLETED] trace_id=%s iterations=%d total_ms=%d models=%s tools=%s",
                state.trace_id, state.metrics.iterations, state.metrics.total_latency_ms,
                state.metrics.models_used, state.metrics.tools_used
            )
            return state

        except Exception as exc:
            state.metrics.total_latency_ms = int((time.monotonic() - start_mono) * 1000)
            state.transition_to(AgentStatus.FAILED)
            state.final_answer = "I encountered an error processing your request. Please try again."
            logger.exception("[AGENT ORCHESTRATOR FAILED] trace_id=%s error=%s", state.trace_id, exc)
            return state
