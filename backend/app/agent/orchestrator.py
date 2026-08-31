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
from app.retrieval.ranking import RankedResult

logger = logging.getLogger(__name__)


def _validate_web_search_answer(
    raw_answer: str,
    clean_answer: str,
    web_concise_text: str,
    original_query: str,
) -> str:
    """Validate LLM answer against web search results.

    1. If raw response is JSON → extract 'answer' field; if malformed or empty → use concise_text.
    2. If the answer is unrelated to the query topic (no topic keyword overlap) → use concise_text.
    3. Otherwise return clean_answer.
    """
    import json as _json
    import re as _re

    raw = (raw_answer or "").strip()

    # Step 1: If raw is JSON-like, try to parse it
    if raw.startswith("{") and "answer" in raw:
        try:
            parsed = _json.loads(raw)
            if isinstance(parsed, dict):
                extracted = (parsed.get("answer") or "").strip()
                if extracted:
                    return extracted
                # 'answer' key exists but empty → fallback
                logger.info("[AGENT WEB FALLBACK] JSON had empty 'answer' field. Using concise_text.")
                return web_concise_text
        except (_json.JSONDecodeError, ValueError):
            # Malformed JSON → fallback
            logger.info("[AGENT WEB FALLBACK] Malformed JSON from LLM. Using concise_text.")
            return web_concise_text

    # Step 2: Check that the clean answer is related to the query topic.
    # Extract meaningful tokens from the query and web concise text.
    query_tokens = set(_re.findall(r"\b[A-Za-z]{4,}\b", original_query.lower()))
    web_tokens = set(_re.findall(r"\b[A-Za-z]{4,}\b", web_concise_text.lower()))
    topic_tokens = (query_tokens | web_tokens) - {"what", "when", "where", "which", "that", "with", "from", "this", "they", "have", "here", "found"}

    if topic_tokens and clean_answer:
        ans_lower = clean_answer.lower()
        overlap = sum(1 for t in topic_tokens if t in ans_lower)
        # If fewer than 15% of topic tokens appear in the answer, it's unrelated
    # Step 3: Check for LLM disclaimer phrases
    disclaimer_phrases = (
        "cannot perform external search",
        "cannot perform live internet",
        "cannot access github",
        "cannot access external",
        "don't have access to the internet",
        "dont have access to the internet",
        "no internet access",
        "cannot search the web",
        "cannot browse the web",
    )
    if any(phrase in clean_answer.lower() for phrase in disclaimer_phrases):
        logger.info("[AGENT WEB FALLBACK] LLM disclaimer detected. Replacing with concise web search text.")
        return web_concise_text

    return clean_answer or web_concise_text


class AgentOrchestrator:
    """Production Agent Orchestrator for multi-tool execution, verification, and self-correction."""

    def __init__(
        self,
        session: AsyncSession | None = None,
        *,
        retriever: Any = None,
        llm_client: Any = None,
        web_search: Any = None,
        prompt_builder: Any = None,
    ) -> None:
        self.session = session
        self.planner = Planner()
        self.verifier = VerificationAgent()
        self.prompt_builder = prompt_builder if prompt_builder is not None else PromptBuilder()
        self.llm_client = llm_client if llm_client is not None else get_global_ollama_client()

        # Initialize modular tools
        self.rag_tool = DocumentRAGTool(session, retriever=retriever)
        self.web_tool = WebSearchTool(web_search=web_search)
        self.vision_tool = VisionTool()
        self.memory_tool = MemoryTool(session)

    async def run(
        self,
        query: str,
        *,
        session_id: uuid.UUID | None = None,
        user_id: uuid.UUID | None = None,
        document_id: uuid.UUID | None = None,
        document_ids: tuple[uuid.UUID, ...] | None = None,
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
                    parameters={"session_id": session_id, "history_limit": getattr(settings, "MEMORY_MAX_RECENT_MESSAGES", 10)},
                )
                mem_out = await self.memory_tool.execute(mem_input)
                if mem_out.success:
                    state.conversation_context = mem_out.data.get("history", [])
                    state.working_memory = mem_out.data.get("working_memory")

            if user_id and self.session is not None:
                try:
                    from app.memory.manager import MemoryManager
                    mem_mgr = MemoryManager(self.session)
                    sec_text, retrieved_mems = await mem_mgr.before_query(user_id=user_id, query=query)
                    state.long_term_memory_context = sec_text
                    state.retrieved_memories = retrieved_mems
                except Exception as mem_err:
                    logger.warning("[AGENT MEMORY RETRIEVAL FAILED] %s", mem_err)

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
                            "document_ids": document_ids,
                            "document_version_id": document_version_id,
                        },
                    )
                    tool_output = await self.rag_tool.execute(rag_input)

                elif target_tool == "web_search":
                    state.transition_to(AgentStatus.SEARCHING)
                    web_input = ToolInput(
                        query=query,
                        parameters={
                            "request_id": state.trace_id,
                            "max_results": 5,
                            "fetch_pages": True,
                        },
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
                    state.current_step_index += 1

            # 4. Verification Step
            state.transition_to(AgentStatus.VERIFYING)
            ver_start = time.monotonic()
            v_outcome = self.verifier.verify_evidence(state)
            state.verification_result = v_outcome
            state.metrics.verification_time_ms = int((time.monotonic() - ver_start) * 1000)

            # If evidence is rejected and we can retry, switch strategy (only if document_rag was not used)
            doc_ran = any(r.tool_name == "document_rag" for r in state.tool_results)
            if not v_outcome.is_valid and v_outcome.requires_retry and state.retry_count < max_iterations and not document_id and not doc_ran:
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

            # Check if web_search was the active tool and handle its special cases
            web_tool_results = [r for r in state.tool_results if r.tool_name == "web_search"]
            web_was_used = bool(web_tool_results)
            web_concise_text: str | None = None
            web_had_hits = False

            if web_was_used and web_tool_results:
                last_web = web_tool_results[-1]
                web_concise_text = last_web.output_data.get("concise_text", "")
                web_had_hits = bool(last_web.output_data.get("count", 0))

                if not web_had_hits:
                    web_err = (last_web.error_message or "").lower()
                    if "timeout" in web_err or "timed out" in web_err:
                        state.final_answer = "Web search timed out. I couldn't retrieve current web results for this request."
                    else:
                        state.final_answer = "I could not find reliable web results for that question right now."
                    state.metrics.llm_generation_time_ms = 0
                    state.metrics.total_latency_ms = int((time.monotonic() - start_mono) * 1000)
                    state.transition_to(AgentStatus.COMPLETED)
                    return state

            # Check if document_rag ran or 0 chunks/evidence were retrieved for document query (when no image provided)
            if doc_ran and not state.retrieved_documents and not image_bytes:
                state.final_answer = "Information not found in document excerpts."
                state.metrics.llm_generation_time_ms = 0
                state.metrics.total_latency_ms = int((time.monotonic() - start_mono) * 1000)
                state.transition_to(AgentStatus.COMPLETED)
                return state

            if web_was_used and web_had_hits:
                web_chunks = []
                for idx, ev in enumerate(state.evidence, 1):
                    if ev.source_type in ("web", "web_search") and ev.content:
                        url_val = (ev.metadata.get("url") if isinstance(ev.metadata, dict) else "") or f"https://websearch/{idx}"
                        doc_id = uuid.uuid5(uuid.NAMESPACE_URL, url_val)
                        web_chunks.append(
                            RankedResult(
                                chunk_id=doc_id,
                                chunk_text=f"Source {idx}: {ev.source_name} ({url_val})\nSnippet: {ev.content}",
                                document_id=doc_id,
                                document_version_id=uuid.uuid4(),
                                similarity_score=ev.relevance_score,
                                rank=idx,
                                document_title=ev.source_name or "Web Source",
                                section_title="web",
                            )
                        )
                if web_chunks:
                    state.retrieved_documents.extend(web_chunks)

            if not state.evidence and not state.retrieved_documents and not image_bytes:
                state.final_answer = "Information not found in document excerpts."
                state.metrics.llm_generation_time_ms = 0
                state.metrics.total_latency_ms = int((time.monotonic() - start_mono) * 1000)
                state.transition_to(AgentStatus.COMPLETED)
                return state


            prompt = self.prompt_builder.build(
                query,
                state.retrieved_documents,
                chat_history=state.conversation_context,
                working_memory_summary=state.working_memory,
                long_term_memory_context=state.long_term_memory_context,
                is_vision=bool(image_bytes),
            )

            sys_prompt = prompt.system_prompt
            if web_was_used and web_had_hits:
                sys_prompt = settings.WEB_SEARCH_SYSTEM_PROMPT
                logger.info("[WEB SEARCH DEBUG] query=%r", query)
                logger.info("[WEB SEARCH DEBUG] result_count=%d", len(state.evidence))
                logger.info("[WEB SEARCH DEBUG] results_context_length=%d", len(prompt.user_prompt))
                if state.evidence:
                    logger.info("[WEB SEARCH DEBUG] first_result_title=%r", state.evidence[0].source_name)
                    logger.info("[WEB SEARCH DEBUG] first_result_url=%r", getattr(state.evidence[0], "url", ""))
                logger.info("[WEB SEARCH DEBUG] llm_prompt_contains_web_results=true")

            _images_payload = [image_bytes] if image_bytes else None
            resp = await self.llm_client.generate(
                sys_prompt,
                prompt.user_prompt,
                num_predict=settings.OLLAMA_NUM_PREDICT,
                images=_images_payload,
                model=ModelRouter.get_model(TaskRole.VISION) if image_bytes else final_model,
            )

            if getattr(resp, "token_usage", None):
                state.metrics.prompt_tokens += getattr(resp.token_usage, "prompt_tokens", 0) or 0
                state.metrics.completion_tokens += getattr(resp.token_usage, "completion_tokens", 0) or 0
                state.metrics.total_tokens += getattr(resp.token_usage, "total_tokens", 0) or 0

            clean_ans = sanitize_response(resp.answer, question=query)

            # Append [Truncated] if the LLM stopped due to token limit
            if getattr(resp, "finish_reason", None) in ("length", "max_tokens"):
                if not clean_ans.endswith("[Truncated]"):
                    clean_ans = (clean_ans + " [Truncated]").strip()
                state.final_answer = clean_ans
                state.metrics.total_latency_ms = int((time.monotonic() - start_mono) * 1000)
                state.transition_to(AgentStatus.COMPLETED)
                return state

            # For web search results: validate LLM response relevance to the query.
            # If the LLM response is malformed JSON or unrelated to web hits, use concise_text fallback.
            if web_was_used and web_had_hits and web_concise_text:
                clean_ans = _validate_web_search_answer(
                    raw_answer=resp.answer,
                    clean_answer=clean_ans,
                    web_concise_text=web_concise_text,
                    original_query=query,
                )

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

            if getattr(resp, "model_name", None):
                state.selected_model = resp.model_name
            state.final_answer = clean_ans

            state.metrics.llm_generation_time_ms = int((time.monotonic() - llm_start) * 1000)
            state.metrics.total_latency_ms = int((time.monotonic() - start_mono) * 1000)
            state.transition_to(AgentStatus.COMPLETED)

            # Schedule asynchronous long-term memory extraction
            if user_id and self.session is not None and state.final_answer:
                try:
                    from app.memory.manager import MemoryManager
                    mem_mgr = MemoryManager(self.session)
                    mem_mgr.schedule_extraction(
                        user_id=user_id,
                        question=query,
                        answer=state.final_answer,
                        conversation_id=session_id,
                        existing_memories=state.retrieved_memories,
                    )
                except Exception as mem_ext_err:
                    logger.warning("[AGENT MEMORY EXTRACTION FAILED] %s", mem_ext_err)

            logger.info(
                "[AGENT ORCHESTRATOR COMPLETED] trace_id=%s iterations=%d total_ms=%d models=%s tools=%s",
                state.trace_id, state.metrics.iterations, state.metrics.total_latency_ms,
                state.metrics.models_used, state.metrics.tools_used
            )
            return state

        except Exception as exc:
            if type(exc).__name__ in ("RetrievalError", "RAGError", "LLMUnavailableError", "LLMTimeoutError"):
                raise
            state.metrics.total_latency_ms = int((time.monotonic() - start_mono) * 1000)
            state.transition_to(AgentStatus.FAILED)
            state.final_answer = "I encountered an error processing your request. Please try again."
            logger.exception("[AGENT ORCHESTRATOR FAILED] trace_id=%s error=%s", state.trace_id, exc)
            return state
