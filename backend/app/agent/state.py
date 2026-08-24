"""Agent state tracking data structures for production Agentic AI Architecture."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AgentStatus(str, Enum):
    IDLE = "IDLE"
    PLANNING = "PLANNING"
    RETRIEVING = "RETRIEVING"
    SEARCHING = "SEARCHING"
    ANALYZING = "ANALYZING"
    VERIFYING = "VERIFYING"
    RETRYING = "RETRYING"
    ANSWERING = "ANSWERING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass
class PlanStep:
    step_number: int
    description: str
    target_tool: str
    expected_outcome: str
    is_completed: bool = False


@dataclass
class ToolResult:
    tool_name: str
    input_data: dict[str, Any]
    output_data: dict[str, Any]
    execution_time_ms: int
    success: bool
    error_message: str | None = None


@dataclass
class EvidenceItem:
    source_type: str  # "document", "web", "vision", "memory"
    content: str
    source_name: str
    relevance_score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationOutcome:
    is_valid: bool
    reason: str | None = None
    relevance_score: float = 0.0
    hallucination_detected: bool = False
    contradiction_detected: bool = False
    requires_retry: bool = False


@dataclass
class ExecutionMetrics:
    start_time: float = field(default_factory=time.monotonic)
    total_latency_ms: int = 0
    planning_time_ms: int = 0
    tool_execution_time_ms: int = 0
    llm_generation_time_ms: int = 0
    verification_time_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    iterations: int = 0
    models_used: list[str] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)


@dataclass
class AgentState:
    user_query: str
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    conversation_context: list[dict[str, str]] = field(default_factory=list)
    working_memory: str | None = None
    intent: str | None = None
    plan: list[PlanStep] = field(default_factory=list)
    current_step_index: int = 0
    selected_tool: str | None = None
    selected_model: str | None = None
    tool_results: list[ToolResult] = field(default_factory=list)
    retrieved_documents: list[Any] = field(default_factory=list)
    evidence: list[EvidenceItem] = field(default_factory=list)
    verification_result: VerificationOutcome | None = None
    retry_count: int = 0
    final_answer: str | None = None
    citations: list[Any] = field(default_factory=list)
    status: AgentStatus = AgentStatus.IDLE
    metrics: ExecutionMetrics = field(default_factory=ExecutionMetrics)

    def transition_to(self, new_status: AgentStatus) -> None:
        """Update agent execution status."""
        self.status = new_status

    def record_tool_result(self, result: ToolResult) -> None:
        """Record output from a tool invocation."""
        self.tool_results.append(result)
        if result.tool_name not in self.metrics.tools_used:
            self.metrics.tools_used.append(result.tool_name)

    def add_model_usage(self, model_name: str) -> None:
        """Track model usage across steps."""
        if model_name and model_name not in self.metrics.models_used:
            self.metrics.models_used.append(model_name)
