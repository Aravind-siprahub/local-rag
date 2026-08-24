"""Modular tool framework base interfaces."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ToolMetadata:
    name: str
    description: str
    version: str = "1.0.0"
    requires_network: bool = False
    requires_gpu: bool = False


@dataclass
class ToolInput:
    query: str
    parameters: dict[str, Any] = field(default_factory=dict)
    state_context: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolOutput:
    success: bool
    data: dict[str, Any]
    error: str | None = None
    execution_time_ms: int = 0


class Tool(ABC):
    """Abstract base class for all Agent tools."""

    def __init__(self, metadata: ToolMetadata) -> None:
        self.metadata = metadata

    @abstractmethod
    async def execute(self, tool_input: ToolInput) -> ToolOutput:
        """Execute the tool's core logic asynchronously."""
        pass
