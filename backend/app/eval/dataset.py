"""Evaluation dataset structures and loader."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EvalTestCase:
    """Represents a single evaluation fixture question and its expected targets."""

    id: str | int
    question: str
    category: str = "direct_factual"
    expected_document: str | list[str] | None = None
    expected_version: str | list[str] | None = None
    expected_answer: str | None = None
    expected_key_facts: list[str] = field(default_factory=list)
    expected_citation: str | list[str] | None = None
    is_negative: bool = False
    required_keywords: list[str] = field(default_factory=list)
    forbidden_keywords: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any], default_id: int = 1) -> EvalTestCase:
        """Instantiate EvalTestCase from dict representation."""
        exp_doc = data.get("expected_document") or data.get("expected_documents")
        exp_ver = data.get("expected_version") or data.get("expected_versions")
        exp_facts = data.get("expected_key_facts") or data.get("required_keywords") or []
        exp_cite = data.get("expected_citation") or data.get("expected_citations") or exp_doc
        category = data.get("category", "direct_factual")
        is_neg = data.get("is_negative", False) or category == "negative questions" or category == "no_answer"

        return cls(
            id=data.get("id", default_id),
            question=data.get("question", ""),
            category=category,
            expected_document=exp_doc,
            expected_version=exp_ver,
            expected_answer=data.get("expected_answer"),
            expected_key_facts=list(exp_facts) if isinstance(exp_facts, (list, tuple)) else [str(exp_facts)],
            expected_citation=exp_cite,
            is_negative=is_neg,
            required_keywords=list(data.get("required_keywords", [])),
            forbidden_keywords=list(data.get("forbidden_keywords", [])),
        )


@dataclass
class EvalDataset:
    """Collection of evaluation test cases loaded from a dataset file."""

    test_cases: list[EvalTestCase] = field(default_factory=list)

    @classmethod
    def load_from_file(cls, filepath: str | Path) -> EvalDataset:
        """Load dataset from JSON file path."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Evaluation dataset file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        items = raw_data.get("test_cases", raw_data) if isinstance(raw_data, dict) else raw_data
        if not isinstance(items, list):
            raise ValueError(f"Invalid dataset format in {path}: expected a list of test cases")

        test_cases = [EvalTestCase.from_dict(item, idx + 1) for idx, item in enumerate(items)]
        return cls(test_cases=test_cases)
