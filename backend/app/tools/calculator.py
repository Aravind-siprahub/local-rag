"""Safe arithmetic calculator tool for Agent Router v1."""
from __future__ import annotations

import ast
import operator
import re
from dataclasses import dataclass


class CalculatorError(Exception):
    """Raised when an expression cannot be evaluated safely."""


@dataclass(frozen=True)
class CalculatorResult:
    value: float
    display: str
    expression: str


_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_PERCENT_OF = re.compile(
    r"(?i)(\d+(?:\.\d+)?)\s*%\s*of\s*(-?\d+(?:\.\d+)?)"
)
_PERCENT_WORD = re.compile(
    r"(?i)(\d+(?:\.\d+)?)\s*percent\s+of\s*(-?\d+(?:\.\d+)?)"
)
_EXPRESSION_CHUNK = re.compile(
    r"(-?\d+(?:\.\d+)?(?:\s*[\+\-\*\/]\s*-?\d+(?:\.\d+)?)+)"
)


def calculate(question: str) -> CalculatorResult:
    """Evaluate a simple arithmetic question safely via AST."""
    text = (question or "").strip()
    if not text:
        raise CalculatorError("Empty calculator input.")

    percent = _PERCENT_OF.search(text) or _PERCENT_WORD.search(text)
    if percent:
        left = float(percent.group(1))
        right = float(percent.group(2))
        value = (left / 100.0) * right
        expression = f"{left}% of {right}"
        return CalculatorResult(value=value, display=_format_number(value), expression=expression)

    match = _EXPRESSION_CHUNK.search(text.replace("×", "*").replace("÷", "/"))
    if not match:
        # Fall back to stripping non-math wording
        cleaned = re.sub(
            r"(?i)^\s*(what\s+is|calculate|compute|evaluate)\s+",
            "",
            text,
        )
        cleaned = cleaned.strip().rstrip("?").strip()
        if not cleaned:
            raise CalculatorError("No arithmetic expression found.")
        expression = cleaned
    else:
        expression = match.group(1).strip()

    value = _eval_ast(expression)
    return CalculatorResult(value=value, display=_format_number(value), expression=expression)


def _format_number(value: float) -> str:
    if value == int(value):
        return str(int(value))
    return f"{value:.10g}"


def _eval_ast(expression: str) -> float:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise CalculatorError("Invalid arithmetic expression.") from exc

    try:
        result = _eval_node(tree.body)
    except (TypeError, ZeroDivisionError, OverflowError) as exc:
        raise CalculatorError("Unable to evaluate expression.") from exc

    if not isinstance(result, (int, float)):
        raise CalculatorError("Expression did not produce a number.")
    return float(result)


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return float(_UNARY_OPS[type(node.op)](_eval_node(node.operand)))
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        return float(_BIN_OPS[type(node.op)](left, right))
    raise CalculatorError("Unsupported expression.")
