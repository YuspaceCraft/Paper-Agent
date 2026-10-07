"""Final-answer format and quality checks.

Tool outputs already have a strict envelope.  This module applies a smaller,
domain-neutral gate to the final answer so obviously broken output can trigger
one repair pass instead of being treated as a successful completion.
"""

from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass, field
from typing import Literal


ExpectedFormat = Literal["text", "json", "csv"]


@dataclass(frozen=True)
class OutputValidation:
    """Metadata-only validation result for one final answer."""

    passed: bool
    expected_format: ExpectedFormat
    issues: tuple[dict, ...] = field(default_factory=tuple)
    repair_hint: str = ""
    safe_response: str = ""

    @property
    def codes(self) -> tuple[str, ...]:
        return tuple(str(item.get("code") or "") for item in self.issues)

    def trace_view(self) -> dict:
        return {
            "passed": self.passed,
            "expected_format": self.expected_format,
            "issues": list(self.issues),
        }


_JSON_REQUEST_RE = re.compile(
    r"(?:返回|输出|以|用|only|return|output|format|as|in)\s*"
    r"(?:一个\s*)?json(?:\s*对象|\s*格式)?",
    re.IGNORECASE,
)
_CSV_REQUEST_RE = re.compile(
    r"(?:csv|逗号分隔|comma[-\s]?separated)",
    re.IGNORECASE,
)
_TRUNCATION_ENDINGS = set("，,、：:；;（(【[{[")
_REPLACEMENT_RE = re.compile("[\ufffd]")


def expected_output_format(query: str) -> ExpectedFormat:
    text = str(query or "")
    if _JSON_REQUEST_RE.search(text):
        return "json"
    if _CSV_REQUEST_RE.search(text):
        return "csv"
    return "text"


def _parse_json_answer(text: str) -> tuple[bool, str]:
    candidate = text.strip()
    if candidate.startswith("```"):
        first_newline = candidate.find("\n")
        last_fence = candidate.rfind("```")
        if first_newline >= 0 and last_fence > first_newline:
            candidate = candidate[first_newline + 1:last_fence].strip()
    try:
        json.loads(candidate)
    except (TypeError, ValueError):
        return False, "answer is not a valid standalone JSON value"
    return True, ""


def _parse_csv_answer(text: str) -> tuple[bool, str]:
    try:
        rows = list(csv.reader(io.StringIO(text.strip())))
    except (csv.Error, TypeError):
        return False, "answer is not valid CSV"
    rows = [row for row in rows if any(cell.strip() for cell in row)]
    if not rows:
        return False, "answer contains no CSV rows"
    widths = {len(row) for row in rows}
    if len(widths) > 1:
        return False, "CSV rows have inconsistent column counts"
    return True, ""


def validate_final_output(query: str, answer: str) -> OutputValidation:
    """Validate format, encoding and obvious incompleteness of a final answer."""
    text = str(answer or "").strip()
    expected = expected_output_format(query)
    issues: list[dict] = []

    if not text:
        issues.append({"code": "EMPTY_OUTPUT", "detail": "answer is empty"})
    else:
        replacement_count = len(_REPLACEMENT_RE.findall(text))
        if replacement_count:
            issues.append({
                "code": "CORRUPTED_OUTPUT",
                "detail": f"answer contains {replacement_count} replacement characters",
            })
        if len(text) >= 40 and text[-1] in _TRUNCATION_ENDINGS:
            issues.append({
                "code": "TRUNCATED_OUTPUT",
                "detail": "answer ends with an incomplete delimiter or punctuation",
            })

    if expected == "json" and text:
        ok, detail = _parse_json_answer(text)
        if not ok:
            issues.append({"code": "FORMAT_JSON_INVALID", "detail": detail})
    elif expected == "csv" and text:
        ok, detail = _parse_csv_answer(text)
        if not ok:
            issues.append({"code": "FORMAT_CSV_INVALID", "detail": detail})

    from .content_safety import assess_content_safety

    safety = assess_content_safety(text)
    if safety.blocked:
        issues.append({
            "code": "POLICY_BLOCKED",
            "detail": safety.category,
        })

    if not issues:
        return OutputValidation(True, expected)

    if safety.blocked:
        return OutputValidation(
            False,
            expected,
            tuple(issues),
            "Return a safe, high-level refusal with prevention or alternatives.",
            safety.safe_response,
        )

    hints = [
        "Return a complete final answer.",
        "Do not add prose outside the requested structured format.",
        "If the answer is incomplete, state the known limitation instead of faking it.",
    ]
    if expected == "json":
        hints.append("Return one valid JSON value only.")
    elif expected == "csv":
        hints.append("Return valid CSV with a consistent number of columns per row.")
    return OutputValidation(
        False,
        expected,
        tuple(issues),
        " ".join(hints),
    )
