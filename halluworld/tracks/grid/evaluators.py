from __future__ import annotations

import json
import re

from halluworld.evaluator import Evaluator, EvalResult
from halluworld.lm.base import LMResponse
from halluworld.probe import ProbeResult


def _parse_yes_no(text: str) -> bool | None:
    """Return True/False if text contains yes/no, else None."""
    t = text.lower()
    if re.search(r"\byes\b", t):
        return True
    if re.search(r"\bno\b", t):
        return False
    return None


class PresenceEvaluator(Evaluator):
    """Exact yes/no match for PresenceProbe responses.

    Scoring:
        - 1.0: predicted matches ground_truth
        - 0.0: predicted contradicts ground_truth
        - 0.5: response ambiguous (treated as incorrect for `correct` flag)
    """

    def evaluate(self, response: LMResponse, probe_result: ProbeResult) -> EvalResult:
        predicted = _parse_yes_no(response.text)
        gt: bool = probe_result.ground_truth

        if predicted is None:
            correct = False
            score = 0.0
            parse_note = "ambiguous_response"
        elif predicted == gt:
            correct = True
            score = 1.0
            parse_note = "exact_match"
        else:
            correct = False
            score = 0.0
            parse_note = "mismatch"

        return EvalResult(
            correct=correct,
            score=score,
            predicted=predicted,
            ground_truth=gt,
            probe_type=probe_result.probe_type,
            lm_text=response.text,
            details={**probe_result.metadata, "parse_note": parse_note},
        )


class LocationEvaluator(Evaluator):
    """Parses and scores LocationProbe responses.

    Tries to extract steps_ahead and lateral from free-form text.

    Scoring:
        - steps_ahead correct AND lateral correct → 1.0
        - Only steps_ahead correct                → 0.5
        - Only lateral correct                    → 0.3
        - Neither correct                         → 0.0
        - ground_truth is None (probe was skipped)→ 0.0, correct=False
    """

    def evaluate(self, response: LMResponse, probe_result: ProbeResult) -> EvalResult:
        gt = probe_result.ground_truth

        if gt is None:
            return EvalResult(
                correct=False, score=0.0,
                predicted=None, ground_truth=None,
                probe_type=probe_result.probe_type,
                lm_text=response.text,
                details={**probe_result.metadata, "parse_note": "probe_skipped"},
            )

        text = response.text
        pred_ahead = None
        pred_lateral = None
        parse_note = "ok"

        # Primary: parse JSON response
        try:
            # Strip markdown code fences if present
            clean = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`")
            # Find first {...} block
            m = re.search(r"\{[^}]+\}", clean)
            if m:
                parsed = json.loads(m.group())
                pred_ahead = int(parsed["steps_ahead"])
                pred_lateral = int(parsed["lateral"])
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            parse_note = "json_parse_failed"

        ahead_correct = (pred_ahead == gt["steps_ahead"])
        lateral_correct = (pred_lateral == gt["lateral"])

        if ahead_correct and lateral_correct:
            score = 1.0
        elif ahead_correct:
            score = 0.5
        elif lateral_correct:
            score = 0.3
        else:
            score = 0.0

        correct = (score == 1.0)

        return EvalResult(
            correct=correct,
            score=score,
            predicted={"steps_ahead": pred_ahead, "lateral": pred_lateral},
            ground_truth=gt,
            probe_type=probe_result.probe_type,
            lm_text=response.text,
            details={
                **probe_result.metadata,
                "ahead_correct": ahead_correct,
                "lateral_correct": lateral_correct,
                "parse_note": parse_note,
            },
        )


class DynamicsEvaluator(Evaluator):
    """Parses and scores DynamicsProbe responses.

    Expected LM format::

        row=3, col=7

    Scoring:
        - row correct AND col correct → 1.0
        - Only row correct             → 0.5
        - Only col correct             → 0.5
        - Neither correct              → 0.0

    Also records ``wind_error``: whether the model gave the naive (no-wind)
    answer instead of the wind-corrected ground truth.  This lets us separate
    "didn't model wind" from "parsed the position wrong".
    """

    def evaluate(self, response: LMResponse, probe_result: ProbeResult) -> EvalResult:
        gt = probe_result.ground_truth  # {"row": int, "col": int}
        text = response.text

        pred_row: int | None = None
        pred_col: int | None = None
        parse_note = "ok"

        # Try "row=<int>, col=<int>" / "row: <int>" / flexible spacing
        row_m = re.search(r"row\s*[=:]\s*(-?\d+)", text, re.IGNORECASE)
        col_m = re.search(r"col(?:umn)?\s*[=:]\s*(-?\d+)", text, re.IGNORECASE)
        if row_m and col_m:
            pred_row = int(row_m.group(1))
            pred_col = int(col_m.group(1))
        else:
            # Fallback: first two integers in the response
            nums = re.findall(r"-?\d+", text)
            if len(nums) >= 2:
                pred_row, pred_col = int(nums[0]), int(nums[1])
                parse_note = "fallback_integers"
            else:
                parse_note = "parse_failed"

        row_correct = (pred_row == gt["row"])
        col_correct = (pred_col == gt["col"])
        score = 1.0 if (row_correct and col_correct) else (0.5 if (row_correct or col_correct) else 0.0)
        correct = score == 1.0

        # Detect wind-naive error: predicted the no-wind landing cell
        meta = probe_result.metadata
        wind_naive_error = (
            meta.get("has_wind", False)
            and meta.get("wind_offset", 0) != 0
            and pred_row == meta.get("naive_row")
            and pred_col == meta.get("naive_col")
        )

        return EvalResult(
            correct=correct,
            score=score,
            predicted={"row": pred_row, "col": pred_col},
            ground_truth=gt,
            probe_type=probe_result.probe_type,
            lm_text=response.text,
            details={
                **probe_result.metadata,
                "row_correct": row_correct,
                "col_correct": col_correct,
                "wind_naive_error": wind_naive_error,
                "parse_note": parse_note,
            },
        )
