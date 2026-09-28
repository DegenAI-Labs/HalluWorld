from __future__ import annotations

import re

from halluworld.evaluator import EvalResult, Evaluator
from halluworld.tracks.grid.evaluators import _parse_yes_no
from halluworld.lm.base import LMResponse
from halluworld.probe import ProbeResult
from halluworld.tracks.chess.serializers import parse_piece_name


_UCI_PATTERN = re.compile(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b", re.IGNORECASE)
_INT_PATTERN = re.compile(r"-?\d+")
_IDK_PATTERN = re.compile(
    r"\b(idk|unknown|don'?t\s*know|cannot\s*tell|can'?t\s*tell|insufficient)\b",
    re.IGNORECASE,
)


def _parse_yes_no_idk(text: str) -> str | None:
    """Return one of 'yes', 'no', 'idk', or None for ambiguous responses."""
    if _IDK_PATTERN.search(text):
        return "idk"
    yn = _parse_yes_no(text)
    if yn is True:
        return "yes"
    if yn is False:
        return "no"
    return None


class ChessYesNoEvaluator(Evaluator):
    """Evaluator for yes/no chess probes."""

    def evaluate(self, response: LMResponse, probe_result: ProbeResult) -> EvalResult:
        predicted = _parse_yes_no(response.text)
        gt: bool = probe_result.ground_truth
        correct = predicted is not None and predicted == gt
        score = 1.0 if correct else 0.0
        parse_note = "exact_match" if correct else ("ambiguous_response" if predicted is None else "mismatch")

        return EvalResult(
            correct=correct,
            score=score,
            predicted=predicted,
            ground_truth=gt,
            probe_type=probe_result.probe_type,
            lm_text=response.text,
            details={**probe_result.metadata, "parse_note": parse_note},
        )


class ChessYesNoStrictEvaluator(Evaluator):
    """Yes/no where the model reply must be exactly ``yes`` or ``no`` (ignoring outer whitespace).

    Stricter than :class:`ChessYesNoEvaluator`, which treats any substring containing
    ``yes``/``no`` as a parse success (so long explanations still score).
    """

    def evaluate(self, response: LMResponse, probe_result: ProbeResult) -> EvalResult:
        raw = response.text.strip().lower()
        gt: bool = probe_result.ground_truth
        if raw in ("yes", "no"):
            predicted = raw == "yes"
        else:
            predicted = None
        correct = predicted is not None and predicted == gt
        score = 1.0 if correct else 0.0
        parse_note = "exact_token" if correct else ("non_token_yes_no" if predicted is None else "mismatch")

        return EvalResult(
            correct=correct,
            score=score,
            predicted=predicted,
            ground_truth=gt,
            probe_type=probe_result.probe_type,
            lm_text=response.text,
            details={**probe_result.metadata, "parse_note": parse_note},
        )


class ChessLegalUciSetEvaluator(Evaluator):
    """Scores whether the model's UCI is one of the legal moves (set membership)."""

    def evaluate(self, response: LMResponse, probe_result: ProbeResult) -> EvalResult:
        legal_ucis = set(probe_result.metadata.get("legal_ucis") or [])
        if not legal_ucis and isinstance(probe_result.ground_truth, list):
            legal_ucis = set(probe_result.ground_truth)

        if not legal_ucis:
            return EvalResult(
                correct=False,
                score=0.0,
                predicted=None,
                ground_truth=probe_result.ground_truth,
                probe_type=probe_result.probe_type,
                lm_text=response.text,
                details={**probe_result.metadata, "parse_note": "no_legal_moves_in_probe"},
            )

        m = _UCI_PATTERN.search(response.text.strip())
        predicted = m.group(1).lower() if m else None
        correct = predicted is not None and predicted in legal_ucis
        score = 1.0 if correct else 0.0

        return EvalResult(
            correct=correct,
            score=score,
            predicted=predicted,
            ground_truth=sorted(legal_ucis),
            probe_type=probe_result.probe_type,
            lm_text=response.text,
            details={
                **probe_result.metadata,
                "parse_note": "uci_in_legal_set" if correct else (
                    "uci_not_found" if not predicted else "illegal_or_wrong_uci"
                ),
            },
        )


class ChessBestMoveEvaluator(Evaluator):
    """Exact UCI match against Stockfish-best-move probe ground truth."""

    def evaluate(self, response: LMResponse, probe_result: ProbeResult) -> EvalResult:
        gt = probe_result.ground_truth
        if gt is None:
            return EvalResult(
                correct=False,
                score=0.0,
                predicted=None,
                ground_truth=None,
                probe_type=probe_result.probe_type,
                lm_text=response.text,
                details={**probe_result.metadata, "parse_note": "probe_skipped"},
            )

        m = _UCI_PATTERN.search(response.text.strip())
        predicted = m.group(1).lower() if m else None
        correct = predicted == str(gt).lower()
        score = 1.0 if correct else 0.0

        return EvalResult(
            correct=correct,
            score=score,
            predicted=predicted,
            ground_truth=gt,
            probe_type=probe_result.probe_type,
            lm_text=response.text,
            details={
                **probe_result.metadata,
                "parse_note": "exact_match" if correct else ("uci_not_found" if not predicted else "mismatch"),
            },
        )


class ChessYesNoIDKEvaluator(Evaluator):
    """Three-way evaluator for probes whose ground truth may be 'idk'.

    Scoring policy:
      * gt is bool, predicted matches: 1.0
      * gt is bool, predicted is 'idk': 0.0 (over-cautious)
      * gt is bool, predicted disagrees: 0.0 (confident wrong)
      * gt is 'idk', predicted is 'idk': 1.0
      * gt is 'idk', predicted is yes/no: 0.0 (over-confident — hallucination)
      * unparseable response: 0.0
    """

    def evaluate(self, response: LMResponse, probe_result: ProbeResult) -> EvalResult:
        predicted = _parse_yes_no_idk(response.text)
        gt = probe_result.ground_truth

        if predicted is None:
            correct, score, note = False, 0.0, "ambiguous_response"
        elif gt == "idk":
            correct = predicted == "idk"
            score = 1.0 if correct else 0.0
            note = "exact_match" if correct else "overconfident"
        else:
            # gt is bool
            mapped = {"yes": True, "no": False}.get(predicted)
            if predicted == "idk":
                correct, score, note = False, 0.0, "abstained_when_known"
            elif mapped == gt:
                correct, score, note = True, 1.0, "exact_match"
            else:
                correct, score, note = False, 0.0, "mismatch"

        return EvalResult(
            correct=correct,
            score=score,
            predicted=predicted,
            ground_truth=gt,
            probe_type=probe_result.probe_type,
            lm_text=response.text,
            details={**probe_result.metadata, "parse_note": note},
        )


class ChessIntegerEvaluator(Evaluator):
    """Evaluator for probes whose ground truth is a non-negative integer."""

    def evaluate(self, response: LMResponse, probe_result: ProbeResult) -> EvalResult:
        gt = probe_result.ground_truth
        if gt is None:
            return EvalResult(
                correct=False,
                score=0.0,
                predicted=None,
                ground_truth=None,
                probe_type=probe_result.probe_type,
                lm_text=response.text,
                details={**probe_result.metadata, "parse_note": "probe_skipped"},
            )

        text = response.text.strip()

        # Strict path: response is exactly one integer token.
        exact = re.fullmatch(r"\s*(-?\d+)\s*", text)
        if exact is not None:
            predicted = int(exact.group(1))
            note = "exact_token"
        else:
            # Common verbose formats: "Final answer: 5", "answer is 5", "**5**", etc.
            final = re.search(
                r"(?:final\s+answer|answer)\s*(?:is|:)?\s*\**\s*(-?\d+)\s*\**\s*$",
                text,
                re.IGNORECASE,
            )
            if final is not None:
                predicted = int(final.group(1))
                note = "final_answer_integer"
            else:
                nums = _INT_PATTERN.findall(text)
                if not nums:
                    return EvalResult(
                        correct=False,
                        score=0.0,
                        predicted=None,
                        ground_truth=gt,
                        probe_type=probe_result.probe_type,
                        lm_text=response.text,
                        details={**probe_result.metadata, "parse_note": "no_integer_found"},
                    )
                # Fallback for chain-of-thought responses that include move numbers:
                # the model's final answer is usually the last integer it writes.
                predicted = int(nums[-1])
                note = "fallback_last_integer"

        if predicted is None:
            return EvalResult(
                correct=False,
                score=0.0,
                predicted=None,
                ground_truth=gt,
                probe_type=probe_result.probe_type,
                lm_text=response.text,
                details={**probe_result.metadata, "parse_note": "no_integer_found"},
            )

        correct = predicted == int(gt)
        return EvalResult(
            correct=correct,
            score=1.0 if correct else 0.0,
            predicted=predicted,
            ground_truth=int(gt),
            probe_type=probe_result.probe_type,
            lm_text=response.text,
            details={
                **probe_result.metadata,
                "parse_note": "exact_match" if correct else "mismatch",
                "integer_parse_mode": note,
            },
        )


class ChessPieceNameEvaluator(Evaluator):
    """Evaluator that matches a canonical piece-name string (or 'empty')."""

    def evaluate(self, response: LMResponse, probe_result: ProbeResult) -> EvalResult:
        gt = probe_result.ground_truth
        if gt is None:
            return EvalResult(
                correct=False,
                score=0.0,
                predicted=None,
                ground_truth=None,
                probe_type=probe_result.probe_type,
                lm_text=response.text,
                details={**probe_result.metadata, "parse_note": "probe_skipped"},
            )

        predicted = parse_piece_name(response.text)
        correct = predicted is not None and predicted == gt
        if predicted is None:
            note = "no_piece_name_found"
        elif correct:
            note = "exact_match"
        else:
            note = "mismatch"

        return EvalResult(
            correct=correct,
            score=1.0 if correct else 0.0,
            predicted=predicted,
            ground_truth=gt,
            probe_type=probe_result.probe_type,
            lm_text=response.text,
            details={**probe_result.metadata, "parse_note": note},
        )
