"""Terminal release evaluator, conservative grading, and CLI wiring."""

from __future__ import annotations

import json
import runpy
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from halluworld.data import DATA_DIR
from halluworld.lm.base import LMResponse
from halluworld.questions import read_bank
from halluworld.results import read_jsonl
from halluworld.tracks.terminal import evaluation

REPO = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize(
    ("expected", "actual", "schema"),
    [
        ("idx=0250,0256", "idx=0250", "idx=<0250,0256|0250|0256|none>"),
        ("counts=4,3,3", "counts=4", "counts=<a,b,c>"),
        (
            "order=passwords->credentials->users",
            "order=passwords",
            "order=<passwords->credentials->users|other>",
        ),
        ("path=..", "path=.", "path=<..|.|unknown>"),
        ("arch=arm64;repo=consistent", "arch=arm64", "arch=<x>;repo=<y>"),
        ("a=x|y", "a=x", "a=<x|y>"),
    ],
)
def test_structured_prefixes_never_grade_correct(expected, actual, schema):
    grade = evaluation.grade_answer(
        expected=expected, actual=actual, answer_schema=schema
    )
    assert not grade.correct


def test_structured_values_keep_punctuation_and_require_every_key():
    expected = "path=/tmp/a=b;c;order=a->b|c;status=ok"
    schema = "path=<value>;order=<value>;status=<value>"
    exact = evaluation.grade_answer(
        expected=expected, actual=expected, answer_schema=schema
    )
    missing = evaluation.grade_answer(
        expected=expected,
        actual="path=/tmp/a=b;c;order=a->b|c",
        answer_schema=schema,
    )
    assert exact.correct
    assert not missing.correct


def test_all_529_terminal_golden_answers_self_grade(bank_dir):
    records = list(read_bank(bank_dir / "terminal.jsonl.gz"))
    assert len(records) == 529
    assert len({record.task_or_level for record in records}) == 110
    for record in records:
        grade = evaluation.grade_answer(
            expected=str(record.ground_truth),
            actual=str(record.ground_truth),
            answer_schema=record.answer_schema,
        )
        assert grade.correct, record.question_id


@pytest.mark.parametrize("command", [("eval", "terminal"), ("terminal", "eval")])
def test_terminal_dry_run_writes_canonical_records(bank_dir, command, tmp_path):
    out = tmp_path / "terminal"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "halluworld",
            *command,
            "--provider",
            "openai",
            "--model",
            "gpt-4o-mini",
            "--out",
            str(out),
            "--dry-run",
            "--limit",
            "3",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    records = list(read_jsonl(out / "results.jsonl"))
    assert len(records) == 3
    assert all(record.track == "terminal" for record in records)
    assert all(record.error == "dry_run" and record.score is None for record in records)
    summary = json.loads((out / "summary.json").read_text())
    assert summary["selected"] == 3


def _args(tmp_path, **overrides):
    values = {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "out": str(tmp_path / "terminal"),
        "version": "v0.1",
        "tasks": None,
        "question_ids": None,
        "probe_types": None,
        "limit": 2,
        "max_tokens": 64,
        "max_context_chars": 0,
        "reasoning_effort": None,
        "thinking_effort": None,
        "temperature": 0.0,
        "retries": 1,
        "api_base": "https://inference.baseten.co/v1",
        "run_id": "terminal-test",
        "resume": False,
        "dry_run": False,
        "verbose": False,
    }
    values.update(overrides)
    return Namespace(**values)


def test_mocked_terminal_execution_emits_scored_probe_records(bank_dir, monkeypatch, tmp_path):
    selected = evaluation.load_terminal_bank()[:2]
    answers = iter(str(record.ground_truth) for record in selected)

    class FakeLM:
        def query(self, system, user):
            return LMResponse(
                text=next(answers), model="gpt-4o-mini-2024-07-18", prompt_tokens=10,
                completion_tokens=2,
            )

    monkeypatch.setattr(evaluation, "_make_lm", lambda args: FakeLM())
    assert evaluation.run_terminal(_args(tmp_path)) == 0
    records = list(read_jsonl(tmp_path / "terminal" / "results.jsonl"))
    assert len(records) == 2
    assert all(record.score == 1.0 for record in records)
    assert all(record.model == "gpt-4o-mini-2024-07-18" for record in records)
    manifest = json.loads((tmp_path / "terminal" / "manifest.json").read_text())
    assert manifest["counts"] == {"excluded": 0, "records": 2, "scored": 2}


def test_api_errors_are_excluded_and_resume_retries_them(bank_dir, monkeypatch, tmp_path):
    class BrokenLM:
        def query(self, system, user):
            raise RuntimeError("provider unavailable")

    monkeypatch.setattr(evaluation, "_make_lm", lambda args: BrokenLM())
    assert evaluation.run_terminal(_args(tmp_path, limit=1)) == 1
    first = list(read_jsonl(tmp_path / "terminal" / "results.jsonl"))
    assert first[0].score is None
    assert first[0].error == "api_error"

    golden = str(evaluation.load_terminal_bank()[0].ground_truth)

    class RecoveredLM:
        def query(self, system, user):
            return LMResponse(text=golden, model="recovered")

    monkeypatch.setattr(evaluation, "_make_lm", lambda args: RecoveredLM())
    assert evaluation.run_terminal(_args(tmp_path, limit=1, resume=True)) == 0
    recovered = list(read_jsonl(tmp_path / "terminal" / "results.jsonl"))
    assert len(recovered) == 1
    assert recovered[0].score == 1.0
    assert recovered[0].error is None


def test_raw_terminal_bank_matches_frozen_bank_exactly(bank_dir):
    from halluworld.data import PROBE_BANKS_DIR

    if not (PROBE_BANKS_DIR / "terminal_20260504").is_dir():
        pytest.skip("raw terminal probes are local-only (raw/ in the HF dataset)")
    namespace = runpy.run_path(str(REPO / "scripts" / "build_question_bank.py"))
    raw = namespace["build_terminal"]()
    frozen = list(read_bank(bank_dir / "terminal.jsonl.gz"))
    assert raw == frozen
