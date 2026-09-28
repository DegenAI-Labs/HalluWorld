"""User-facing evaluation commands and their domain-runner wiring."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from halluworld import config as cfgmod
from halluworld import evaluation
from halluworld.data import DATA_DIR

REPO = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize(
    ("domain", "extra", "expected"),
    [
        ("grid", [], "innav:    off"),
        (
            "grid",
            ["--include-innav"],
            "innav:    enabled (P1_dense_array, P2_corridor_gauntlet, P3_rotation_challenge)",
        ),
        ("chess", [], "fen mode:  off"),
        ("innav", [], "probe timesteps: 5"),
    ],
)
def test_polished_eval_commands_dry_run(domain, extra, expected, tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "halluworld",
            "eval",
            domain,
            "--provider",
            "openai",
            "--model",
            "gpt-4o-mini",
            "--out",
            str(tmp_path / domain),
            "--dry-run",
            *extra,
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert expected in result.stdout
    assert not (tmp_path / domain).exists(), "dry-run must not create output"


def test_limit_caps_episodes_per_level_in_dry_run(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "halluworld",
            "eval",
            "grid",
            "--provider",
            "openai",
            "--model",
            "gpt-4o-mini",
            "--out",
            str(tmp_path / "grid"),
            "--episodes",
            "10",
            "--limit",
            "2",
            "--dry-run",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "episodes:  2 per level" in result.stdout
    assert "limit:     2 episodes per level" in result.stdout


def test_limit_must_be_positive(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "halluworld",
            "eval",
            "grid",
            "--provider",
            "openai",
            "--model",
            "gpt-4o-mini",
            "--out",
            str(tmp_path / "grid"),
            "--limit",
            "0",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "--limit must be at least 1" in result.stderr


def _grid_args(tmp_path, *, include_innav=False):
    return Namespace(
        provider="openai",
        model="gpt-4o-mini",
        out=str(tmp_path / "grid"),
        levels=["P1_dense_array"],
        episodes=1,
        seed=0,
        max_tokens=64,
        serializer="symbolic",
        reasoning_effort=None,
        thinking_effort=None,
        resume=False,
        verbose=False,
        include_innav=include_innav,
        innav_levels=list(evaluation.INNAV_PARITY_LEVELS),
        innav_episodes=2,
        innav_serializer=None,
        probe_timesteps=3,
        max_steps=20,
        navigation_model=None,
        dry_run=False,
    )


def test_grid_eval_forwards_provider_and_keeps_innav_off(monkeypatch, tmp_path):
    calls = []

    def fake_grid_main(argv):
        calls.append(argv)
        output = Path(argv[argv.index("--output") + 1])
        path = output.parent / "results_gpt-4o-mini.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["model", "score"])
            writer.writeheader()
            writer.writerow({"model": "gpt-4o-mini", "score": 1.0})

    monkeypatch.setattr(evaluation, "_require_provider_key", lambda provider: None)
    from halluworld.tracks.grid import perception

    monkeypatch.setattr(perception, "main", fake_grid_main)
    monkeypatch.setattr(
        evaluation,
        "run_innav",
        lambda args: pytest.fail("InNav must be off unless explicitly requested"),
    )

    assert evaluation.run_grid(_grid_args(tmp_path)) == 0
    assert calls and calls[0][:4] == [
        "--provider",
        "openai",
        "--models",
        "gpt-4o-mini",
    ]
    manifest = json.loads((tmp_path / "grid" / "manifest.json").read_text())
    assert manifest["track"] == "grid"
    assert manifest["counts"]["records"] == 1


def test_grid_eval_forwards_effective_episode_limit(monkeypatch, tmp_path):
    calls = []

    def fake_grid_main(argv):
        calls.append(argv)

    args = _grid_args(tmp_path)
    args.episodes = 10
    args.limit = 2
    monkeypatch.setattr(evaluation, "_require_provider_key", lambda provider: None)
    from halluworld.tracks.grid import perception

    monkeypatch.setattr(perception, "main", fake_grid_main)
    assert evaluation.run_grid(args) == 0
    assert calls[0][calls[0].index("--episodes") + 1] == "2"


def test_grid_eval_opt_in_dispatches_innav_subset(monkeypatch, tmp_path):
    captured = []

    def fake_grid_main(argv):
        output = Path(argv[argv.index("--output") + 1])
        path = output.parent / "results_gpt-4o-mini.csv"
        path.write_text("model,score\ngpt-4o-mini,1.0\n")

    monkeypatch.setattr(evaluation, "_require_provider_key", lambda provider: None)
    from halluworld.tracks.grid import perception

    monkeypatch.setattr(perception, "main", fake_grid_main)
    monkeypatch.setattr(
        evaluation, "run_innav", lambda args: (captured.append(args), 0)[1]
    )

    assert evaluation.run_grid(_grid_args(tmp_path, include_innav=True)) == 0
    assert captured[0].levels == list(evaluation.INNAV_PARITY_LEVELS)
    assert captured[0].out == str((tmp_path / "grid" / "innav").resolve())


def test_innav_eval_forwards_provider_levels_and_output(monkeypatch, tmp_path):
    calls = []

    def fake_innav_main(argv):
        calls.append(argv)
        output = Path(argv[argv.index("--output") + 1])
        output.write_text("model,level,innav_accuracy\ngpt-4o-mini,P1_dense_array,1.0\n")

    monkeypatch.setattr(evaluation, "_require_provider_key", lambda provider: None)
    from halluworld.tracks.innav import runner

    monkeypatch.setattr(runner, "main", fake_innav_main)
    args = Namespace(
        provider="openai",
        model="gpt-4o-mini",
        out=str(tmp_path / "innav"),
        levels=["P1_dense_array"],
        episodes=5,
        limit=1,
        seed=42,
        max_tokens=64,
        reasoning_effort="low",
        navigation_model=None,
        probe_timesteps=3,
        max_steps=20,
        serializer="symbolic",
        trace_dir=None,
        dry_run=False,
        verbose=False,
    )
    assert evaluation.run_innav(args) == 0
    assert calls[0][:4] == [
        "--provider",
        "openai",
        "--models",
        "gpt-4o-mini",
    ]
    assert calls[0][calls[0].index("--levels") + 1] == "P1_dense_array"
    assert calls[0][calls[0].index("--episodes") + 1] == "1"
    assert calls[0][calls[0].index("--output") + 1] == str(
        (tmp_path / "innav" / "results.csv").resolve()
    )
    manifest = json.loads((tmp_path / "innav" / "manifest.json").read_text())
    assert manifest["track"] == "innav"
    assert manifest["counts"]["records"] == 1


def test_chess_eval_uses_packaged_config_and_exact_output(monkeypatch, tmp_path):
    seen = {}

    def fake_battery_main():
        seen["provider"] = os.environ["LM_PROVIDER"]
        seen["model"] = os.environ["ANTHROPIC_MODEL"]
        seen["episodes"] = os.environ["N_EPISODES"]
        out = Path(os.environ["HALLUWORLD_RESULTS_DIR"])
        out.mkdir(parents=True, exist_ok=True)
        (out / "results.jsonl").write_text(
            json.dumps({"probe_name": "chess_can_capture"}) + "\n"
        )

    monkeypatch.setattr(evaluation, "_require_provider_key", lambda provider: None)
    from halluworld.tracks.chess import battery

    monkeypatch.setattr(battery, "main", fake_battery_main)
    args = Namespace(
        provider="anthropic",
        model="claude-sonnet-4-6",
        out=str(tmp_path / "chess"),
        levels=None,
        episodes=5,
        limit=1,
        seed=0,
        max_tokens=1024,
        fen_mode="off",
        reasoning_effort=None,
        thinking_effort="low",
        verbose=False,
        dry_run=False,
        # The battery path is opt-in since 2026-09-17; the default replays the
        # frozen bank. This test is specifically about the battery wiring.
        regenerate=True,
    )
    assert evaluation.run_chess(args) == 0
    assert seen == {
        "provider": "anthropic",
        "model": "claude-sonnet-4-6",
        "episodes": "1",
    }
    manifest = json.loads((tmp_path / "chess" / "manifest.json").read_text())
    assert manifest["counts"]["records"] == 1


@pytest.mark.parametrize(
    ("module_name", "class_name"),
    [("grid", "_AnthropicLM"), ("innav", "AnthropicLM")],
)
def test_explicit_provider_overrides_model_name_inference(
    monkeypatch, module_name, class_name
):
    if module_name == "grid":
        from halluworld.tracks.grid import perception as module
    else:
        from halluworld.tracks.innav import runner as module

    captured = {}

    class FakeAnthropic:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setenv("HALLUWORLD_PROVIDER", "anthropic")
    monkeypatch.setattr(module, class_name, FakeAnthropic)
    module._make_lm("vendor-neutral-model", max_tokens=64)
    assert captured["model"] == "vendor-neutral-model"


# --------------------------------------------------------------------------- #
# chess replays the frozen bank                                                #
# --------------------------------------------------------------------------- #

def _chess_replay_args(tmp_path, **over):
    base = dict(
        provider="openai",
        model="oracle",
        out=str(tmp_path / "chess"),
        levels=None,
        episodes=2,
        limit=None,
        seed=0,
        max_tokens=64,
        fen_mode="off",
        reasoning_effort=None,
        thinking_effort=None,
        verbose=False,
        dry_run=False,
        regenerate=False,
        version="v0.1",
    )
    base.update(over)
    return Namespace(**base)


class _OracleLM:
    """Answers each frozen question from its own ground truth."""

    def __init__(self, records):
        self._by_context = {r.context: r for r in records}

    def query(self, system, user, max_tokens=None):
        from halluworld.lm.base import LMResponse

        gt = self._by_context[user].ground_truth
        if isinstance(gt, list):
            text = gt[0]
        elif isinstance(gt, bool):
            text = "yes" if gt else "no"
        else:
            text = str(gt)
        return LMResponse(text=text, model="oracle")


def test_chess_eval_replays_the_frozen_bank_rather_than_regenerating(monkeypatch, tmp_path):
    """The questions asked must be the questions frozen.

    run_chess used to always call battery.main(), which re-derives questions
    from a config and trusts determinism to land on the bank's. Any drift
    between that config and the one the bank was built under scored the model
    on different questions with nothing in the output saying so.
    """
    from halluworld.tracks.chess import replay

    called = []
    from halluworld.tracks.chess import battery

    monkeypatch.setattr(battery, "main", lambda: called.append("battery"))
    monkeypatch.setattr(evaluation, "_require_provider_key", lambda provider: None)
    monkeypatch.setattr(
        replay, "_make_lm", lambda args: _OracleLM(replay.load_chess_bank("v0.1"))
    )

    assert evaluation.run_chess(_chess_replay_args(tmp_path)) == 0
    assert called == [], "battery must not run on the replay path"

    records = [json.loads(l) for l in (tmp_path / "chess" / "records.jsonl").read_text().splitlines() if l.strip()]
    assert records, "replay wrote no records"
    # An oracle answering from frozen ground truth must score perfectly; any
    # miss means a probe type is wired to the wrong evaluator.
    assert all(r["score"] == 1.0 for r in records)

    bank_ids = {r.question_id for r in replay.load_chess_bank("v0.1")}
    assert {r["question_id"] for r in records} <= bank_ids
    assert {r["extra"]["bank_version"] for r in records} == {"v0.1"}

    manifest = json.loads((tmp_path / "chess" / "manifest.json").read_text())
    assert manifest["question_bank"]["version"] == "v0.1"
    assert manifest["config"]["source"] == "question_bank_replay"


def test_chess_replay_covers_every_probe_type_in_the_bank():
    """A probe type with no evaluator must raise, not be graded by the wrong one."""
    from halluworld.tracks.chess import replay

    for record in replay.load_chess_bank("v0.1"):
        assert replay._evaluator_for(record.probe_type) is not None

    with pytest.raises(replay.ChessReplayError, match="no evaluator known"):
        replay._evaluator_for("chess_not_a_real_probe")


def test_chess_regenerate_flag_still_reaches_the_battery(monkeypatch, tmp_path):
    called = []
    from halluworld.tracks.chess import battery

    def fake_main():
        called.append("battery")
        out = Path(os.environ["HALLUWORLD_RESULTS_DIR"])
        out.mkdir(parents=True, exist_ok=True)
        (out / "results.jsonl").write_text(json.dumps({"probe_name": "x"}) + "\n")

    monkeypatch.setattr(battery, "main", fake_main)
    monkeypatch.setattr(evaluation, "_require_provider_key", lambda provider: None)
    assert evaluation.run_chess(_chess_replay_args(tmp_path, regenerate=True)) == 0
    assert called == ["battery"]


def test_chess_replay_threads_preserve_order_and_scores(monkeypatch, tmp_path):
    """--workers must not change results, only wall clock."""
    from halluworld.tracks.chess import replay

    args = _chess_replay_args(tmp_path, episodes=4)
    records = replay.select_records(args, "v0.1")
    monkeypatch.setattr(replay, "_make_lm", lambda a: _OracleLM(records))

    args.workers = 1
    sequential = replay.run_chess_replay(args, "v0.1", tmp_path)
    args.workers = 8
    threaded = replay.run_chess_replay(args, "v0.1", tmp_path)

    assert [r.question_id for r in sequential] == [r.question_id for r in threaded]
    assert [r.score for r in sequential] == [r.score for r in threaded]


def test_chess_replay_baseten_needs_a_base_url(monkeypatch, tmp_path):
    """BasetenLM has no default base_url, so omitting it is a TypeError."""
    from halluworld.tracks.chess import replay

    monkeypatch.setenv("BASETEN_API_KEY", "x")
    monkeypatch.delenv("BASETEN_BASE_URL", raising=False)
    args = _chess_replay_args(tmp_path, provider="baseten", api_base=None, retries=3)
    lm = replay._make_lm(args)
    assert lm.base_url == "https://inference.baseten.co/v1"

    args = _chess_replay_args(
        tmp_path, provider="baseten", api_base="https://api.x.ai/v1", retries=3
    )
    assert replay._make_lm(args).base_url == "https://api.x.ai/v1"


# --------------------------------------------------------------------------- #
# grid replays an authored bank                                                #
# --------------------------------------------------------------------------- #

def test_grid_v02_is_replayed_not_regenerated(monkeypatch, tmp_path):
    """An authored grid bank must be replayed; make_probes cannot produce it.

    eval grid regenerates via perception.make_probes, which reproduces v0.1
    because v0.1 was extracted from that function. It produces 0 of grid v0.2's
    authored questions, so regenerating there would score v0.1's questions
    under a v0.2 label.
    """
    from halluworld.tracks.grid import replay as gridreplay

    bank = DATA_DIR / "questions" / "v0.2" / "grid.jsonl.gz"
    if not bank.exists():
        pytest.skip("grid v0.2 bank not built")

    records = gridreplay.load_grid_bank("v0.2")
    assert records and all(r.context for r in records), "records must carry a prompt"
    assert all((r.extra or {}).get("system_prompt") for r in records)

    called = []
    from halluworld.tracks.grid import perception

    monkeypatch.setattr(perception, "main", lambda argv: called.append(argv))
    monkeypatch.setattr(evaluation, "_require_provider_key", lambda provider: None)

    # Key on the whole prompt: several records share one scene, so matching on
    # context alone returns another question's ground truth.
    by_prompt = {"%s\n\n%s" % (r.context, r.question): r for r in records}

    class _GridOracle:
        def query(self, system, user, max_tokens=None):
            from halluworld.lm.base import LMResponse

            gt = by_prompt[user].ground_truth
            text = "yes" if gt is True else "no" if gt is False else str(gt)
            return LMResponse(text=text, model="oracle")

    import halluworld.lm.factory as factory

    monkeypatch.setattr(factory, "make_lm", lambda args: _GridOracle())

    args = Namespace(
        provider="openai", model="oracle", out=str(tmp_path / "grid"), levels=None,
        episodes=2, limit=None, seed=0, max_tokens=64, serializer="symbolic",
        reasoning_effort=None, thinking_effort=None, resume=False, verbose=False,
        include_innav=False, innav_levels=[], innav_episodes=1, innav_serializer=None,
        probe_timesteps=1, max_steps=10, navigation_model=None, dry_run=False,
        version="v0.2", workers=4, retries=3, api_base=None, regenerate=False,
    )
    assert evaluation.run_grid(args) == 0
    assert called == [], "perception.main must not run for an authored bank"

    written = [json.loads(l) for l in
               (tmp_path / "grid" / "records.jsonl").read_text().splitlines() if l.strip()]
    assert written
    assert all(r["score"] == 1.0 for r in written), "oracle answers must grade correct"


@pytest.mark.parametrize(
    ("expected", "answer", "schema", "want"),
    [
        (2, "2", "one integer", True),
        # The package flags the old scorer for taking the FIRST integer, so a
        # reasoned reply mentioning another number first was mis-scored.
        (2, "There are 3 doors, so the answer is 2.", "one integer", True),
        (2, "3", "one integer", False),
        (True, "yes", "yes|no", True),
        (False, "No.", "yes|no", True),
        ("cannot_determine", "cannot_determine", "cannot_determine", True),
        ("cannot_determine", "**cannot_determine**", "cannot_determine", True),
        ("cannot_determine", "The door is closed.", "cannot_determine", False),
        ("open to open", "The gate goes open to open.", "open to open|open to closed", True),
        ("open to closed", "open to open", "open to open|open to closed", False),
    ],
)
def test_grid_grading_by_schema(expected, answer, schema, want):
    from halluworld.tracks.grid.replay import grade

    correct, _ = grade(expected, answer, schema)
    assert correct is want


# --------------------------------------------------------------------------- #
# provider endpoints                                                           #
# --------------------------------------------------------------------------- #

def _endpoint(provider, monkeypatch, api_base=None):
    from halluworld.lm.factory import make_lm

    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "BASETEN_API_KEY", "XAI_API_KEY"):
        monkeypatch.setenv(key, "test")
    monkeypatch.delenv("BASETEN_BASE_URL", raising=False)
    lm = make_lm(Namespace(provider=provider, model="m", max_tokens=64,
                           retries=3, api_base=api_base, temperature=0.0))
    return str(getattr(lm, "base_url", None) or lm._client.base_url)


def test_each_provider_reaches_its_own_endpoint(monkeypatch):
    """A provider must not inherit another provider's endpoint.

    `eval terminal --api-base` used to default to Baseten's URL, which was
    harmless while only the Baseten branch read it. Once terminal delegated to
    the shared factory, that default reached the OpenAI branch too and every
    openai/xai terminal request went to Baseten, failing with Baseten's 403.
    """
    assert "openai.com" in _endpoint("openai", monkeypatch)
    assert "api.x.ai" in _endpoint("xai", monkeypatch)
    assert "baseten.co" in _endpoint("baseten", monkeypatch)
    # An explicit override still wins, for the provider that asked for it.
    assert _endpoint("baseten", monkeypatch, "https://custom/v1") == "https://custom/v1"


def test_terminal_api_base_does_not_default_to_a_provider(monkeypatch, tmp_path):
    """The terminal parser must leave --api-base unset by default.

    A provider-specific default here silently reroutes every other provider.
    """
    from halluworld import cli
    import halluworld.tracks.terminal.evaluation as termeval

    seen = {}
    monkeypatch.setattr(termeval, "run_terminal",
                        lambda args: (seen.update(vars(args)), 0)[1])
    rc = cli.main([
        "eval", "terminal", "--provider", "openai", "--model", "m",
        "--out", str(tmp_path / "t"),
    ])
    assert rc == 0
    assert seen["api_base"] is None


def test_anthropic_omits_temperature_when_the_sdk_rejects_it():
    """Passing an unsupported parameter fails client-side, before any request.

    anthropic SDK 1.1.0 dropped `temperature` from Messages.create. Sending it
    unconditionally raised TypeError on every call, so all three Claude models
    scored zero across all three tracks and the records read `api_error` --
    which looks like a provider outage rather than a version mismatch.
    """
    from unittest.mock import MagicMock

    from halluworld.lm.anthropic_lm import AnthropicLM

    lm = AnthropicLM(model="claude-opus-5", api_key="x", temperature=0.0)
    sent = {}

    def fake_create(**kwargs):
        sent.update(kwargs)
        message = MagicMock()
        block = MagicMock()
        block.text = "yes"
        message.content = [block]
        message.usage.input_tokens = 1
        message.usage.output_tokens = 1
        message.model = "claude-opus-5"
        return message

    lm._client.messages.create = fake_create
    assert lm.query("sys", "user").text == "yes"
    if not lm._sdk_accepts_temperature:
        assert "temperature" not in sent
    assert {"model", "max_tokens", "system", "messages"} <= set(sent)


def test_resume_keeps_scored_records_and_reasks_only_the_rest(monkeypatch, tmp_path):
    """A partial run must cost only the questions it never got answers for.

    A run that dies partway -- rate limited, out of credits -- leaves a mix of
    scored and unscored records. Re-asking the scored ones pays again for
    answers already held; skipping the job entirely drops the unanswered ones.
    """
    from halluworld.results import ProbeRecord, write_jsonl
    from halluworld.tracks.chess import replay

    out = tmp_path / "chess"
    out.mkdir()
    bank = replay.load_chess_bank("v0.1")
    already, rest = bank[:5], bank[5:12]
    write_jsonl(
        [
            ProbeRecord(
                run_id="prev", track="chess", suite=r.suite, level=r.task_or_level,
                # Must match the args below: resume refuses output belonging
                # to a different provider/model rather than mixing them.
                model="oracle", model_requested="oracle", provider="openai",
                question_id=r.question_id, probe_type=r.probe_type,
                question=r.question, ground_truth=r.ground_truth,
                response="yes", is_correct=True, score=1.0,
            )
            for r in already
        ],
        out / "records.jsonl",
    )

    asked = []

    class _Counting:
        def query(self, system, user, max_tokens=None):
            from halluworld.lm.base import LMResponse

            asked.append(user)
            return LMResponse(text="yes", model="m")

    import halluworld.lm.factory as factory

    monkeypatch.setattr(factory, "make_lm", lambda args: _Counting())
    monkeypatch.setattr(evaluation, "_require_provider_key", lambda provider: None)

    args = _chess_replay_args(tmp_path, episodes=12, out=str(out))
    args.resume = True
    assert evaluation.run_chess(args) == 0

    done_ids = {r.question_id for r in already}
    assert len(asked) > 0
    assert len(asked) == len([r for r in replay.select_records(args, "v0.1")
                              if r.question_id not in done_ids])

    written = [json.loads(l) for l in
               (out / "records.jsonl").read_text().splitlines() if l.strip()]
    ids = {r["question_id"] for r in written}
    assert done_ids <= ids, "previously scored records must survive the merge"
    assert all(r["score"] is not None for r in written if r["question_id"] in done_ids)


@pytest.mark.parametrize(
    ("tokens", "stop_reason", "text", "want"),
    [
        (0, "refusal", "[refusal]", "refusal"),         # the strict case
        (0, None, "", "empty_response"),                # nothing came back
        (0, "end_turn", "", "empty_response"),          # zero tokens, no flag
        (3, "refusal", "[refusal] I can't.", None),     # flagged BUT emitted text: graded
        (900, "max_tokens", "", "empty_response"),      # budget spent, no answer
        (5, "end_turn", "yes", None),                   # a real answer
    ],
)
def test_exclusion_reason_is_strictly_zero_tokens_plus_flag(tokens, stop_reason, text, want):
    """A refusal is ZERO output tokens AND the provider's flag; both required.

    Anything that emitted tokens is graded as what it emitted -- a flagged
    refusal with a message included -- so refusing-with-words cannot be
    excluded, and only a genuinely silent response can.
    """
    from halluworld.lm.base import LMResponse
    from halluworld.results import exclusion_reason

    response = LMResponse(text=text, model="m", completion_tokens=tokens,
                          stop_reason=stop_reason)
    assert exclusion_reason(response) == want


def test_chess_replay_reads_the_requested_fen_condition(bank_dir, tmp_path, monkeypatch):
    """--fen-mode selects a bank file and never silently falls back to No-FEN."""
    import gzip
    import shutil

    from halluworld.tracks.chess import replay

    assert replay.chess_bank_file("off") == "chess.jsonl.gz"
    assert replay.chess_bank_file(None) == "chess.jsonl.gz"
    assert replay.chess_bank_file("transpose") == "chess_fen_transpose.jsonl.gz"

    # v0.1 froze only fen-off: asking for transpose must fail, not score No-FEN.
    with pytest.raises(replay.ChessReplayError, match="--regenerate"):
        replay.load_chess_bank("v0.1", "transpose")

    # A version that freezes the condition beside chess.jsonl.gz is read from it.
    fake = tmp_path / "vtest"
    fake.mkdir()
    (fake / "manifest.json").write_text("{}")
    shutil.copy2(bank_dir / "chess.jsonl.gz", fake / "chess.jsonl.gz")
    with gzip.open(bank_dir / "chess.jsonl.gz", "rt") as fh:
        first = [line for line in fh if line.strip()][:3]
    with gzip.open(fake / "chess_fen_transpose.jsonl.gz", "wt") as fh:
        fh.writelines(first)
    monkeypatch.setenv("HALLUWORLD_QUESTIONS_DIR", str(tmp_path))
    assert len(replay.load_chess_bank("vtest", "off")) == 350
    assert len(replay.load_chess_bank("vtest", "transpose")) == 3
