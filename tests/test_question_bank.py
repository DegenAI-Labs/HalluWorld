"""Guards for the frozen question bank.

The bank is the release artifact: it is what every model is evaluated on and
what a published number traces back to. Two things must hold.

1. It must match its manifest. A bank that drifts from its checksum has
   silently changed what the benchmark asks.

2. The gridworld bank must be a faithful extraction. It was lifted out of a
   4775-line make_probes() function holding 384 FixedProbe literals, and the
   only thing that makes that safe is proving the extracted text is identical
   to what the original function produces.
"""

import json
import random

import pytest

from halluworld.data import DATA_DIR, QuestionBankUnavailable, questions_dir
from halluworld.questions import TIERS, QuestionRecord, read_bank, verify_bank


def _resolve_bank_dir():
    try:
        return questions_dir("v0.1")
    except QuestionBankUnavailable:
        return None


BANK_DIR = _resolve_bank_dir()


@pytest.fixture(scope="module")
def manifest():
    if BANK_DIR is None:
        pytest.skip("question bank unavailable; see docs/QUESTIONS.md")
    return json.loads((BANK_DIR / "manifest.json").read_text())


def test_every_bank_matches_its_checksum(manifest):
    for track, entry in manifest["tracks"].items():
        ok, msg = verify_bank(BANK_DIR / entry["file"], entry)
        assert ok, msg


def test_records_validate_and_ids_are_unique(manifest):
    seen = set()
    for track, entry in manifest["tracks"].items():
        for rec in read_bank(BANK_DIR / entry["file"]):
            assert rec.track == track
            assert rec.cognitive_tier in TIERS
            assert rec.question_id not in seen, "duplicate id %s" % rec.question_id
            seen.add(rec.question_id)
            if rec.kind == "fixed":
                assert rec.question, "%s: fixed record with no question" % rec.question_id
            else:
                assert rec.probe_class, "%s: generated record with no class" % rec.question_id
    assert len(seen) == manifest["total_questions"]


def test_terminal_tier_matches_probe_type(manifest):
    """Terminal's probe_type and cognitive_tier encode the same thing."""
    expect = {
        "perceptual": "P", "memory": "M", "causal": "C",
        "uncertainty": "U", "cross-tier compound": "X",
    }
    entry = manifest["tracks"].get("terminal")
    if entry is None:
        pytest.skip("terminal bank not built")
    for rec in read_bank(BANK_DIR / entry["file"]):
        assert rec.cognitive_tier == expect[rec.probe_type], rec.question_id


def test_chess_tiers_match_the_documented_battery(manifest):
    """The battery's P/C/M/X assignment is documented; the bank must agree."""
    expect = {
        "chess_can_capture": "P",
        "chess_defended": "P",
        "chess_hanging": "P",
        "chess_hypothetical_in_check": "C",
        "chess_after_move_undefended_count": "C",
        "chess_hidden_side_capture_stats": "M",
        "chess_san_legal_move": "X",
    }
    entry = manifest["tracks"].get("chess")
    if entry is None:
        pytest.skip("chess bank not built")

    counts = {}
    for rec in read_bank(BANK_DIR / entry["file"]):
        assert rec.probe_type in expect, "unknown chess probe %r" % rec.probe_type
        assert rec.cognitive_tier == expect[rec.probe_type], rec.question_id
        counts[rec.probe_type] = counts.get(rec.probe_type, 0) + 1

    # The export is 50 episodes per probe across the 7-probe battery.
    assert set(counts) == set(expect), "missing probes: %s" % (set(expect) - set(counts))
    assert all(n == 50 for n in counts.values()), counts


def test_chess_battery_has_no_uncertainty_probe(manifest):
    """Documents a real coverage gap rather than letting it pass unnoticed.

    The chess battery has no U-tier probe -- docs/CHESS.md states this outright.
    Grid and terminal both cover U. If a U probe is ever added to the battery,
    this test should fail and be updated, so the gap closing is a deliberate,
    visible event rather than a silent one.
    """
    entry = manifest["tracks"].get("chess")
    if entry is None:
        pytest.skip("chess bank not built")
    tiers = {rec.cognitive_tier for rec in read_bank(BANK_DIR / entry["file"])}
    assert "U" not in tiers, (
        "chess now has a U-tier probe -- good. Update this test and the "
        "coverage note in docs/BENCHMARK.md."
    )
    assert tiers == {"P", "C", "M", "X"}


def test_grid_bank_is_a_faithful_extraction(manifest):
    """The frozen grid questions must be byte-identical to make_probes' output.

    This is the guard that makes extracting 384 hand-written literals out of a
    4775-line function safe. If it fails, the bank and the code that generated
    it have diverged, and any number measured against the bank is no longer
    attributable to the probe definitions in this repository.
    """
    entry = manifest["tracks"].get("grid")
    if entry is None:
        pytest.skip("grid bank not built")

    import halluworld.tracks.grid.perception as perception
    from halluworld.tracks.grid.probes.visibility import FixedProbe

    banked = {}
    for rec in read_bank(BANK_DIR / entry["file"]):
        banked.setdefault(rec.task_or_level, []).append(rec)
    for recs in banked.values():
        recs.sort(key=lambda r: r.question_id)

    # Must be the same seed the builder used, or the generated probes differ.
    seed = 0
    checked = 0
    for level_key, recs in banked.items():
        live = perception.make_probes(level_key, random.Random(seed))
        assert len(live) == len(recs), (
            "%s: bank has %d records, make_probes produced %d"
            % (level_key, len(recs), len(live)))
        for probe, rec in zip(live, recs):
            if isinstance(probe, FixedProbe):
                assert rec.kind == "fixed", rec.question_id
                assert probe._probe_type == rec.probe_type, rec.question_id
                assert probe._question == rec.question, rec.question_id
                assert probe._ground_truth == rec.ground_truth, rec.question_id
                checked += 1
            else:
                assert rec.kind == "generated", rec.question_id
                assert type(probe).__name__ == rec.probe_class, rec.question_id
    assert checked > 300, "expected to verify hundreds of fixed probes, checked %d" % checked


def test_no_machine_paths_leaked_into_the_bank(manifest):
    """A published dataset must not carry anyone's home directory."""
    for entry in manifest["tracks"].values():
        for rec in read_bank(BANK_DIR / entry["file"]):
            blob = rec.to_json()
            assert "/Users/" not in blob, rec.question_id
            assert "/home/" not in blob, rec.question_id


def test_schema_rejects_bad_records():
    with pytest.raises(ValueError):
        QuestionRecord(question_id="x", track="nope", suite="s",
                       task_or_level="l", kind="fixed", probe_type="p", question="q")
    with pytest.raises(ValueError):
        QuestionRecord(question_id="x", track="grid", suite="s",
                       task_or_level="l", kind="fixed", probe_type="p")  # no question
    with pytest.raises(ValueError):
        QuestionRecord(question_id="x", track="grid", suite="s", task_or_level="l",
                       kind="fixed", probe_type="p", question="q", cognitive_tier="Z")


def test_innav_matches_static_only_where_static_is_also_generated(manifest):
    """InNav can only reuse static probes where those are generated, not fixed.

    The static gridworld benchmark uses FixedProbe on 30 of 33 levels, and
    those questions hardcode egocentric spatial references ("11 steps ahead
    and 3 to your right") that presuppose a known agent position. InNav's
    agent navigates to a model-dependent position, so such questions refer to
    nothing there. It follows that InNav can only match the static probe set
    on the levels where the static set is itself generated.

    This asserts that equivalence exactly: the levels where the two agree are
    precisely the levels the static benchmark builds without FixedProbe. If
    that stops holding, either the dispatch or the static probe definitions
    changed in a way that needs looking at.
    """
    import random

    import halluworld.tracks.grid.perception as perception
    from halluworld.tracks.grid.probes.visibility import FixedProbe
    from halluworld.tracks.innav.canonical_probes import make_canonical_probes

    matching, generated_only = set(), set()
    for level_key in perception.LEVELS:
        static = perception.make_probes(level_key, random.Random(0))
        innav = make_canonical_probes(level_key, random.Random(0))
        if not any(isinstance(p, FixedProbe) for p in static):
            generated_only.add(level_key)
        if [type(p).__name__ for p in innav] == [type(p).__name__ for p in static]:
            matching.add(level_key)

    assert matching == generated_only, (
        "InNav matches the static set on %s but the static set is "
        "generated-only on %s; these should be identical."
        % (sorted(matching), sorted(generated_only)))
    assert matching == {"P1_dense_array", "P2_corridor_gauntlet",
                        "P3_rotation_challenge"}, sorted(matching)


# --------------------------------------------------------------------------- #
# v0.2 regeneration                                                            #
# --------------------------------------------------------------------------- #

def _build_chess_from(tmp_path, rows, env):
    """Run build_chess() against a synthetic export directory."""
    import sys

    sys.path.insert(0, str(DATA_DIR.parent.parent / "scripts"))
    from build_question_bank import build_chess

    src = tmp_path / "export"
    src.mkdir(parents=True)
    (src / "questions.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    if env is not None:
        (src / "generation_config.json").write_text(
            json.dumps({"exported": "2026-09-13", "env": env}), encoding="utf-8"
        )
    return build_chess(str(src))


_ROW = {
    "probe": "chess_can_capture",
    "env_reset_id": 0,
    "env_seed": 1,
    "question": "Can white legally capture the black knight on f6?",
    "ground_truth": "yes",
    "user_prompt": "board",
    "metadata": {"model": "stub"},
}


def test_regenerated_chess_ids_do_not_collide_with_v0_1(tmp_path):
    """A regenerated bank must not reuse v0.1's question ids.

    env_reset_id restarts at 0 every run, so without a namespace a v0.2 bank
    carries v0.1's exact ids while holding different questions -- and any
    result keyed by question_id would silently conflate the two banks.
    """
    v01 = _build_chess_from(tmp_path / "a", [_ROW], env=None)
    v02 = _build_chess_from(tmp_path / "b", [_ROW], env={"BENCHMARK_SEED": "20250101"})

    assert v01[0].question_id == "chess/chess_can_capture/e00"
    assert v02[0].question_id == "chess/s20250101/chess_can_capture/e00"
    assert not {r.question_id for r in v01} & {r.question_id for r in v02}


def test_regenerated_chess_records_its_generation_config(tmp_path):
    """How a held-out bank was made stays reproducible even if it stays private."""
    env = {"BENCHMARK_SEED": "20250101", "STRESS_MODE": "1",
           "OBSERVATION_FEN_MODE": "transpose"}
    records = _build_chess_from(tmp_path, [_ROW], env=env)

    provenance = records[0].provenance
    assert provenance["generation_env"] == env
    assert provenance["benchmark_seed"] == 20250101
    assert provenance["fen_mode"] == "transpose"


def test_v0_1_chess_provenance_is_unchanged_without_a_config(tmp_path):
    """v0.1 has no generation_config.json and must keep its frozen facts."""
    provenance = _build_chess_from(tmp_path, [_ROW], env=None)[0].provenance

    assert provenance["benchmark_seed"] == 42
    assert provenance["fen_mode"] == "fen-off"
    assert provenance["exported"] == "2026-05-05"
    assert "generation_env" not in provenance
