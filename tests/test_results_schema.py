"""Guards for the unified result schema and the legacy migration.

The migration touches every number this project has ever reported, so the tests
that matter are the ones proving it changed none of them.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from halluworld.results import COLUMNS, ProbeRecord, read_jsonl, summarize, write_jsonl

REPO = Path(__file__).resolve().parent.parent
MIGRATED = REPO / "results" / "migrated" / "v1" / "records.jsonl"


def test_schema_has_the_documented_column_set():
    assert len(COLUMNS) == 30
    for required in ("track", "level", "variant", "score", "pair_id", "extra"):
        assert required in COLUMNS


def test_score_none_and_zero_are_distinct():
    """None means excluded, 0.0 means wrong. Conflating them turns an outage
    into a reported hallucination rate."""
    wrong = ProbeRecord(track="grid", score=0.0)
    excluded = ProbeRecord(track="grid", score=None, error="bad_request")
    assert wrong.score == 0.0 and excluded.score is None
    stats = summarize([wrong, excluded])
    key = next(iter(stats))
    assert stats[key]["n"] == 2
    assert stats[key]["scored"] == 1
    assert stats[key]["excluded"] == 1
    assert stats[key]["accuracy"] == 0.0      # the excluded row did not drag it down


def test_invalid_records_are_rejected():
    with pytest.raises(ValueError):
        ProbeRecord(track="nope")
    with pytest.raises(ValueError):
        ProbeRecord(track="grid", score=1.5)
    with pytest.raises(ValueError):
        ProbeRecord(track="grid", error="not_a_known_error")


def test_jsonl_round_trip_survives_embedded_newlines(tmp_path):
    """Why JSONL is canonical and CSV is only a view: questions and responses
    contain newlines, and CSV round-tripping them is what produced the
    'corrupted CSVs' this project had to recover from."""
    rec = ProbeRecord(track="grid", question="line one\nline two",
                      response="a\nb\r\nc", score=1.0)
    p = tmp_path / "r.jsonl"
    write_jsonl([rec], p)
    back = list(read_jsonl(p))
    assert len(back) == 1
    assert back[0].question == rec.question
    assert back[0].response == rec.response


@pytest.mark.skipif(not MIGRATED.exists(), reason="run scripts/migrate_results.py first")
class TestMigratedCorpus:
    @pytest.fixture(scope="class")
    @classmethod
    def records(cls):
        return list(read_jsonl(MIGRATED))

    def test_every_innav_pair_has_exactly_two_arms(self, records):
        """The 1->2 fan-out must be exact. A pair with one arm is a dropped
        observation; a pair with three is a colliding pair_id -- which is a bug
        this test caught, when run_id was derived from the file stem and four
        stems collided across directories."""
        import collections

        pairs = collections.Counter(r.pair_id for r in records if r.track == "innav")
        assert pairs, "no innav records"
        bad = {k: v for k, v in pairs.items() if v != 2}
        assert not bad, "pairs without exactly 2 arms: %s" % list(bad.items())[:5]

    def test_both_arms_present_for_every_pair(self, records):
        import collections

        arms = collections.defaultdict(set)
        for r in records:
            if r.track == "innav":
                arms[r.pair_id].add(r.variant)
        assert all(v == {"innav", "static_control"} for v in arms.values())

    def test_run_ids_are_unique_per_source_file(self, records):
        assert len({r.run_id for r in records}) == 849

    def test_no_record_violates_the_schema(self, records):
        assert len(records) > 100_000
        for r in records[:5000]:
            assert r.track in ("grid", "chess", "innav", "terminal")
            assert r.score is None or 0.0 <= r.score <= 1.0


LEGACY = REPO / "results" / "legacy"


@pytest.mark.skipif(not LEGACY.is_dir(),
                    reason="legacy result corpus is gitignored and not shipped; "
                           "run this where results/legacy/ exists")
def test_migration_preserves_aggregates():
    """The guard that migration changed no published number.

    Re-runs the migration in --check mode, which compares the mean score of
    every source file against the mean of its migrated records.

    Skipped in a published checkout: the legacy corpus is deliberately not
    shipped, so there is nothing to migrate there. This is a data-dependent
    test, and skipping is the honest outcome rather than failing on absent
    input.
    """
    out = subprocess.run(
        [sys.executable, "scripts/migrate_results.py", "--in", "results/legacy", "--check"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, cwd=REPO)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "aggregate preservation: OK" in out.stdout


def test_unknown_header_raises_rather_than_guessing(tmp_path):
    """A silently skipped file is a silently missing number."""
    bad = tmp_path / "weird.csv"
    bad.write_text("alpha,beta,gamma\n1,2,3\n")
    out = subprocess.run(
        [sys.executable, "scripts/migrate_results.py", "--in", str(tmp_path), "--check"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, cwd=REPO)
    assert out.returncode != 0
    assert "unrecognised header" in (out.stdout + out.stderr)
