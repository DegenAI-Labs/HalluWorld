"""Guards for the YAML run config and the CLI.

The config layer exists to replace 73 environment variables. The risk it
introduces is that a config file silently differs from the shell invocation it
replaces, so the tests that matter are the fidelity ones.
"""

import subprocess
import sys
from pathlib import Path

import pytest

from halluworld import config as cfgmod

REPO = Path(__file__).resolve().parent.parent
from halluworld.data import DATA_DIR

CHESS_CONFIGS = DATA_DIR / "configs" / "chess"
CONFIGS = sorted(CHESS_CONFIGS.glob("*.yaml"))


def test_every_shipped_config_loads():
    assert CONFIGS, "no configs found"
    for path in CONFIGS:
        cfg = cfgmod.load(path)
        assert cfg, path.name


def test_config_declares_every_variable_the_battery_reads():
    """A variable the battery reads but the config cannot express is a hole.

    Anyone converting a shell invocation would lose that setting silently, and
    the run would use a default nobody chose.
    """
    import re

    src = (REPO / "halluworld" / "tracks" / "chess" / "battery.py").read_text()
    read = set(re.findall(r'os\.environ(?:\.get)?[\(\[]"([A-Z_0-9]+)"', src))
    declared = cfgmod.CHESS_ENV_KEYS | cfgmod.FORBIDDEN_KEYS
    assert not (read - declared), "battery reads undeclared vars: %s" % sorted(read - declared)


def test_unknown_key_is_rejected():
    """The reason the config exists: SAN_MIN_PLYS was silently ignored forever."""
    with pytest.raises(cfgmod.ConfigError, match="unknown key"):
        cfgmod.load(_tmp_yaml("track: chess\nprobes:\n  SAN_MIN_PLYS: 8\n"))


def test_api_keys_cannot_come_from_a_config_file():
    with pytest.raises(cfgmod.ConfigError, match="must come from the environment"):
        cfgmod.load(_tmp_yaml("track: chess\nprovider:\n  OPENAI_API_KEY: sk-nope\n"))


def test_extends_merges_parent_and_child_wins():
    base = cfgmod.load(CHESS_CONFIGS / "_base.yaml")
    child = cfgmod.load(CHESS_CONFIGS / "fen_transpose.yaml")
    # inherited
    assert child["N_EPISODES"] == base["N_EPISODES"]
    # child-only
    assert child["OBSERVATION_FEN_MODE"] == "transpose"
    assert child["OBSERVATION_FEN_SEED"] == 7


def test_fen_configs_reproduce_the_documented_invocation():
    """Fidelity gate: the YAML must set exactly what the documented run set.

    Transcribed from the canonical invocation in docs/CHESS.md plus the FEN_MODES
    table in scripts/run_chess_benchmark.sh. Anything the documented run left
    to a default must stay unset here -- pinning it would decouple it from the
    difficulty setting it is derived from. VALIDATE_RESETS is the live example:
    the battery computes 55 when CHESS_EXTREME=1, and an earlier draft of this
    config hardcoded 3.
    """
    documented = {
        "CHESS_EXTREME": "1",
        "HARD_HIDDEN_MIN_PLIES": "70",
        "HARD_HIDDEN_MAX_PLIES": "120",
        "HARD_HIDDEN_CAPTURE_BIAS": "0.62",
        "USE_LICHESS": "1",
        "N_EPISODES": "50",
        "BENCHMARK_VERBOSE": "1",
        "INCLUDE_FEN": "1",
        "OBSERVATION_FEN_MODE": "transpose",
        "OBSERVATION_FEN_SEED": "7",
        "OPENAI_MODEL": "o3",
        "OPENAI_REASONING_EFFORT": "medium",
        "OPENAI_MAX_COMPLETION_TOKENS": "16384",
    }
    produced = cfgmod.apply_to_environ(
        cfgmod.load(CHESS_CONFIGS / "fen_transpose.yaml"), env={})
    for key, want in documented.items():
        assert produced.get(key) == want, "%s: want %r got %r" % (key, want, produced.get(key))
    # LM_PROVIDER is stated explicitly but equals the battery's own default.
    assert set(produced) - set(documented) == {"LM_PROVIDER"}
    assert produced["LM_PROVIDER"] == "openai"

    for derived in ("VALIDATE_RESETS", "STEPS_BEFORE_PROBE", "CHESS_VARIANT"):
        assert derived not in produced, (
            "%s must stay unset so the battery derives it from CHESS_EXTREME" % derived)


def test_round_trip_env_to_yaml_to_env():
    env = {"N_EPISODES": "50", "CHESS_EXTREME": "1", "INCLUDE_FEN": "0",
           "OPENAI_MODEL": "o3", "SAN_N_PLIES": "16"}
    captured = cfgmod.from_environ(env)
    assert captured == env
    restored = cfgmod.apply_to_environ(cfgmod.load(_tmp_yaml(cfgmod.to_yaml(captured))), env={})
    assert restored == env


def _tmp_yaml(text: str) -> Path:
    import tempfile

    fh = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    fh.write(text)
    fh.close()
    return Path(fh.name)


@pytest.mark.parametrize("argv", [
    ["version"],
    ["questions", "stats"],
    ["questions", "verify"],
    ["release", "check"],
    ["config", "check"] + [str(p) for p in CONFIGS],
    ["run", "--config", str(CHESS_CONFIGS / "fen_off.yaml"), "--dry-run"],
])
def test_cli_commands_succeed(argv):
    out = subprocess.run([sys.executable, "-m", "halluworld", *argv],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         universal_newlines=True, cwd=REPO)
    assert out.returncode == 0, "%s failed:\n%s" % (argv, out.stderr)


def test_results_never_land_inside_the_package(tmp_path, monkeypatch):
    """Regression: the battery once wrote output into the installed package.

    _results_dir used Path(__file__).parents[1], which was the repository root
    when this module lived at examples/chess_full_probes.py. Moving it to
    halluworld/tracks/chess/ silently redirected every run into
    halluworld/tracks/results/. Counting parent directories is what broke, so
    the root is now the working directory or HALLUWORLD_RESULTS_ROOT, and
    never derived from the module's own depth.
    """
    from halluworld.tracks.chess import battery

    monkeypatch.setenv("HALLUWORLD_RESULTS_ROOT", str(tmp_path))
    monkeypatch.setenv("RESULTS_SUBDIR", "unit")
    out = battery._results_dir()
    assert str(out).startswith(str(tmp_path)), out
    pkg = Path(battery.__file__).resolve().parent
    assert not str(out).startswith(str(pkg)), "results dir is inside the package: %s" % out


def test_results_root_is_a_configurable_key():
    assert "HALLUWORLD_RESULTS_ROOT" in cfgmod.CHESS_ENV_KEYS
