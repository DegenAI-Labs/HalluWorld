"""YAML run configuration, replacing the chess battery's 73 environment variables.

WHY THE CONFIG MATERIALIZES INTO THE ENVIRONMENT

tracks/chess/battery.py reads its configuration from 73 environment variables
across ~1200 lines. Rewriting all of those call sites to consume a config
object would be a large, high-risk diff through the exact code that implements
the determinism contract -- the seeding, the replay logic, the probe RNGs.

So this module does the opposite: it validates a YAML file and then exports it
back into os.environ before the battery runs. The battery is untouched, its
behavior is identical by construction, and the fidelity check ("does the YAML
path produce the same questions as the env-var path") is satisfied trivially
rather than argued.

What that buys, which raw environment variables did not:

  * typo detection. `SAN_MIN_PLYS=8` was silently ignored forever, because
    os.environ.get returns the default for a key nobody set. An unknown key in
    a YAML file is an error here.
  * a place for comments. A config file explaining what `prefer_ghost_captures`
    means is documentation the shell export line could never be.
  * one artifact to record in a run manifest, hash, and diff.

Refactoring battery.py to take a config object directly is worth doing later.
It is not worth doing at the same time as everything else in this release.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

#: Every environment variable the chess battery reads, minus the API keys.
#: Keys are exported as `SECTION.key` in YAML and flattened to the variable
#: name on the way out; see FLAT_KEYS.
CHESS_ENV_KEYS = {
    # run
    "N_EPISODES", "STEPS_BEFORE_PROBE", "VALIDATE_RESETS", "BENCHMARK_VERBOSE",
    "RESULTS_SUBDIR", "USE_STUB_LM", "HALLUWORLD_RESULTS_ROOT",
    "HALLUWORLD_RESULTS_DIR",
    # provider
    "LM_PROVIDER",
    "OPENAI_MODEL", "OPENAI_REASONING_EFFORT", "OPENAI_MAX_COMPLETION_TOKENS",
    "OPENAI_TEMPERATURE",
    "ANTHROPIC_MODEL", "ANTHROPIC_THINKING_EFFORT", "ANTHROPIC_TEMPERATURE",
    "ANTHROPIC_MAX_TOKENS",
    "BASETEN_MODEL", "BASETEN_TEMPERATURE", "BASETEN_MAX_TOKENS",
    "BASETEN_BASE_URL",
    # environment
    "CHESS_VARIANT", "CHESS_EXTREME", "STRESS_MODE", "USE_LICHESS", "LICHESS_SAMPLES",
    # Which puzzles USE_LICHESS draws. Previously hardcoded, so a held-out set
    # could not get new positions out of the corpus; see lichess_pool_kwargs().
    "LICHESS_SEED", "LICHESS_MIN_RATING", "LICHESS_MAX_RATING", "LICHESS_THEMES",
    "ATOMIC_HF_FENS_CACHE", "ATOMIC_HF_MAX_PLIES", "ATOMIC_HF_MIN_PLIES",
    "ATOMIC_HF_MAX_SCAN_ROWS", "ATOMIC_HF_SEED", "ATOMIC_HF_TARGET_FENS",
    "ATOMIC_WARMUP_STEPS", "OLD_BISHOP_WARMUP_STEPS",
    # observation
    "INCLUDE_FEN", "OBSERVATION_FEN_MODE", "OBSERVATION_FEN_OVERRIDE",
    "OBSERVATION_FEN_SEED",
    "CHESS_CHAT_MIN_GAMES", "CHESS_CHAT_MAX_GAMES", "CHESS_CHAT_SAN_MIN_PLIES",
    "CHESS_CHAT_SAN_MAX_PLIES", "CHESS_CHAT_RUMOR_RATE", "CHESS_CHAT_LOBBY_LINES",
    "CHESS_CHAT_PREFIX_STACK", "CHESS_CHAT_POST_OBS_LINES",
    # probes
    "SQUARE_SCOPE", "ENABLE_BENCHMARKS", "DISABLE_BENCHMARKS",
    "SAN_N_PLIES", "SAN_MIN_PLIES", "SAN_CAPTURE_BIAS", "SAN_MAX_REPLAY_ATTEMPTS",
    "SAN_OBSERVATION_MODE", "SAN_REPLY_CONSTRAINT", "SAN_MAX_CONSTRAINED_REPLIES",
    "CAPTURE_COUNT_N_PLIES", "CAPTURE_COUNT_CAPTURE_BIAS",
    "HARD_HIDDEN_MIN_PLIES", "HARD_HIDDEN_MAX_PLIES", "HARD_HIDDEN_CAPTURE_BIAS",
    "MATE_STALE_MAX_PLIES", "MATE_STALE_CAPTURE_BIAS", "MATE_STALE_MAX_ATTEMPTS",
    "UNDEFENDED_COUNT_MODE", "UNDEFENDED_HYPOTHESIS_PLIES", "UNDEFENDED_MAX_TRIES",
    "TWO_STEP_HANG_MAX_TRIES", "HISTORY_READOUT_N_PLIES",
    "CONFLICTING_PROMPT_LIE", "CONFLICTING_PROMPT_FALSE_FEN",
    "ATOMIC_OBS_MODE", "ATOMIC_HYPOTHESIS_PLIES_MIN", "ATOMIC_HYPOTHESIS_PLIES_MAX",
    "ATOMIC_UCI_FAKE_FIDE_RATE", "ATOMIC_UCI_MINIMAL_RULES_RATE",
    "OLD_BISHOP_UCI_FAKE_FIDE_RATE", "OLD_BISHOP_UCI_MINIMAL_RULES_RATE",
}

#: Never sourced from a config file. Credentials come from the environment
#: only, so a config can be committed and shared without becoming a secret.
FORBIDDEN_KEYS = {"OPENAI_API_KEY", "ANTHROPIC_API_KEY", "BASETEN_API_KEY"}


class ConfigError(ValueError):
    """Raised for an unknown key, a forbidden key, or a malformed file."""


def _flatten(section: dict, prefix: str = "") -> dict:
    out = {}
    for key, value in (section or {}).items():
        if isinstance(value, dict):
            out.update(_flatten(value, prefix))
        else:
            out[key] = value
    return out


def _as_env_value(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if value is None:
        return ""
    return str(value)


def load(path: str | Path, _seen: tuple = ()) -> dict:
    """Load and validate a run config. Returns the flattened key/value mapping.

    `extends: other.yaml` (resolved relative to this file) merges the parent
    first, so a child only states what it changes. That is what keeps the three
    observation conditions down to the two or three lines that actually differ
    between them, instead of three near-identical 14-key files that can drift.
    """
    path = Path(path)
    if str(path.resolve()) in _seen:
        raise ConfigError("circular extends chain at %s" % path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError("%s: %s" % (path, exc)) from exc
    if not isinstance(raw, dict):
        raise ConfigError("%s: top level must be a mapping" % path)

    meta = {k: raw.pop(k) for k in ("track", "suite", "description", "extends")
            if k in raw}
    flat = _flatten(raw)

    parent = meta.get("extends")
    if parent:
        base = load(path.parent / parent, _seen + (str(path.resolve()),))
        base.pop("_meta", None)
        base.update(flat)          # child wins
        flat = base

    unknown = sorted(set(flat) - CHESS_ENV_KEYS - FORBIDDEN_KEYS)
    if unknown:
        raise ConfigError(
            "%s: unknown key(s): %s\n"
            "This is the whole point of the config file -- a typo like "
            "SAN_MIN_PLYS was silently ignored as an environment variable."
            % (path, ", ".join(unknown)))

    forbidden = sorted(set(flat) & FORBIDDEN_KEYS)
    if forbidden:
        raise ConfigError(
            "%s: %s must come from the environment, never a config file."
            % (path, ", ".join(forbidden)))

    flat["_meta"] = meta
    return flat


def apply_to_environ(config: dict, env: dict | None = None) -> dict:
    """Export a loaded config into an environment mapping.

    Returns the mapping that was written to, so a caller can pass a copy of
    os.environ rather than mutating the process. Existing values are
    overwritten: the config is the source of truth for a run.
    """
    env = os.environ if env is None else env
    for key, value in config.items():
        if key == "_meta":
            continue
        env[key] = _as_env_value(value)
    return env


def from_environ(env: dict | None = None) -> dict:
    """Snapshot the currently-set config variables, for `config from-env`.

    Only keys that are actually set are captured, so the result is what a run
    would differ by rather than a dump of every default.
    """
    env = os.environ if env is None else env
    return {k: env[k] for k in sorted(CHESS_ENV_KEYS) if k in env}


def to_yaml(config: dict, track: str = "chess", suite: str = "chess_standard") -> str:
    """Render a flat config back to YAML, grouped for readability."""
    groups = {
        "run": ("N_EPISODES", "STEPS_BEFORE_PROBE", "VALIDATE_RESETS",
                "BENCHMARK_VERBOSE", "RESULTS_SUBDIR", "USE_STUB_LM"),
        "provider": tuple(k for k in sorted(CHESS_ENV_KEYS)
                          if k.startswith(("LM_", "OPENAI_", "ANTHROPIC_", "BASETEN_"))),
        "env": tuple(k for k in sorted(CHESS_ENV_KEYS)
                     if k.startswith(("CHESS_VARIANT", "CHESS_EXTREME", "STRESS_",
                                      "USE_LICHESS", "LICHESS_", "ATOMIC_HF_",
                                      "ATOMIC_WARMUP", "OLD_BISHOP_WARMUP"))),
        "observation": tuple(k for k in sorted(CHESS_ENV_KEYS)
                             if k.startswith(("INCLUDE_FEN", "OBSERVATION_", "CHESS_CHAT_"))),
    }
    used = {k for ks in groups.values() for k in ks}
    groups["probes"] = tuple(k for k in sorted(CHESS_ENV_KEYS) if k not in used)

    doc: dict[str, Any] = {"track": track, "suite": suite}
    for name, keys in groups.items():
        section = {k: config[k] for k in keys if k in config}
        if section:
            doc[name] = section
    return yaml.safe_dump(doc, sort_keys=False, default_flow_style=False)
