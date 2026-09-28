"""HalluWorld: measuring hallucination in LLM agents across four tracks.

The package is organized by track. Everything specific to a track lives under
``halluworld.tracks.<name>``; the shared abstractions and run loops live here
at the top level.

Shared
    benchmark.py    run_benchmark, BenchmarkRun, EpisodeResult
    multiturn.py    the multi-turn run loop
    probe.py        Probe, ProbeResult
    serializer.py   Serializer
    evaluator.py    Evaluator, EvalResult
    lm/             OpenAI, Anthropic, Baseten, and stub providers
    data/           packaged levels, probe banks, and terminal tasks

Tracks
    tracks.grid      MiniGrid levels across the P/M/C/U/X tiers
    tracks.chess     chess variants: standard, atomic, old-bishop
    tracks.innav     retrospective paired probing during navigation
    tracks.terminal  Docker/tmux terminal agents with runtime probe injection

A benchmark run composes five pieces, one per axis:

    run_benchmark(env=..., serializer=..., probes=[...], lm=..., evaluators=[...])

Typical use:

    from halluworld import run_benchmark
    from halluworld.tracks.grid import SymbolicSerializer, PresenceProbe, PresenceEvaluator
"""

from halluworld.benchmark import BenchmarkRun, EpisodeResult, run_benchmark
from halluworld.evaluator import EvalResult, Evaluator
from halluworld.probe import Probe, ProbeResult
from halluworld.serializer import Serializer

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "run_benchmark",
    "BenchmarkRun",
    "EpisodeResult",
    "Probe",
    "ProbeResult",
    "Serializer",
    "Evaluator",
    "EvalResult",
]
