"""benchmark.py — top-level orchestration for a HalluWorld evaluation run.

Typical usage::

    from halluworld.tracks.grid import make_env
    from halluworld.tracks.grid import SymbolicSerializer
    from halluworld.tracks.grid import PresenceProbe, LocationProbe
    from halluworld.lm import StubLM
    from halluworld.tracks.grid import PresenceEvaluator, LocationEvaluator
    from halluworld.benchmark import run_benchmark

    results = run_benchmark(
        env=make_env(render_mode="rgb_array", agent_view_size=7),
        serializer=SymbolicSerializer(),
        probes=[PresenceProbe(), LocationProbe()],
        lm=StubLM(mode="random", seed=0),
        evaluators=[PresenceEvaluator(), LocationEvaluator()],
        n_episodes=50,
        seed=42,
        verbose=True,
    )
    print(results.summary())
    results.to_jsonl("results.jsonl")
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd
from tqdm import tqdm
from minigrid.minigrid_env import MiniGridEnv

from halluworld.evaluator import Evaluator
from halluworld.lm.base import LM
from halluworld.probe import Probe
from halluworld.serializer import Serializer

# System prompt used for every LM query.
SYSTEM_PROMPT = (
    "You are an agent navigating a gridworld. Answer questions about your current "
    "observation concisely and accurately. Base your answers only on what you are "
    "told you can see. Do not infer or guess about things outside your view."
)


@dataclass
class EpisodeResult:
    """One probe trial — the atomic unit of the benchmark."""
    episode_id: int
    probe_name: str       # e.g. "presence", "location"
    question: str
    ground_truth: Any
    lm_response: str
    is_correct: bool | None
    score: float | None    # partial credit in [0, 1]; None means EXCLUDED, not wrong
    # None occurs when the LM call itself failed (LMResponse.error is set, e.g.
    # a reasoning model's token budget was too small for a 400 BadRequestError)
    # rather than the model giving a scoreable answer. accuracy()/summary()
    # drop these from their denominators the same way hallucination_rate()
    # already does; do not add a probe-skip path that leaves this at 0.0.
    metadata: dict = field(default_factory=dict)
    # metadata always includes: env_seed, agent_view_size, n_visible_objects,
    # steps_before_probe, model, and any probe-specific fields
    messages: list[dict] = field(default_factory=list)
    # messages contains the system/user/assistant conversation trace


@dataclass
class BenchmarkRun:
    """Collected results from a full benchmark run."""
    results: list[EpisodeResult] = field(default_factory=list)
    run_metadata: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Scalar metrics                                                       #
    # ------------------------------------------------------------------ #

    def accuracy(self, probe_name: str | None = None) -> float | None:
        """Fraction of scoreable trials the LM answered correctly.

        Returns 0.0 for no trials at all (preserves the pre-existing empty-run
        convention), but None when trials ran and every one was EXCLUDED
        (score=None -- an infrastructure failure, not a wrong answer). Do not
        collapse that second case back to 0.0: a run where every call hit a
        provider error is not a run that is 0% accurate.
        """
        all_rs = self._filter(probe_name)
        if not all_rs:
            return 0.0
        rs = [r for r in all_rs if r.is_correct is not None]
        return sum(r.is_correct for r in rs) / len(rs) if rs else None

    def hallucination_rate(self, probe_name: str | None = None) -> float | None:
        """Fraction of trials where the LM gave a wrong confident answer.

        For presence probes this is: predicted an object that wasn't there
        (False Positive) or denied one that was (False Negative).
        For location probes: any positional error.
        Excluded trials (score=None) are dropped from the denominator; see
        accuracy()'s docstring for why an all-excluded run returns None
        rather than 0.0.
        """
        all_rs = self._filter(probe_name)
        if not all_rs:
            return 0.0
        rs = [r for r in all_rs if r.score is not None]
        return (1.0 - (sum(r.score for r in rs) / len(rs))) if rs else None

    def mean_score(self, probe_name: str | None = None) -> float | None:
        all_rs = self._filter(probe_name)
        if not all_rs:
            return 0.0
        rs = [r for r in all_rs if r.score is not None]
        return sum(r.score for r in rs) / len(rs) if rs else None

    # ------------------------------------------------------------------ #
    # Structured output                                                    #
    # ------------------------------------------------------------------ #

    def summary(self) -> pd.DataFrame:
        """DataFrame with one row per probe type — accuracy, hallucination rate, mean score, n."""
        by_type: dict[str, list[EpisodeResult]] = {}
        for r in self.results:
            by_type.setdefault(r.probe_name, []).append(r)

        rows = []
        for pname, rs in sorted(by_type.items()):
            n = len(rs)
            scored = [r for r in rs if r.score is not None]
            excluded = n - len(scored)
            acc = sum(r.is_correct for r in scored) / len(scored) if scored else None
            h_rate = (1.0 - sum(r.score for r in scored) / len(scored)) if scored else None
            rows.append({
                "probe": pname,
                "n": n,
                "excluded": excluded,
                "accuracy": round(acc, 4) if acc is not None else None,
                "hallucination_rate": round(h_rate, 4) if h_rate is not None else None,
                "mean_score": round(sum(r.score for r in scored) / len(scored), 4) if scored else None,
            })

        total_n = len(self.results)
        if total_n:
            total_excluded = sum(1 for r in self.results if r.score is None)
            overall_acc = self.accuracy()
            overall_hrate = self.hallucination_rate()
            overall_mean = self.mean_score()
            rows.append({
                "probe": "ALL",
                "n": total_n,
                "excluded": total_excluded,
                "accuracy": round(overall_acc, 4) if overall_acc is not None else None,
                "hallucination_rate": round(overall_hrate, 4) if overall_hrate is not None else None,
                "mean_score": round(overall_mean, 4) if overall_mean is not None else None,
            })
        return pd.DataFrame(rows)

    def to_jsonl(self, path: str) -> None:
        """Write one JSON line per EpisodeResult for offline analysis."""
        with open(path, "w") as f:
            for r in self.results:
                d = asdict(r)
                # Ensure ground_truth is JSON-serialisable
                d["ground_truth"] = _jsonable(d["ground_truth"])
                f.write(json.dumps(d) + "\n")

    def save_structured(self, output_dir: str) -> None:
        """Save results with structured directory layout and per-episode message traces.

        Creates structure:
            {output_dir}/
                results.jsonl                    # All results in JSONL format
                episode_000_presence/
                    messages.jsonl               # System/user/assistant trace
                episode_000_location/
                    messages.jsonl
                episode_001_presence/
                    messages.jsonl
                ...

        Args:
            output_dir: Directory path to save results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save main results file
        self.to_jsonl(str(output_path / "results.jsonl"))

        # Save per-episode message traces
        for r in self.results:
            env_reset_id = r.metadata.get("env_reset_id", r.episode_id)
            episode_dir = output_path / f"episode_{env_reset_id:03d}_{r.probe_name}"
            episode_dir.mkdir(exist_ok=True)

            # Save messages trace
            with open(episode_dir / "messages.jsonl", "w") as f:
                for msg in r.messages:
                    f.write(json.dumps(msg) + "\n")

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _filter(self, probe_name: str | None) -> list[EpisodeResult]:
        if probe_name is None:
            return self.results
        return [r for r in self.results if r.probe_name == probe_name]


def run_benchmark(
    env: Any,
    serializer: Serializer,
    probes: list[Probe],
    lm: LM,
    evaluators: list[Evaluator],
    n_episodes: int = 50,
    steps_before_probe: int = 0,
    seed: int = 42,
    verbose: bool = False,
    system_prompt: str | None = None,
    include_obs: bool = True,
    warmup_fn=None,
) -> BenchmarkRun:
    """Run the full eval loop and return a BenchmarkRun.

    On each episode:
      1. env.reset() with a fresh seed
      2. Optionally take random warm-up steps to vary the visible state
      3. For each (probe, evaluator) pair: generate question, query LM, score

    Args:
        env:                Environment instance implementing reset() and optionally
                            action_space.sample() + step() for warm-up transitions.
        serializer:         Converts env state to string for the LM.
        probes:             List of Probe instances — one probe per query type.
        lm:                 Language model to query.
        evaluators:         List of Evaluator instances — paired with probes by index.
        n_episodes:         Number of independent env resets.
        steps_before_probe: Random actions before probing to vary visible state.
        seed:               Master RNG seed for reproducibility.
        verbose:            Print each trial result to stdout.
        include_obs:        If True, include serialized observation in the user prompt;
                            if False, send only the probe question.

    Returns:
        BenchmarkRun containing all EpisodeResults.
    """
    assert len(probes) == len(evaluators), (
        f"probes and evaluators must be the same length, "
        f"got {len(probes)} probes and {len(evaluators)} evaluators"
    )

    rng = random.Random(seed)
    episode_id = 0
    run = BenchmarkRun(run_metadata={
        "n_episodes": n_episodes,
        "steps_before_probe": steps_before_probe,
        "seed": seed,
        "agent_view_size": getattr(env, "agent_view_size", None),
        "include_obs": include_obs,
    })

    for env_reset_id in tqdm(range(n_episodes), desc="episodes", unit="ep"):
        env_seed = rng.randint(0, 2**31)
        env.reset(seed=env_seed)

        # Warm-up steps; warmup_fn(env) -> action_idx overrides random sampling
        for _ in range(steps_before_probe):
            if not hasattr(env, "action_space") or not hasattr(env, "step"):
                break
            action = warmup_fn(env) if warmup_fn is not None else env.action_space.sample()
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                env.reset(seed=rng.randint(0, 2**31))

        obs_text = serializer.serialize(env)

        for probe, evaluator in zip(probes, evaluators):
            probe_result = probe.generate(env)

            # Probe returns None when it can't construct a valid question (e.g.
            # empty FOV for LocationProbe). Log a skip and continue.
            if probe_result is None or probe_result.metadata.get("skipped"):
                episode_id += 1
                continue

            if (
                probe_result.prepend_env_observation
                and probe_result.observation_override is not None
            ):
                obs_for_probe = (
                    obs_text.rstrip() + "\n\n---\n\n" + probe_result.observation_override
                )
            else:
                obs_for_probe = probe_result.observation_override or obs_text

            if include_obs:
                user_prompt = (
                    f"## Current observation\n{obs_for_probe}\n\n"
                    f"## Question\n{probe_result.question}"
                )
            else:
                user_prompt = probe_result.question

            response = lm.query(system=system_prompt or SYSTEM_PROMPT, user=user_prompt)

            base_metadata = {
                "env_reset_id": env_reset_id,
                "env_seed": env_seed,
                "agent_view_size": getattr(env, "agent_view_size", None),
                "steps_before_probe": steps_before_probe,
                "model": response.model,
                **probe_result.metadata,
            }
            messages = [
                {"role": "system", "content": system_prompt or SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": response.text},
            ]

            if response.error:
                # The call itself failed (e.g. a reasoning model's token budget
                # was too small for a 400 BadRequestError) -- there is no answer
                # to evaluate. Record the trial as EXCLUDED (score=None), not as
                # a wrong answer: running it through the evaluator would score
                # an infrastructure failure as a measured hallucination.
                result = EpisodeResult(
                    episode_id=episode_id,
                    probe_name=probe_result.probe_type,
                    question=probe_result.question,
                    ground_truth=probe_result.ground_truth,
                    lm_response=response.text,
                    is_correct=None,
                    score=None,
                    metadata={**base_metadata, "error": response.error},
                    messages=messages,
                )
                run.results.append(result)
                if verbose:
                    print(f"[ep {episode_id:4d}][{probe_result.probe_type:10s}] "
                          f"EXCLUDED ({response.error}) "
                          f"gt={str(probe_result.ground_truth):<12}")
                episode_id += 1
                continue

            eval_result = evaluator.evaluate(response, probe_result)

            result = EpisodeResult(
                episode_id=episode_id,
                probe_name=probe_result.probe_type,
                question=probe_result.question,
                ground_truth=probe_result.ground_truth,
                lm_response=response.text,
                is_correct=eval_result.correct,
                score=eval_result.score,
                metadata={
                    **base_metadata,
                    **{k: v for k, v in eval_result.details.items()
                       if k not in probe_result.metadata},
                },
                messages=messages,
            )
            run.results.append(result)

            if verbose:
                tag = "✓" if eval_result.correct else "✗"
                print(f"[ep {episode_id:4d}][{probe_result.probe_type:10s}] {tag} "
                      f"gt={str(probe_result.ground_truth):<12} "
                      f"pred={str(eval_result.predicted):<12} "
                      f"lm={response.text[:50]!r}")

            episode_id += 1

    return run


def _jsonable(v: Any) -> Any:
    """Recursively make a value JSON-serialisable."""
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, (bool, int, float, str)) or v is None:
        return v
    return str(v)

