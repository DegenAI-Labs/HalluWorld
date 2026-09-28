#!/usr/bin/env python3
"""Legacy artifact evaluator for the historical Terminal leaderboard.

New release evaluations must use ``halluworld terminal eval``. This script is retained to document
and inspect the unpublished historical run format; its older permissive scoring behavior is not the
release contract.

Mirrors the interface of `halluworld eval`:
use --provider to select the backend, --model for the exact model name,
and provider-specific flags (--reasoning-effort / --thinking-effort) as needed.

Examples:
  # Claude Sonnet (Anthropic)
  python3 tools/evaluate_probes.py \
    --provider anthropic \
    --probe-dir extracted_llm_probes_20260504 \
    --out-dir runs/probe_eval_sonnet \
    --model claude-sonnet-4-6 \
    -n 8

  # GPT-5.5 with high reasoning (OpenAI)
  python3 tools/evaluate_probes.py \
    --provider openai \
    --probe-dir extracted_llm_probes_20260504 \
    --out-dir runs/probe_eval_gpt55 \
    --model gpt-5.5 \
    --reasoning-effort high \
    -n 8

  # gpt-4o-mini as cheap difficulty filter
  python3 tools/evaluate_probes.py \
    --provider openai \
    --probe-dir extracted_llm_probes_20260504 \
    --out-dir runs/probe_eval_4omini \
    --model gpt-4o-mini \
    -n 16

  # Claude Opus
  python3 tools/evaluate_probes.py \
    --provider anthropic \
    --probe-dir extracted_llm_probes_20260504 \
    --out-dir runs/probe_eval_opus \
    --model claude-opus-4-6

  # Dry-run (no API calls, for testing)
  python3 tools/evaluate_probes.py \
    --provider openai \
    --probe-dir extracted_llm_probes_20260504 \
    --out-dir /tmp/dry \
    --model gpt-4o-mini \
    --dry-run --limit 5
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import re
import ssl
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-6"
DEFAULT_OPENAI_API_BASE = "https://api.openai.com/v1"
DEFAULT_MAX_OUTPUT_TOKENS = 256

SYSTEM_PROMPT = (
    "Answer probe questions with only the exact requested answer string. "
    "Do not include prose, markdown, or explanation."
)

PROBE_USER_PROMPT = (
    "You are answering a Terminal-Bench runtime probe.\n"
    "Use only the terminal context below. The context is captured immediately "
    "before the current command executes and contains only the terminal pane "
    "visible at that time. If the question asks about previous commands, answer "
    "from the commands visible in this context only.\n"
    "Return only the requested answer string in the exact requested schema. Do not explain.\n\n"
    "TERMINAL CONTEXT:\n"
    "<<<CONTEXT\n"
    "{context}\n"
    "CONTEXT>>>\n\n"
    "QUESTION:\n{question}\n"
)

ASSIGNMENT_RE = re.compile(
    r"\b([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:<\s*)?([A-Za-z0-9_./:+-]+)(?:\s*>)?"
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def find_probe_files(probe_dir: Path) -> list[Path]:
    return sorted(
        p for p in probe_dir.glob("*.json")
        if p.name != "summary.json" and p.is_file()
    )


def result_path_for_probe(out_dir: Path, probe_file: Path) -> Path:
    return out_dir / f"{probe_file.stem}.result.json"


def build_prompt(probe: dict[str, Any], max_context_chars: int | None) -> str:
    context = probe.get("context") or ""
    if max_context_chars and max_context_chars > 0 and len(context) > max_context_chars:
        omitted = len(context) - max_context_chars
        context = f"[Context truncated: omitted {omitted} leading characters.]\n" + context[-max_context_chars:]
    return PROBE_USER_PROMPT.format(context=context, question=probe.get("question", ""))


def parse_assignments(text: str) -> dict[str, str]:
    return {k: v.strip("`'\".,;") for k, v in ASSIGNMENT_RE.findall(text or "")}


def schema_keys(answer_schema: str | None) -> list[str]:
    if not answer_schema:
        return []
    keys: list[str] = []
    for key in re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*=", answer_schema):
        if key not in keys:
            keys.append(key)
    return keys


def normalize_value(value: str | None) -> str:
    v = (value or "").strip().strip("`'\".,;").lower()
    # Treat leading-zero variants as equal: "0121" == "121"
    try:
        return str(int(v))
    except ValueError:
        return v


def grade_answer(*, expected: str, actual: str, answer_schema: str | None) -> dict[str, Any]:
    expected_stripped = (expected or "").strip()
    actual_stripped = (actual or "").strip()
    exact_match = actual_stripped == expected_stripped

    keys = schema_keys(answer_schema)
    expected_assignments = parse_assignments(expected_stripped)
    actual_assignments = parse_assignments(actual_stripped)

    key_matches: dict[str, bool] = {}
    for key in keys:
        if key not in expected_assignments:
            continue
        key_matches[key] = (
            normalize_value(actual_assignments.get(key))
            == normalize_value(expected_assignments[key])
        )

    if key_matches:
        correct = all(key_matches.values())
        match_kind = "schema_key_values"
    else:
        # Also accept bare values: model returned "foo" instead of "key=foo".
        # For single-key schemas, check if the whole actual string matches the expected value.
        expected_values = list(expected_assignments.values())
        actual_norm = normalize_value(actual_stripped)
        if (
            len(expected_values) == 1
            and actual_norm == normalize_value(expected_values[0])
        ):
            correct = True
            match_kind = "bare_value"
        else:
            correct = exact_match
            match_kind = "exact"

    return {
        "correct": correct,
        "match_kind": match_kind,
        "exact_match": exact_match,
        "schema_keys": keys,
        "expected_assignments": expected_assignments,
        "actual_assignments": actual_assignments,
        "key_matches": key_matches,
    }


# ---------------------------------------------------------------------------
# Provider: OpenAI (Responses API)
# ---------------------------------------------------------------------------

def _default_reasoning_effort(model: str) -> str | None:
    m = model.lower()
    if m in {"gpt-5.1", "gpt-5.2", "gpt-5.4"} or m.startswith(("gpt-5.1-", "gpt-5.2-", "gpt-5.4-")):
        return "none"
    if m.startswith("o3") or m.startswith("o4"):
        return "low"
    # gpt-5.5+ dropped "minimal"; use "low" as the cheapest valid option
    if m in {"gpt-5.5"} or m.startswith("gpt-5.5-"):
        return "low"
    if m.startswith("gpt-5") or re.match(r"^o[1-9]", m):
        return "minimal"
    return None


def _build_ssl_context(ca_bundle: Path | None, insecure: bool) -> ssl.SSLContext | None:
    if insecure:
        return ssl._create_unverified_context()
    if ca_bundle:
        return ssl.create_default_context(cafile=str(ca_bundle))
    try:
        import certifi
        p = certifi.where()
        if Path(p).is_file():
            return ssl.create_default_context(cafile=p)
    except ImportError:
        pass
    return None


def call_openai(
    *,
    api_key: str,
    api_base: str,
    model: str,
    prompt: str,
    max_output_tokens: int,
    reasoning_effort: str | None,
    temperature: float | None,
    timeout_sec: float,
    retries: int,
    ca_bundle: Path | None,
    insecure: bool,
) -> tuple[str, dict[str, Any]]:
    url = api_base.rstrip("/") + "/responses"
    payload: dict[str, Any] = {
        "model": model,
        "instructions": SYSTEM_PROMPT,
        "input": prompt,
        "max_output_tokens": max_output_tokens,
        "store": False,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if reasoning_effort is not None:
        payload["reasoning"] = {"effort": reasoning_effort}

    body = json.dumps(payload).encode()
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    ssl_ctx = _build_ssl_context(ca_bundle, insecure)

    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout_sec, context=ssl_ctx) as resp:
                raw = json.loads(resp.read().decode())
                # extract text from Responses API shape
                text = raw.get("output_text") or ""
                if not text:
                    for item in raw.get("output", []) or []:
                        for chunk in (item.get("content") or []):
                            text += chunk.get("text", "")
                return text.strip(), raw.get("usage", {})
        except urllib.error.HTTPError as exc:
            if exc.code not in {408, 409, 429, 500, 502, 503, 504} or attempt >= retries:
                raise RuntimeError(f"OpenAI HTTP {exc.code}: {exc.read().decode(errors='replace')}") from exc
            time.sleep(min(30.0, 2.0 ** attempt + random.random()))
        except urllib.error.URLError as exc:
            if attempt >= retries:
                raise RuntimeError(f"OpenAI request failed: {exc}") from exc
            time.sleep(min(30.0, 2.0 ** attempt + random.random()))

    raise RuntimeError("OpenAI request failed after retries")


# ---------------------------------------------------------------------------
# Provider: OpenAI-compatible Chat Completions (Baseten serverless/deployed)
# ---------------------------------------------------------------------------

def call_chat_completions(
    *,
    api_key: str,
    api_base: str,
    model: str,
    prompt: str,
    max_output_tokens: int,
    temperature: float | None,
    timeout_sec: float,
    retries: int,
    ca_bundle: Path | None,
    insecure: bool,
) -> tuple[str, dict[str, Any]]:
    url = api_base.rstrip("/") + "/chat/completions"
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_output_tokens,
    }
    if temperature is not None:
        payload["temperature"] = temperature

    body = json.dumps(payload).encode()
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    ssl_ctx = _build_ssl_context(ca_bundle, insecure)

    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout_sec, context=ssl_ctx) as resp:
                raw = json.loads(resp.read().decode())
                text = (raw.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
                return text.strip(), raw.get("usage", {})
        except urllib.error.HTTPError as exc:
            if exc.code not in {408, 409, 429, 500, 502, 503, 504} or attempt >= retries:
                raise RuntimeError(f"Chat HTTP {exc.code}: {exc.read().decode(errors='replace')}") from exc
            time.sleep(min(30.0, 2.0 ** attempt + random.random()))
        except urllib.error.URLError as exc:
            if attempt >= retries:
                raise RuntimeError(f"Chat request failed: {exc}") from exc
            time.sleep(min(30.0, 2.0 ** attempt + random.random()))

    raise RuntimeError("Chat completions request failed after retries")


# ---------------------------------------------------------------------------
# Provider: Anthropic (Messages API)
# ---------------------------------------------------------------------------

def call_anthropic(
    *,
    client: Any,  # anthropic.Anthropic
    model: str,
    prompt: str,
    max_output_tokens: int,
    thinking_effort: str | None,
    timeout_sec: float,
    retries: int,
) -> tuple[str, dict[str, Any]]:
    import anthropic

    supports_adaptive_thinking = "opus-4-6" in model or "sonnet-4-6" in model

    for attempt in range(retries + 1):
        try:
            params: dict[str, Any] = {
                "model": model,
                "max_tokens": max_output_tokens,
                "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": prompt}],
                "timeout": timeout_sec,
            }
            if thinking_effort is not None and supports_adaptive_thinking:
                params["thinking"] = {"type": "adaptive"}
                params["output_config"] = {"effort": thinking_effort}
            msg = client.messages.create(**params)
            # Extract text — skip thinking blocks, grab first text block
            text = ""
            for block in (msg.content or []):
                if getattr(block, "type", None) == "text":
                    text = block.text
                    break
            usage = {"input_tokens": msg.usage.input_tokens, "output_tokens": msg.usage.output_tokens}
            return text.strip(), usage
        except anthropic.RateLimitError:
            if attempt >= retries:
                raise
            time.sleep(min(30.0, 2.0 ** attempt))
        except anthropic.APIStatusError as exc:
            if exc.status_code not in {429, 500, 502, 503, 529} or attempt >= retries:
                raise
            time.sleep(min(30.0, 2.0 ** attempt))

    raise RuntimeError("Anthropic request failed after retries")


# ---------------------------------------------------------------------------
# Per-probe evaluation (provider-agnostic)
# ---------------------------------------------------------------------------

def evaluate_probe_file(
    *,
    probe_file: Path,
    args: argparse.Namespace,
    openai_key: str,
    anthropic_client: Any,
) -> dict[str, Any]:
    probe = load_json(probe_file)
    prompt = build_prompt(probe, max_context_chars=args.max_context_chars or None)
    started = time.time()

    result: dict[str, Any] = {
        "probe_file": str(probe_file),
        "model": args.model,
        "provider": args.provider,
        "question": probe.get("question"),
        "expected_answer": probe.get("answer"),
        "metadata": probe.get("metadata", {}),
        "prompt_chars": len(prompt),
    }

    if args.dry_run:
        result.update({"status": "dry_run", "model_answer": None, "correct": None,
                        "elapsed_sec": round(time.time() - started, 3)})
        return result

    if args.provider == "openai":
        model_answer, usage = call_openai(
            api_key=openai_key,
            api_base=args.api_base,
            model=args.model,
            prompt=prompt,
            max_output_tokens=args.max_output_tokens,
            reasoning_effort=args.reasoning_effort,
            temperature=args.temperature,
            timeout_sec=args.timeout_sec,
            retries=args.retries,
            ca_bundle=args.ca_bundle,
            insecure=args.insecure_skip_tls_verify,
        )
    elif args.provider == "baseten":
        model_answer, usage = call_chat_completions(
            api_key=openai_key,
            api_base=args.api_base,
            model=args.model,
            prompt=prompt,
            max_output_tokens=args.max_output_tokens,
            temperature=args.temperature,
            timeout_sec=args.timeout_sec,
            retries=args.retries,
            ca_bundle=args.ca_bundle,
            insecure=args.insecure_skip_tls_verify,
        )
    else:  # anthropic
        model_answer, usage = call_anthropic(
            client=anthropic_client,
            model=args.model,
            prompt=prompt,
            max_output_tokens=args.max_output_tokens,
            thinking_effort=getattr(args, "thinking_effort", None),
            timeout_sec=args.timeout_sec,
            retries=args.retries,
        )

    grade = grade_answer(
        expected=str(probe.get("answer", "")),
        actual=model_answer,
        answer_schema=(probe.get("metadata") or {}).get("answer_schema"),
    )
    result.update({"status": "ok", "model_answer": model_answer, **grade,
                   "usage": usage, "elapsed_sec": round(time.time() - started, 3)})
    return result


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def summarize(results: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    ok = [r for r in results if r.get("status") == "ok"]
    errors = [r for r in results if r.get("status") == "error"]
    correct = [r for r in ok if r.get("correct") is True]

    def breakdown(key_fn):
        d: dict[str, dict[str, int]] = {}
        for item in ok:
            k = key_fn(item) or "unknown"
            b = d.setdefault(k, {"total": 0, "correct": 0})
            b["total"] += 1
            if item.get("correct") is True:
                b["correct"] += 1
        for b in d.values():
            b["accuracy"] = round(b["correct"] / b["total"], 4) if b["total"] else 0
        return dict(sorted(d.items()))

    return {
        "model": args.model,
        "provider": args.provider,
        "probe_dir": str(args.probe_dir),
        "out_dir": str(args.out_dir),
        "total": len(results),
        "completed": len(ok),
        "errors": len(errors),
        "correct": len(correct),
        "accuracy": round(len(correct) / len(ok), 4) if ok else None,
        "by_probe_type": breakdown(lambda r: (r.get("metadata") or {}).get("probe_type")),
        "by_failure_mode": breakdown(lambda r: (r.get("metadata") or {}).get("failure_mode_target")),
        "by_injector": breakdown(lambda r: (r.get("metadata") or {}).get("injector")),
        "config": {
            "max_context_chars": args.max_context_chars,
            "max_output_tokens": args.max_output_tokens,
            "reasoning_effort": getattr(args, "reasoning_effort", None),
            "thinking_effort": getattr(args, "thinking_effort", None),
            "temperature": getattr(args, "temperature", None),
            "limit": args.limit,
            "concurrent_n": args.concurrent_n,
        },
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def evaluate_one_indexed(
    *, index: int, total: int, probe_file: Path,
    args: argparse.Namespace, openai_key: str, anthropic_client: Any,
) -> tuple[int, dict[str, Any], str]:
    out_path = result_path_for_probe(args.out_dir, probe_file)
    if out_path.exists() and args.resume:
        result = load_json(out_path)
        return index, result, "skipped"

    try:
        result = evaluate_probe_file(
            probe_file=probe_file, args=args,
            openai_key=openai_key, anthropic_client=anthropic_client,
        )
    except Exception as exc:
        result = {"probe_file": str(probe_file), "model": args.model,
                  "status": "error", "error": f"{type(exc).__name__}: {exc}"}
        if args.fail_fast:
            write_json(out_path, result)
            raise

    result["input_index"] = index
    result["input_total"] = total
    write_json(out_path, result)
    return index, result, str(result.get("status"))


def log_result(index: int, total: int, result: dict[str, Any], label: str) -> None:
    status = "skipped" if label == "skipped" else result.get("status")
    correct = result.get("correct")
    note = " empty_answer" if status == "ok" and not str(result.get("model_answer") or "").strip() else ""
    print(f"[{index}/{total}] {status} correct={correct}{note} {Path(str(result.get('probe_file',''))).name}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate any LM provider on extracted runtime probes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Provider + model (mirrors halluworld run_benchmark.py)
    parser.add_argument("--provider", required=True, choices=["openai", "anthropic", "baseten"],
                        help="LM provider (baseten uses OpenAI-compatible chat/completions)")
    parser.add_argument("--model", required=True, help="Model name")

    # I/O
    parser.add_argument("--probe-dir", type=Path, required=True,
                        help="Directory of probe JSON files (from extract_run_probes.py)")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Directory to write result JSON files and summary.json")

    # OpenAI-specific
    parser.add_argument("--api-base", default=DEFAULT_OPENAI_API_BASE,
                        help="OpenAI API base URL (default: api.openai.com/v1)")
    parser.add_argument("--reasoning-effort",
                        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
                        default=None,
                        help="Reasoning effort for GPT-5/o-series (auto-detected if omitted)")

    # Anthropic-specific
    parser.add_argument("--thinking-effort",
                        choices=["low", "medium", "high", "max"],
                        default=None,
                        help="Adaptive thinking effort for Claude 4.6 models (low/medium/high/max)")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Sampling temperature (omit for reasoning models)")
    parser.add_argument("--ca-bundle", type=Path, default=None)
    parser.add_argument("--insecure-skip-tls-verify", action="store_true")

    # Token budget
    parser.add_argument("--max-output-tokens", type=int, default=None,
                        help="Max output tokens per probe (default: 16384 for reasoning/thinking models, 256 otherwise)")
    parser.add_argument("--max-context-chars", type=int, default=0,
                        help="Truncate probe context to this many chars (0 = no limit)")

    # Run control
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only first N probes")
    parser.add_argument("-n", "--concurrent-n", type=int, default=1,
                        help="Concurrent probe evaluations (default: 1)")
    parser.add_argument("--delay-sec", type=float, default=0.0,
                        help="Delay between probe submissions (sequential mode only)")
    parser.add_argument("--timeout-sec", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--resume", action="store_true",
                        help="Skip probes with existing result files")
    parser.add_argument("--dry-run", action="store_true",
                        help="Parse probes without calling the API")
    parser.add_argument("--fail-fast", action="store_true",
                        help="Stop on first error")

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Validate
    if not args.probe_dir.is_dir():
        print(f"error: probe directory does not exist: {args.probe_dir}", file=sys.stderr)
        return 2
    if args.concurrent_n < 1:
        print("error: --concurrent-n must be >= 1", file=sys.stderr)
        return 2

    # Set up provider clients
    openai_key = ""
    anthropic_client = None

    if args.provider == "openai":
        openai_key = os.environ.get("OPENAI_API_KEY", "")
        if not openai_key and not args.dry_run:
            print("error: OPENAI_API_KEY is not set", file=sys.stderr)
            return 2
        if args.reasoning_effort is None:
            args.reasoning_effort = _default_reasoning_effort(args.model)
        # Auto-scale token budget for reasoning models
        if args.max_output_tokens is None:
            if args.reasoning_effort and args.reasoning_effort != "none":
                args.max_output_tokens = 16384
            else:
                args.max_output_tokens = DEFAULT_MAX_OUTPUT_TOKENS
    elif args.provider == "baseten":
        openai_key = os.environ.get("BASETEN_API_KEY", "")
        if not openai_key and not args.dry_run:
            print("error: BASETEN_API_KEY is not set", file=sys.stderr)
            return 2
    else:  # anthropic
        import anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key and not args.dry_run:
            print("error: ANTHROPIC_API_KEY is not set", file=sys.stderr)
            return 2
        anthropic_client = anthropic.Anthropic(api_key=api_key)
        # Auto-scale token budget for thinking models
        if args.max_output_tokens is None:
            if getattr(args, "thinking_effort", None):
                args.max_output_tokens = 16384
            else:
                args.max_output_tokens = DEFAULT_MAX_OUTPUT_TOKENS

    if args.max_output_tokens is None:
        args.max_output_tokens = DEFAULT_MAX_OUTPUT_TOKENS

    # Print header (mirrors halluworld run_benchmark.py)
    print(f"\n{'='*60}")
    print(f"HalluWorld Terminal — Probe Evaluator")
    print(f"{'='*60}")
    print(f"Provider : {args.provider}")
    print(f"Model    : {args.model}")
    if args.provider == "openai" and args.reasoning_effort:
        print(f"Reasoning: {args.reasoning_effort}")
    if args.provider == "anthropic" and getattr(args, "thinking_effort", None):
        print(f"Thinking : {args.thinking_effort}")
    print(f"Probe dir: {args.probe_dir}")
    print(f"Out dir  : {args.out_dir}")
    print(f"{'='*60}\n")

    # Load probes
    probe_files = find_probe_files(args.probe_dir)
    if args.limit is not None:
        probe_files = probe_files[: args.limit]
    total = len(probe_files)
    print(f"Evaluating {total} probes with -n {args.concurrent_n}...\n")

    # Prepare output dir
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        for p in args.out_dir.glob("*.result.json"):
            p.unlink()
        summary_path = args.out_dir / "summary.json"
        if summary_path.exists():
            summary_path.unlink()

    # Run
    indexed = list(enumerate(probe_files, start=1))
    results_by_index: dict[int, dict[str, Any]] = {}

    run_kwargs = dict(args=args, openai_key=openai_key, anthropic_client=anthropic_client)

    if args.concurrent_n == 1:
        for index, probe_file in indexed:
            _, result, label = evaluate_one_indexed(
                index=index, total=total, probe_file=probe_file, **run_kwargs
            )
            results_by_index[index] = result
            log_result(index, total, result, label)
            if args.delay_sec > 0 and index < total:
                time.sleep(args.delay_sec)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrent_n) as pool:
            futures = {
                pool.submit(evaluate_one_indexed,
                            index=idx, total=total, probe_file=pf, **run_kwargs): idx
                for idx, pf in indexed
            }
            for fut in concurrent.futures.as_completed(futures):
                idx, result, label = fut.result()
                results_by_index[idx] = result
                log_result(idx, total, result, label)

    results = [results_by_index[i] for i, _ in indexed]
    summary = summarize(results, args)
    write_json(args.out_dir / "summary.json", summary)

    # Print summary (mirrors halluworld)
    print(f"\n{'='*60}")
    print(f"RESULTS — {args.provider} / {args.model}")
    print(f"{'='*60}")
    print(f"completed={summary['completed']}  errors={summary['errors']}  "
          f"correct={summary['correct']}  accuracy={summary['accuracy']}")
    print("\nBy probe type:")
    for pt, b in summary["by_probe_type"].items():
        bar = "#" * round(b["accuracy"] * 20)
        print(f"  {pt:<22} {b['correct']:>3}/{b['total']:<3} acc={b['accuracy']:.3f} |{bar:<20}|")
    print(f"\nSaved: {args.out_dir}/summary.json")

    return 1 if summary["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
