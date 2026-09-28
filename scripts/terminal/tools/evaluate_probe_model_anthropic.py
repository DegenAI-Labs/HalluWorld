#!/usr/bin/env python3
"""
Evaluate an Anthropic (Claude) model on extracted runtime probes.

Mirrors the interface of tools/evaluate_probe_model.py so the same
audit/sweep tools work on the output.  Requires the `anthropic` package
and ANTHROPIC_API_KEY to be set.

Examples:
  python3 tools/evaluate_probe_model_anthropic.py \
    --probe-dir extracted_llm_probes_20260504 \
    --out-dir runs/probe_eval_claude_sonnet \
    --model claude-sonnet-4-6 \
    -n 8

  python3 tools/evaluate_probe_model_anthropic.py \
    --probe-dir extracted_llm_probes_20260504 \
    --out-dir runs/probe_eval_claude_opus \
    --model claude-opus-4-6 \
    --limit 50
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import anthropic

DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_MAX_OUTPUT_TOKENS = 256
SYSTEM_PROMPT = (
    "Answer probe questions with only the exact requested answer string. "
    "Do not include prose, markdown, or explanation."
)


# ---------------------------------------------------------------------------
# Shared helpers (mirrors evaluate_probe_model.py)
# ---------------------------------------------------------------------------

ASSIGNMENT_RE = re.compile(
    r"\b([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:<\s*)?([A-Za-z0-9_./:+-]+)(?:\s*>)?"
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def find_probe_files(probe_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in probe_dir.glob("*.json")
        if path.name != "summary.json" and path.is_file()
    )


def result_path_for_probe(out_dir: Path, probe_file: Path) -> Path:
    return out_dir / f"{probe_file.stem}.result.json"


def parse_assignments(text: str) -> dict[str, str]:
    return {
        key: value.strip("`'\".,;")
        for key, value in ASSIGNMENT_RE.findall(text or "")
    }


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


def grade_answer(
    *,
    expected: str,
    actual: str,
    answer_schema: str | None,
) -> dict[str, Any]:
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
        key_matches[key] = normalize_value(actual_assignments.get(key)) == normalize_value(
            expected_assignments[key]
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


def build_prompt(probe: dict[str, Any], max_context_chars: int | None) -> str:
    context = probe.get("context") or ""
    if max_context_chars and max_context_chars > 0 and len(context) > max_context_chars:
        omitted = len(context) - max_context_chars
        context = (
            f"[Context truncated: omitted {omitted} leading characters.]\n"
            + context[-max_context_chars:]
        )
    return (
        "You are answering a Terminal-Bench runtime probe.\n"
        "Use only the terminal context below. The context is captured immediately "
        "before the current command executes and contains only the terminal pane "
        "visible at that time. If the question asks about previous commands, answer "
        "from the commands visible in this context only.\n"
        "Return only the requested answer string in the exact requested schema. Do not explain.\n\n"
        "TERMINAL CONTEXT:\n"
        "<<<CONTEXT\n"
        f"{context}\n"
        "CONTEXT>>>\n\n"
        f"QUESTION:\n{probe.get('question')}\n"
    )


# ---------------------------------------------------------------------------
# Anthropic API call
# ---------------------------------------------------------------------------

def call_anthropic(
    *,
    client: anthropic.Anthropic,
    model: str,
    prompt: str,
    max_output_tokens: int,
    timeout_sec: float,
    retries: int,
) -> tuple[str, dict[str, Any]]:
    """Returns (answer_text, usage_dict)."""
    for attempt in range(retries + 1):
        try:
            message = client.messages.create(
                model=model,
                max_tokens=max_output_tokens,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
                timeout=timeout_sec,
            )
            answer = message.content[0].text if message.content else ""
            usage = {
                "input_tokens": message.usage.input_tokens,
                "output_tokens": message.usage.output_tokens,
            }
            return answer.strip(), usage
        except anthropic.RateLimitError:
            if attempt >= retries:
                raise
            time.sleep(min(30.0, 2.0 ** attempt))
        except anthropic.APIStatusError as exc:
            if exc.status_code not in {429, 500, 502, 503, 529} or attempt >= retries:
                raise
            time.sleep(min(30.0, 2.0 ** attempt))

    raise RuntimeError("Anthropic API failed after retries")


# ---------------------------------------------------------------------------
# Per-probe evaluation
# ---------------------------------------------------------------------------

def evaluate_probe_file(
    *,
    probe_file: Path,
    out_dir: Path,
    client: anthropic.Anthropic,
    model: str,
    max_context_chars: int | None,
    max_output_tokens: int,
    timeout_sec: float,
    retries: int,
    save_raw_response: bool,
    dry_run: bool,
) -> dict[str, Any]:
    probe = load_json(probe_file)
    prompt = build_prompt(probe, max_context_chars=max_context_chars)
    started = time.time()

    result: dict[str, Any] = {
        "probe_file": str(probe_file),
        "model": model,
        "question": probe.get("question"),
        "expected_answer": probe.get("answer"),
        "metadata": probe.get("metadata", {}),
        "prompt_chars": len(prompt),
    }

    if dry_run:
        result.update(
            {
                "status": "dry_run",
                "model_answer": None,
                "correct": None,
                "elapsed_sec": round(time.time() - started, 3),
            }
        )
        return result

    model_answer, usage = call_anthropic(
        client=client,
        model=model,
        prompt=prompt,
        max_output_tokens=max_output_tokens,
        timeout_sec=timeout_sec,
        retries=retries,
    )
    grade = grade_answer(
        expected=str(probe.get("answer", "")),
        actual=model_answer,
        answer_schema=(probe.get("metadata") or {}).get("answer_schema"),
    )

    result.update(
        {
            "status": "ok",
            "model_answer": model_answer,
            **grade,
            "usage": usage,
            "elapsed_sec": round(time.time() - started, 3),
        }
    )
    return result


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def summarize(results: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    ok = [r for r in results if r.get("status") == "ok"]
    errors = [r for r in results if r.get("status") == "error"]
    correct = [r for r in ok if r.get("correct") is True]

    by_injector: dict[str, dict[str, int]] = {}
    for item in ok:
        injector = ((item.get("metadata") or {}).get("injector")) or "unknown"
        bucket = by_injector.setdefault(injector, {"total": 0, "correct": 0})
        bucket["total"] += 1
        if item.get("correct") is True:
            bucket["correct"] += 1

    for bucket in by_injector.values():
        total = bucket["total"]
        bucket["accuracy"] = round(bucket["correct"] / total, 6) if total else 0

    by_probe_type: dict[str, dict[str, int]] = {}
    for item in ok:
        pt = ((item.get("metadata") or {}).get("probe_type")) or "unknown"
        bucket = by_probe_type.setdefault(pt, {"total": 0, "correct": 0})
        bucket["total"] += 1
        if item.get("correct") is True:
            bucket["correct"] += 1
    for bucket in by_probe_type.values():
        total = bucket["total"]
        bucket["accuracy"] = round(bucket["correct"] / total, 6) if total else 0

    return {
        "model": args.model,
        "probe_dir": str(args.probe_dir),
        "out_dir": str(args.out_dir),
        "total": len(results),
        "completed": len(ok),
        "errors": len(errors),
        "correct": len(correct),
        "accuracy": round(len(correct) / len(ok), 6) if ok else None,
        "by_injector": dict(sorted(by_injector.items())),
        "by_probe_type": dict(sorted(by_probe_type.items())),
        "config": {
            "max_context_chars": args.max_context_chars,
            "max_output_tokens": args.max_output_tokens,
            "limit": args.limit,
            "delay_sec": args.delay_sec,
            "concurrent_n": args.concurrent_n,
        },
    }


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a Claude model on extracted probe JSON files."
    )
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-context-chars", type=int, default=0)
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--timeout-sec", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--delay-sec", type=float, default=0.0)
    parser.add_argument("-n", "--concurrent-n", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-raw-response", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def evaluate_one_indexed(
    *,
    index: int,
    total: int,
    probe_file: Path,
    args: argparse.Namespace,
    client: anthropic.Anthropic,
) -> tuple[int, dict[str, Any], str]:
    out_path = result_path_for_probe(args.out_dir, probe_file)
    if out_path.exists() and args.resume:
        result = load_json(out_path)
        result.setdefault("input_index", index)
        result.setdefault("input_total", total)
        return index, result, "skipped"

    try:
        result = evaluate_probe_file(
            probe_file=probe_file,
            out_dir=args.out_dir,
            client=client,
            model=args.model,
            max_context_chars=args.max_context_chars,
            max_output_tokens=args.max_output_tokens,
            timeout_sec=args.timeout_sec,
            retries=args.retries,
            save_raw_response=args.save_raw_response,
            dry_run=args.dry_run,
        )
    except Exception as exc:
        result = {
            "probe_file": str(probe_file),
            "model": args.model,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }
        if args.fail_fast:
            write_json(out_path, result)
            raise

    result["input_index"] = index
    result["input_total"] = total
    write_json(out_path, result)
    return index, result, str(result.get("status"))


def log_result(index: int, total: int, result: dict[str, Any], label: str) -> None:
    probe_file = Path(str(result.get("probe_file", "")))
    status = "skipped" if label == "skipped" else result.get("status")
    correct = result.get("correct")
    warning = ""
    if status == "ok" and not str(result.get("model_answer") or "").strip():
        warning = " empty_answer"
    print(f"[{index}/{total}] {status} correct={correct}{warning} {probe_file.name}")


def main() -> int:
    args = parse_args()
    if args.concurrent_n < 1:
        print("error: --concurrent-n must be >= 1", file=sys.stderr)
        return 2
    if not args.probe_dir.is_dir():
        print(f"error: probe directory does not exist: {args.probe_dir}", file=sys.stderr)
        return 2

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key and not args.dry_run:
        print("error: ANTHROPIC_API_KEY is not set", file=sys.stderr)
        return 2

    client = anthropic.Anthropic(api_key=api_key)

    probe_files = find_probe_files(args.probe_dir)
    if args.limit is not None:
        probe_files = probe_files[: args.limit]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        for path in args.out_dir.glob("*.result.json"):
            path.unlink()
        summary_path = args.out_dir / "summary.json"
        if summary_path.exists():
            summary_path.unlink()

    indexed_probe_files = list(enumerate(probe_files, start=1))
    results_by_index: dict[int, dict[str, Any]] = {}
    total = len(indexed_probe_files)

    if args.concurrent_n == 1:
        for index, probe_file in indexed_probe_files:
            _, result, label = evaluate_one_indexed(
                index=index,
                total=total,
                probe_file=probe_file,
                args=args,
                client=client,
            )
            results_by_index[index] = result
            log_result(index, total, result, label)
            if args.delay_sec > 0 and index < total:
                time.sleep(args.delay_sec)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrent_n) as executor:
            futures: dict[concurrent.futures.Future[tuple[int, dict[str, Any], str]], int] = {}
            for index, probe_file in indexed_probe_files:
                future = executor.submit(
                    evaluate_one_indexed,
                    index=index,
                    total=total,
                    probe_file=probe_file,
                    args=args,
                    client=client,
                )
                futures[future] = index
                if args.delay_sec > 0 and index < total:
                    time.sleep(args.delay_sec)

            for future in concurrent.futures.as_completed(futures):
                index, result, label = future.result()
                results_by_index[index] = result
                log_result(index, total, result, label)

    results = [results_by_index[index] for index, _ in indexed_probe_files]
    summary = summarize(results, args)
    write_json(args.out_dir / "summary.json", summary)
    print(
        f"completed={summary['completed']} errors={summary['errors']} "
        f"correct={summary['correct']} accuracy={summary['accuracy']}"
    )
    return 1 if summary["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
