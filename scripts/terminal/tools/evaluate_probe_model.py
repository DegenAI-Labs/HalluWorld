#!/usr/bin/env python3
"""
Evaluate an OpenAI model on extracted runtime probes.

Input files should be produced by tools/extract_run_probes.py. The evaluator
sends each probe's full context and question to a model, records the model
answer, and grades it against the golden answer.

Examples:
  python3 tools/evaluate_probe_model.py \
    --probe-dir runs/all-tasks__20260502_201427/extracted_probes \
    --out-dir runs/all-tasks__20260502_201427/gpt-4o-mini_probe_eval \
    --model gpt-4o-mini \
    -n 8

  python3 tools/evaluate_probe_model.py \
    --probe-dir runs/all-tasks__20260502_201427/extracted_probes \
    --out-dir /tmp/probe_eval_smoke \
    --model gpt-4o-mini \
    --limit 10
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


DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_API_BASE = "https://api.openai.com/v1"
DEFAULT_MAX_OUTPUT_TOKENS = 256
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


def extract_output_text(response: dict[str, Any]) -> str:
    output_text = response.get("output_text")
    if isinstance(output_text, str):
        return output_text

    chunks: list[str] = []
    for item in response.get("output", []) or []:
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []) or []:
            if not isinstance(content, dict):
                continue
            text = content.get("text")
            if isinstance(text, str):
                chunks.append(text)
    return "\n".join(chunks).strip()


def certifi_bundle_path() -> str | None:
    try:
        import certifi  # type: ignore[import-not-found]
    except ImportError:
        return None

    path = certifi.where()
    return path if Path(path).is_file() else None


def build_ssl_context(
    *,
    ca_bundle: Path | None,
    insecure_skip_tls_verify: bool,
) -> ssl.SSLContext | None:
    if insecure_skip_tls_verify:
        return ssl._create_unverified_context()

    if ca_bundle is not None:
        return ssl.create_default_context(cafile=str(ca_bundle))

    certifi_path = certifi_bundle_path()
    if certifi_path:
        return ssl.create_default_context(cafile=certifi_path)

    return None


def default_reasoning_effort(model: str) -> str | None:
    normalized = model.lower()
    if (
        normalized == "gpt-5.1"
        or normalized.startswith("gpt-5.1-")
        or normalized == "gpt-5.2"
        or normalized.startswith("gpt-5.2-")
        or normalized == "gpt-5.4"
        or normalized.startswith("gpt-5.4-")
    ):
        return "none"
    if normalized.startswith("o3") or normalized.startswith("o4"):
        return "low"
    if normalized.startswith("gpt-5") or re.match(r"^o[1-9]", normalized):
        return "minimal"
    return None


def call_openai_responses(
    *,
    api_key: str,
    api_base: str,
    model: str,
    prompt: str,
    max_output_tokens: int,
    temperature: float | None,
    reasoning_effort: str | None,
    timeout_sec: float,
    retries: int,
    ca_bundle: Path | None,
    insecure_skip_tls_verify: bool,
) -> dict[str, Any]:
    url = api_base.rstrip("/") + "/responses"
    payload: dict[str, Any] = {
        "model": model,
        "instructions": (
            "Answer probe questions with only the exact requested answer string. "
            "Do not include prose, markdown, or explanation."
        ),
        "input": prompt,
        "max_output_tokens": max_output_tokens,
        "store": False,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if reasoning_effort is not None:
        payload["reasoning"] = {"effort": reasoning_effort}
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    ssl_context = build_ssl_context(
        ca_bundle=ca_bundle,
        insecure_skip_tls_verify=insecure_skip_tls_verify,
    )

    for attempt in range(retries + 1):
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(
                request,
                timeout=timeout_sec,
                context=ssl_context,
            ) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            if exc.code not in {408, 409, 429, 500, 502, 503, 504} or attempt >= retries:
                raise RuntimeError(f"OpenAI API HTTP {exc.code}: {error_body}") from exc
            time.sleep(min(30.0, 2.0**attempt + random.random()))
        except urllib.error.URLError as exc:
            if attempt >= retries:
                raise RuntimeError(f"OpenAI API request failed: {exc}") from exc
            time.sleep(min(30.0, 2.0**attempt + random.random()))

    raise RuntimeError("OpenAI API request failed after retries")


def result_path_for_probe(out_dir: Path, probe_file: Path) -> Path:
    return out_dir / f"{probe_file.stem}.result.json"


def evaluate_probe_file(
    *,
    probe_file: Path,
    out_dir: Path,
    api_key: str,
    api_base: str,
    model: str,
    max_context_chars: int | None,
    max_output_tokens: int,
    temperature: float | None,
    reasoning_effort: str | None,
    timeout_sec: float,
    retries: int,
    ca_bundle: Path | None,
    insecure_skip_tls_verify: bool,
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

    response = call_openai_responses(
        api_key=api_key,
        api_base=api_base,
        model=model,
        prompt=prompt,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
        timeout_sec=timeout_sec,
        retries=retries,
        ca_bundle=ca_bundle,
        insecure_skip_tls_verify=insecure_skip_tls_verify,
    )
    model_answer = extract_output_text(response)
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
            "response_id": response.get("id"),
            "usage": response.get("usage"),
            "elapsed_sec": round(time.time() - started, 3),
        }
    )
    if save_raw_response:
        result["raw_response"] = response
    return result


def summarize(results: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    ok = [item for item in results if item.get("status") == "ok"]
    errors = [item for item in results if item.get("status") == "error"]
    correct = [item for item in ok if item.get("correct") is True]

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
        "config": {
            "max_context_chars": args.max_context_chars,
            "max_output_tokens": args.max_output_tokens,
            "temperature": args.temperature,
            "reasoning_effort": args.reasoning_effort,
            "limit": args.limit,
            "delay_sec": args.delay_sec,
            "concurrent_n": args.concurrent_n,
            "ca_bundle": str(args.ca_bundle) if args.ca_bundle else None,
            "insecure_skip_tls_verify": args.insecure_skip_tls_verify,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an OpenAI model on extracted probe JSON files."
    )
    parser.add_argument(
        "--probe-dir",
        type=Path,
        required=True,
        help="Directory produced by tools/extract_run_probes.py.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory where result JSON files and summary.json will be written.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=0,
        help="If >0, keep only this many trailing context characters.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=DEFAULT_MAX_OUTPUT_TOKENS,
        help=(
            "Maximum output tokens. Reasoning models may spend some of this "
            "budget on hidden reasoning before emitting the short final answer."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature. Omitted by default because some models, including GPT-5, do not support it.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
        default=None,
        help=(
            "Responses API reasoning effort for GPT-5/o-series models. "
            "If omitted, uses a low-effort default for known reasoning models."
        ),
    )
    parser.add_argument("--timeout-sec", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--ca-bundle",
        type=Path,
        default=None,
        help="PEM CA bundle to use for TLS verification. Defaults to certifi if installed.",
    )
    parser.add_argument(
        "--insecure-skip-tls-verify",
        action="store_true",
        help="Disable TLS certificate verification. Use only for local debugging.",
    )
    parser.add_argument("--delay-sec", type=float, default=0.0)
    parser.add_argument(
        "-n",
        "--concurrent-n",
        type=int,
        default=1,
        help="Number of probes to evaluate concurrently.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip existing result files. By default, existing result files are overwritten.",
    )
    parser.add_argument(
        "--save-raw-response",
        action="store_true",
        help="Store the full OpenAI response JSON inside each result file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build result files without calling the API.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop at the first probe evaluation error.",
    )
    return parser.parse_args()


def evaluate_one_indexed(
    *,
    index: int,
    total: int,
    probe_file: Path,
    args: argparse.Namespace,
    api_key: str,
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
            api_key=api_key,
            api_base=args.api_base,
            model=args.model,
            max_context_chars=args.max_context_chars,
            max_output_tokens=args.max_output_tokens,
            temperature=args.temperature,
            reasoning_effort=args.reasoning_effort,
            timeout_sec=args.timeout_sec,
            retries=args.retries,
            ca_bundle=args.ca_bundle,
            insecure_skip_tls_verify=args.insecure_skip_tls_verify,
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
        usage = result.get("usage") or {}
        output_details = usage.get("output_tokens_details") or {}
        reasoning_tokens = output_details.get("reasoning_tokens")
        warning = (
            f" empty_answer reasoning_tokens={reasoning_tokens}"
            if reasoning_tokens is not None
            else " empty_answer"
        )
    print(f"[{index}/{total}] {status} correct={correct}{warning} {probe_file.name}")


def main() -> int:
    args = parse_args()
    if args.reasoning_effort is None:
        args.reasoning_effort = default_reasoning_effort(args.model)
    if args.concurrent_n < 1:
        print("error: --concurrent-n must be >= 1", file=sys.stderr)
        return 2
    if args.ca_bundle is not None and not args.ca_bundle.is_file():
        print(f"error: --ca-bundle does not exist: {args.ca_bundle}", file=sys.stderr)
        return 2
    if not args.probe_dir.is_dir():
        print(f"error: probe directory does not exist: {args.probe_dir}", file=sys.stderr)
        return 2

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key and not args.dry_run:
        print("error: OPENAI_API_KEY is not set", file=sys.stderr)
        return 2

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
                api_key=api_key,
            )
            results_by_index[index] = result
            log_result(index, total, result, label)

            if args.delay_sec > 0 and index < total:
                time.sleep(args.delay_sec)
    else:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.concurrent_n
        ) as executor:
            futures: dict[concurrent.futures.Future[tuple[int, dict[str, Any], str]], int] = {}
            for index, probe_file in indexed_probe_files:
                future = executor.submit(
                    evaluate_one_indexed,
                    index=index,
                    total=total,
                    probe_file=probe_file,
                    args=args,
                    api_key=api_key,
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
