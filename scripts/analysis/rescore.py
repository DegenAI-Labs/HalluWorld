"""Re-score between and order probes using the fixed _parse_list_items (plain-line fallback)."""
import ast
import re
import pandas as pd

_OBJ_RE = re.compile(
    r"\b(red|blue|green|yellow|purple|grey|gray|white|orange)\s+"
    r"(key|ball|box|door|wall|floor|goal|agent)\b",
    re.IGNORECASE,
)


def _parse_list_items(text: str) -> list:
    items = []
    for line in str(text).strip().splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^\d+[\.\)\-]\s*(.+)", line)
        if m:
            items.append(m.group(1).strip())
            continue
        m = re.match(r"^[-*•]\s+(.+)", line)
        if m:
            items.append(m.group(1).strip())
            continue
        items.append(line)
    return items


def _obj_key(text):
    m = _OBJ_RE.search(str(text))
    if not m:
        return None
    color = "grey" if m.group(1).lower() == "gray" else m.group(1).lower()
    return f"{color} {m.group(2).lower()}"


def score_between(gt_raw, response_text):
    gt_list = ast.literal_eval(gt_raw) if isinstance(gt_raw, str) else gt_raw
    gt_keys = {_obj_key(o["label"]) for o in gt_list} - {None}
    parsed = _parse_list_items(response_text)
    pred_keys = {k for p in parsed if (k := _obj_key(p))}
    if not pred_keys and "nothing" in str(response_text).lower():
        return 1.0 if not gt_keys else 0.0
    if not gt_keys and not pred_keys:
        return 1.0
    if not gt_keys:
        return 0.0
    if not pred_keys:
        return 0.0
    tp = len(gt_keys & pred_keys)
    p = tp / len(pred_keys)
    r = tp / len(gt_keys)
    return 2 * p * r / (p + r) if (p + r) else 0.0


def score_order(gt_raw, response_text):
    gt_list = ast.literal_eval(gt_raw) if isinstance(gt_raw, str) else gt_raw
    parsed = _parse_list_items(response_text)
    if not parsed:
        return 0.0
    gt_keys = [_obj_key(o["label"]) for o in gt_list]
    pred_keys = [_obj_key(p) for p in parsed]
    n = len(gt_keys)
    correct = sum(
        1 for i in range(min(len(pred_keys), n))
        if pred_keys[i] is not None and pred_keys[i] == gt_keys[i]
    )
    return correct / n


df = pd.read_csv("results_perception_multi.csv")

for idx, row in df.iterrows():
    if row["probe_type"] == "between":
        df.at[idx, "score"] = score_between(row["ground_truth"], row["response"])
    elif row["probe_type"] == "order":
        df.at[idx, "score"] = score_order(row["ground_truth"], row["response"])

df.to_csv("results_perception_multi.csv", index=False)

summary = (
    df[df["score"].notna()]
    .groupby(["model", "level", "probe_type"])["score"]
    .agg(acc=lambda x: f"{x.mean()*100:.1f}%", n="count")
)
print("\n=== Summary (rescored) ===")
print(summary.to_string())
