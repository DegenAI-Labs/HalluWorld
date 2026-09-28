import pandas as pd

df = pd.read_csv("results_dynamics_v2.csv")
print(f"Total rows: {df.shape[0]}\n")

rows = []
for (model, hint), mdf in df.groupby(["model", "wind_hint_mode"]):
    wind_df = mdf[mdf["has_wind"] & (mdf["wind_offset"] != 0)]
    nowind_df = mdf[~(mdf["has_wind"] & (mdf["wind_offset"] != 0))]
    acc = mdf["correct"].mean()
    hall = 1.0 - mdf["score"].mean()
    parse_fail = (mdf["parse_note"] != "ok").sum()
    wind_acc = wind_df["correct"].mean() if len(wind_df) else None
    wind_naive = wind_df["wind_naive_error"].mean() if len(wind_df) else None
    rows.append({
        "model": model,
        "hint": hint,
        "n": len(mdf),
        "accuracy": round(acc, 3),
        "hallucination": round(hall, 3),
        "parse_fail": parse_fail,
        "wind_n": len(wind_df),
        "wind_acc": round(wind_acc, 3) if wind_acc is not None else "—",
        "wind_naive_err": round(wind_naive, 3) if wind_naive is not None else "—",
        "nowind_acc": round(nowind_df["correct"].mean(), 3) if len(nowind_df) else "—",
    })
print(pd.DataFrame(rows).to_string(index=False))
