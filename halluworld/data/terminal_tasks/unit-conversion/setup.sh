#!/bin/bash
set -euo pipefail

python3 - <<'PY'
import csv
from pathlib import Path

rows = []
for i in range(1, 51):
    rows.append({"trip_id": f"T-{i:03d}", "distance_km": 10.0 + (i % 40)})
# outlier: meters not km
rows.append({"trip_id": "T-OUT-7", "distance_km": 50000.0})

with open("/workspace/trips.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["trip_id", "distance_km"])
    w.writeheader()
    w.writerows(rows)
PY
