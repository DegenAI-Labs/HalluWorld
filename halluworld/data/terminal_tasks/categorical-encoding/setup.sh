#!/bin/bash
set -euo pipefail

python3 - <<'PY'
import csv
from pathlib import Path

cats = ["Small", "Medium", "Large"]
rows = []
for i in range(99):
    rows.append({"sku": f"SKU-{i+1:03d}", "category": cats[i % 3]})
rows.append({"sku": "SKU-100", "category": "X-Large"})

with open("/workspace/inventory.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["sku", "category"])
    w.writeheader()
    w.writerows(rows)
PY
