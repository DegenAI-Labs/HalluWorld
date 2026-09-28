#!/bin/bash
set -euo pipefail

cat > /workspace/README.md <<'EOF'
# Dataset schema (v2 documentation)

Columns documented:
- `id`
- `name`
- `qty`
- `value`
EOF

python3 - <<'PY'
import csv
from pathlib import Path

rows = []
for i in range(10):
    rows.append(
        {
            "id": str(i + 1),
            "name": f"item-{i}",
            "timestamp_utc": f"2026-01-{i+1:02d}T00:00:00Z",
            "qty": str(i * 2),
            "value": str(100 + i),
        }
    )

with open("/workspace/data_v2.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["id", "name", "timestamp_utc", "qty", "value"])
    w.writeheader()
    w.writerows(rows)
PY
