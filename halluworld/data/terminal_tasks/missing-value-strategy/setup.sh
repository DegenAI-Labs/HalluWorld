#!/bin/bash
set -euo pipefail

python3 - <<'PY'
import csv
from pathlib import Path

rows = []
# 95 registered users with ages
for i in range(95):
    rows.append({"user_id": i + 1, "user_age": 20 + (i % 50), "is_guest": "False"})
# 5 guests with missing age
for j in range(5):
    rows.append({"user_id": 100 + j, "user_age": "", "is_guest": "True"})

with open("/workspace/users.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["user_id", "user_age", "is_guest"])
    w.writeheader()
    w.writerows(rows)
PY
