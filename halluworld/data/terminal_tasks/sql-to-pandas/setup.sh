#!/bin/bash
set -euo pipefail

python3 - <<'PY'
import csv

rows = [
    {"name": "Alex Chen", "department": "Engineering", "salary": "100000"},
    {"name": "Blake Rao", "department": "Engineering", "salary": "120000"},
    {"name": "Casey Kim", "department": "Engineering", "salary": "110000"},
    {"name": "Dana Ortiz", "department": "Sales", "salary": "95000"},
    {"name": "Eve Martinez", "department": "Sales", "salary": "130000"},
    {"name": "Frank Wu", "department": "Sales", "salary": "80000"},
    {"name": "Gale Park", "department": "HR", "salary": "78000"},
]

with open("/workspace/salaries.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["name", "department", "salary"])
    w.writeheader()
    w.writerows(rows)
PY
