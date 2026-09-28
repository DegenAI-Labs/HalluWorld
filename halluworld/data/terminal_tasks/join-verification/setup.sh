#!/bin/bash
set -euo pipefail

python3 - <<'PY'
import csv
from pathlib import Path

with open("/workspace/customers.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["customer_id", "name"])
    w.writeheader()
    for i in range(1, 11):
        w.writerow({"customer_id": str(i), "name": f"Customer {i}"})

with open("/workspace/orders.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["order_id", "customer_id", "amount"])
    w.writeheader()
    for i in range(1, 11):
        cid = str(i) if i != 6 else "999"
        w.writerow({"order_id": f"ORD-{i:03d}", "customer_id": cid, "amount": str(10 * i)})
PY
