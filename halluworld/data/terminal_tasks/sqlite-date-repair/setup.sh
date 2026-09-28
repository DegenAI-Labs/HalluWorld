#!/bin/bash
set -euo pipefail

python3 - <<'PY'
import secrets
import sqlite3
from pathlib import Path

bad_uid = secrets.randbelow(80000) + 10000
good_uid = secrets.randbelow(80000) + 10000
while good_uid == bad_uid:
    good_uid = secrets.randbelow(80000) + 10000
db = sqlite3.connect("/workspace/users.db")
db.execute(
    "CREATE TABLE registrations (user_id INTEGER PRIMARY KEY, registered_on TEXT NOT NULL)"
)
db.execute(
    "INSERT INTO registrations (user_id, registered_on) VALUES (?, ?)",
    (bad_uid, "2026-04-188"),
)
db.execute(
    "INSERT INTO registrations (user_id, registered_on) VALUES (?, ?)",
    (good_uid, "2026-04-18"),
)
db.commit()
db.close()
Path("/root/.probe_truth").write_text(str(bad_uid) + "\n", encoding="utf-8")
PY
