#!/bin/bash
set -euo pipefail

python3 - <<'PY'
import secrets
from pathlib import Path

host = "prod_" + secrets.token_hex(6)
Path("/workspace/polluted_env.sh").write_text(
    f"export DB_HOST={host}\n", encoding="utf-8"
)
Path("/workspace/.env").write_text("DB_HOST=localhost\n", encoding="utf-8")
Path("/root/.probe_truth").write_text(host + "\n", encoding="utf-8")
PY

printf '%s\n' "source /workspace/polluted_env.sh" >> /root/.bashrc
