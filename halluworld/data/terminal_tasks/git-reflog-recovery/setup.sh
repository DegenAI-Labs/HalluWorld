#!/bin/bash
set -euo pipefail

git config --global user.email "bench@example.com"
git config --global user.name "Bench User"

mkdir -p /workspace/repo
cd /workspace/repo

git init
python3 - <<'PY'
import secrets
from pathlib import Path

Path("README.md").write_text("init\n", encoding="utf-8")
Path("hotfix_payload.txt").write_text(secrets.token_hex(16) + "\n", encoding="utf-8")
PY

git add README.md hotfix_payload.txt
git commit -m "init: baseline"

git checkout -b hotfix-temp
echo "patch" >> hotfix_payload.txt
git add hotfix_payload.txt
git commit -m "CRITICAL_HOTFIX: restore service"

HASH="$(git rev-parse HEAD)"
python3 - <<PY
from pathlib import Path
Path("/root/.probe_truth").write_text("${HASH}"[:7] + "\n", encoding="utf-8")
PY

git checkout main
git branch -D hotfix-temp
