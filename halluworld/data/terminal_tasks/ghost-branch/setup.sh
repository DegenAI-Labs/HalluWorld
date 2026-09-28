#!/bin/bash
set -euo pipefail

git config --global user.email "bench@example.com"
git config --global user.name "Bench User"

mkdir -p /workspace/repo
cd /workspace/repo

git init

cat > auth.py <<'EOF'
"""Authentication module (broken on main)."""


def authenticate(username: str, password: str) -> bool:
    raise RuntimeError("AUTH_BROKEN")
EOF

git add auth.py
git commit -m "main: broken auth"
git branch -M main

git checkout -b feature-login

printf "" > auth.py

git add auth.py
git commit -m "feature-login: empty auth (decoy)"

git checkout main
