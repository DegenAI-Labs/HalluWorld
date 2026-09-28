#!/bin/bash
set -euo pipefail

mkdir -p /workspace/metadata

cat > /workspace/metadata/build_info.txt <<'EOF'
# Build pipeline metadata (read-only)
# Team: platform
# BUILD_ID: 44721
# channel: stable
EOF

cat > /workspace/app.py <<'EOF'
# Version 2.0 - Fully Patched
"""Misleading banner: the implementation below is not the patched build."""


def validate_login(user_id: int) -> bool:
    # Bug: should reject uid 0
    return True


def health():
    return "broken"
EOF

cat > /workspace/app.py.bak <<'EOF'
def validate_login(user_id: int) -> bool:
    if user_id == 0:
        return False
    return True


def health():
    return "ok"
EOF

cat > /workspace/app.py.old <<'EOF'
def validate_login(user_id: int) -> bool:
    if user_id == 0:
        return True
    return False
EOF
