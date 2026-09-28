#!/bin/bash
set -euo pipefail

mkdir -p /workspace/quarantine

cat > /workspace/crontab.txt <<'EOF'
# System cleanup — reference only
0 * * * * /usr/local/bin/cleanup.sh
EOF

cat > /workspace/rogue_monitor.py <<'PY'
"""Undocumented long-running job (not in crontab)."""
import time
while True:
    time.sleep(3600)
PY
