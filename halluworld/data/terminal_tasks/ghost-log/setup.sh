#!/bin/bash
set -euo pipefail

mkdir -p /workspace/logs /workspace/system

# Stale "main" log — large, last entries are old (2025)
{
  for i in $(seq 1 400); do
    echo "2025-06-15T12:00:00Z FAILED_LOGIN ip=198.51.100.$((i % 200)) user=svc"
  done
  echo "2025-12-31T23:59:59Z FAILED_LOGIN ip=198.51.100.99 user=admin"
} > /workspace/logs/access.log

# Rotated log — actually current (2026); contains the real recent failure
cat > /workspace/logs/access.log.1 <<'EOF'
2026-04-17T08:22:01Z OK ip=192.0.2.10 user=alice
2026-04-18T09:15:33Z FAILED_LOGIN ip=203.0.113.44 user=root
2026-04-18T09:16:01Z OK ip=192.0.2.10 user=alice
EOF

python3 - <<'PY'
from pathlib import Path
lines = ["{"] + [f'  "slot_{i}": {i},' for i in range(11)] + ['  "end": true', "}"]
path = Path("/workspace/system/manifest.json")
path.write_text("\n".join(lines) + "\n")
assert len(path.read_text().splitlines()) == 14
PY
