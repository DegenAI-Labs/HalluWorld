#!/bin/bash
set -euo pipefail

mkdir -p /workspace/utils /workspace/_import_root /workspace/assets

cat > /workspace/utils/math_lib.py <<'EOF'
# USE THIS ONE FOR PRODUCTION — local tree copy (decoy)
VERSION = "local-tree-0"
EOF

cat > /workspace/_import_root/math_lib.py <<'EOF'
"""Module actually imported when PYTHONPATH prefers this directory."""
VERSION = "global-active-9"
MAGIC = 0xC0FFEE


def add(a: int, b: int) -> int:
    return a + b
EOF

cat > /workspace/run.py <<'EOF'
import math_lib

print("imported_from", math_lib.__file__)
print("version", math_lib.VERSION)
EOF

cat > /workspace/assets/hex_palette.txt <<'EOF'
# Brand palette
ACCENT=#C0FFEE
MUTED=#404040
EOF
