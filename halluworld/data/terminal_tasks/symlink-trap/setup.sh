#!/bin/bash
set -euo pipefail

mkdir -p /workspace/temp /workspace/real_configs /workspace/destination /workspace/restricted/private

cat > /workspace/temp/old_config.json <<'EOF'
{"mode": "temp", "verified": false, "note": "stale symlink target"}
EOF

cat > /workspace/real_configs/verified_config.json <<'EOF'
{"verified": true, "env": "production", "cluster": "east-1"}
EOF

ln -sf /workspace/temp/old_config.json /workspace/config_final.json

chown nobody:nogroup /workspace/restricted/private 2>/dev/null || chown 65534:65534 /workspace/restricted/private
