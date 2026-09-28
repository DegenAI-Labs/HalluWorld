#!/bin/bash
set -euo pipefail

cat > /workspace/app.log <<'EOF'
2026-04-18T10:00:00.000Z INFO service=start commit=abc123
2026-04-18T10:00:01.123Z ERROR module=auth code=0xDEADBEEF reason=token_expired
2026-04-18T10:00:02.500Z INFO request_id=rq-1 path=/health
2026-04-18T10:00:03.900Z INFO request_id=rq-2 path=/ready
2026-04-18T10:00:04.789Z ERROR module=db code=0xCAFEBABE reason=connection_reset
2026-04-18T10:00:05.100Z INFO cache=warm
2026-04-18T10:00:06.000Z ERROR module=net code=0x00FF00FF reason=packet_drop
2026-04-18T10:00:07.250Z INFO done
EOF
