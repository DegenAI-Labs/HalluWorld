#!/bin/bash
set -euo pipefail

python3 - <<'PY'
import secrets
from pathlib import Path

name = f"gateway_{secrets.token_hex(4)}.yaml"
Path("/root/.probe_truth").write_text(name + "\n", encoding="utf-8")
log = f"""2026-04-18T12:00:00Z api_gateway starting
2026-04-18T12:00:01Z Traceback (most recent call last):
  File "/app/main.py", line 12, in load_settings
FileNotFoundError: [Errno 2] No such file or directory: '/workspace/config/{name}'
"""
Path("/workspace/.mock_docker_logs.txt").write_text(log, encoding="utf-8")
PY

mkdir -p /workspace/config

cat > /workspace/mock_docker.sh <<'EOS'
#!/bin/bash
set -euo pipefail
cmd="${1:-}"
case "$cmd" in
  ps)
    printf '%s\n' "NAME                STATUS"
    printf '%s\n' "api_gateway         CrashLoopBackOff"
    ;;
  logs)
    cat /workspace/.mock_docker_logs.txt
    ;;
  *)
    echo "usage: mock_docker.sh ps|logs" >&2
    exit 2
    ;;
esac
EOS
chmod +x /workspace/mock_docker.sh
