#!/bin/bash
set -euo pipefail

mkdir -p /workspace/logs

cat > /workspace/monitor.sh <<'EOF'
#!/bin/bash
trap 'printf "kernel_event exit_signal=SIGTERM exit_code=15 signal_name=SIGTERM\n" >> /workspace/logs/kern_events.log' TERM
exec sleep 999
EOF
chmod +x /workspace/monitor.sh

cat > /workspace/run_monitor_once.sh <<'EOF'
#!/bin/bash
set -euo pipefail
/workspace/monitor.sh &
MPID=$!
sleep 2
kill -TERM "$MPID" 2>/dev/null || true
wait "$MPID" 2>/dev/null || true
EOF
chmod +x /workspace/run_monitor_once.sh
