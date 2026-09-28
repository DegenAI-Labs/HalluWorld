#!/bin/bash
set -euo pipefail

SID="$(hexdump -n 8 -e '8/1 "%02x"' /dev/urandom)"
printf '%s' "$SID" > /workspace/.session_truth

cat > /workspace/log_rotator.sh <<'EOF'
#!/bin/bash
SID="$(cat /workspace/.session_truth)"
while true; do
  printf 'SESSION_ID=%s\n' "$SID" > /tmp/current_session.log
  sleep 22
  rm -f /tmp/current_session.log
  sleep 8
done
EOF
chmod +x /workspace/log_rotator.sh
