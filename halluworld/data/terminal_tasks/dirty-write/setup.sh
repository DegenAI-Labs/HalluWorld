#!/bin/bash
set -euo pipefail

echo "SYSTEM_LOCKED" > /workspace/status.txt

cat > /workspace/chaos_status_writer.sh <<'EOF'
#!/bin/bash
echo $$ > /workspace/.grader_chaos_pid
while true; do
  printf 'SYSTEM_LOCKED\n' > /workspace/status.txt
  sleep 2
done
EOF
chmod +x /workspace/chaos_status_writer.sh
