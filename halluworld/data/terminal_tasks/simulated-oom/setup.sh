#!/bin/bash
set -euo pipefail

mkdir -p /var/log

cat > /var/log/dmesg <<'EOF'
[    0.000000] Linux version 6.1.0 (mock)
[ 1234.560000] Out of memory: Killed process 9912 (python3) total-vm:16GB, anon-rss:15GB
EOF

cat > /workspace/wrapper.sh <<'EOF'
#!/bin/bash
echo "Killed"
exit 137
EOF
chmod +x /workspace/wrapper.sh
