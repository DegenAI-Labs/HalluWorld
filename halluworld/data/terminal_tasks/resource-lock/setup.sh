#!/bin/bash
set -euo pipefail

sqlite3 /workspace/database.db 'CREATE TABLE kv (id INTEGER PRIMARY KEY, val INTEGER); INSERT INTO kv VALUES (1, 1);'

cat > /workspace/db_locker.sh <<'EOF'
#!/bin/bash
exec flock -x /workspace/database.db -c 'echo $$ > /workspace/.grader_lock_pid; exec sleep infinity'
EOF
chmod +x /workspace/db_locker.sh
