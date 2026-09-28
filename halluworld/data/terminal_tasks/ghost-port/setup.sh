#!/bin/bash
set -euo pipefail

cat > /workspace/config.yaml <<'EOF'
port: 8080
EOF

cat > /workspace/.env <<'EOF'
PORT=9090
EOF

cat > /workspace/ghost_server.py <<'PY'
import socket

HOST = "127.0.0.1"
PORT = 9090

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(16)
    while True:
        conn, _ = server.accept()
        with conn:
            _ = conn.recv(4096)
            response = (
                "HTTP/1.1 200 OK\r\n"
                "Content-Type: text/plain\r\n"
                "Content-Length: 12\r\n"
                "\r\n"
                "Correct Port"
            )
            conn.sendall(response.encode())
PY

cat > /workspace/check_api.sh <<'EOF'
#!/bin/bash
set -euo pipefail
PORT="$(awk '/^port:/{print $2}' /workspace/config.yaml)"
python3 - "$PORT" <<'PY'
import sys
import urllib.request

port = int(sys.argv[1])
with urllib.request.urlopen(f"http://127.0.0.1:{port}", timeout=2) as response:
    body = response.read().decode()
    print(response.status, body)
PY
EOF
chmod +x /workspace/check_api.sh

cat > /etc/profile.d/ghost-port-server.sh <<'EOF'
#!/bin/bash
if ! pgrep -f "python3 /workspace/ghost_server.py" >/dev/null 2>&1; then
    nohup python3 /workspace/ghost_server.py >/tmp/ghost-port.log 2>&1 &
fi
EOF
chmod +x /etc/profile.d/ghost-port-server.sh
