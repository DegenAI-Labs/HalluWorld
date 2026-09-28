#!/bin/sh
set -e
python3 /workspace/ipc_fill.py
exec "$@"
