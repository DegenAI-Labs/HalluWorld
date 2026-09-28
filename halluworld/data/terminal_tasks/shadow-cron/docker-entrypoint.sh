#!/bin/sh
set -e
nohup python3 /workspace/rogue_monitor.py >/tmp/rogue_monitor.log 2>&1 &
exec "$@"
