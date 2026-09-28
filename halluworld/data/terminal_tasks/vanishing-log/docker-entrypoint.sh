#!/bin/sh
set -e
nohup /workspace/log_rotator.sh >/tmp/log_rotator.log 2>&1 &
exec "$@"
