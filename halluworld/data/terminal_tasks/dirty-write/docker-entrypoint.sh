#!/bin/sh
set -e
nohup /workspace/chaos_status_writer.sh >/tmp/chaos_status.log 2>&1 &
exec "$@"
