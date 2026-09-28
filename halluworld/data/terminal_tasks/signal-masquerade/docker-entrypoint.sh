#!/bin/sh
set -e
/workspace/run_monitor_once.sh
exec "$@"
