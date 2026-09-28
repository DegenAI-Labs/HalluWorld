#!/bin/sh
set -e
nohup /workspace/db_locker.sh >/tmp/db_locker.log 2>&1 &
exec "$@"
