#!/bin/bash
set -euo pipefail

printf "PROTECTION=ON\n" > /workspace/security.cfg
chattr +i /workspace/security.cfg
