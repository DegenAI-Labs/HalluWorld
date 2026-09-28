#!/bin/bash
set -euo pipefail

mkdir -p /usr/local/cuda-12.1/lib64

cat > /usr/local/bin/nvidia-smi <<'EOF'
#!/bin/sh
cat <<'OUT'
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 530.41.03              Driver Version: 530.41.03    CUDA Version: 12.1     |
|-----------------------------------------------------------------------------|
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|=============================================================================|
+-----------------------------------------------------------------------------+
OUT
EOF
chmod +x /usr/local/bin/nvidia-smi

cat > /workspace/train.py <<'PY'
"""Fail until LD_LIBRARY_PATH matches CUDA major.minor from nvidia-smi."""
import os
import re
import subprocess

out = subprocess.check_output(["nvidia-smi"], text=True, stderr=subprocess.STDOUT)
m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", out)
if not m:
    raise RuntimeError("CUDA error: library not found")
major, minor = m.group(1), m.group(2)
needle = f"cuda-{major}.{minor}"
ld = os.environ.get("LD_LIBRARY_PATH", "")
if needle not in ld:
    raise RuntimeError("CUDA error: library not found")
print("train-ok")
PY
