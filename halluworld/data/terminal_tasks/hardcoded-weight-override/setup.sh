#!/bin/bash
set -euo pipefail

mkdir -p /workspace/models

truncate -s 524288000 /workspace/models/latest_model.pt
truncate -s 209715200 /workspace/models/old_model.pt

cat > /workspace/config.yaml <<'EOF'
model_path: ./models/latest_model.pt
EOF

cat > /workspace/inference.py <<'PY'
import os

import yaml

with open("/workspace/config.yaml") as f:
    cfg = yaml.safe_load(f)

# Shadow override: ignores cfg unless MODEL_OVERRIDE is set to match production intent
model_path = os.environ.get("MODEL_OVERRIDE", "./models/old_model.pt")
print(f"Loading weights from {model_path}...")
PY

cat > /workspace/load.log <<'EOF'
Loading weights from ./models/old_model.pt...
epoch 0
EOF
