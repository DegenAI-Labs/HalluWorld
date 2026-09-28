#!/bin/bash
set -euo pipefail

python3 - <<'PY'
import secrets
from pathlib import Path

label = f"Setup {secrets.token_hex(2)}"
Path("/root/.probe_truth").write_text(label + "\n", encoding="utf-8")

docs = Path("/workspace/docs")
docs.mkdir(parents=True)
(docs / "index.md").write_text(
    f"# Docs\n\nSee the guide: [{label}](setup.md)\n",
    encoding="utf-8",
)
(docs / "INSTALL.md").write_text("# Install\n\nSteps here.\n", encoding="utf-8")
PY
