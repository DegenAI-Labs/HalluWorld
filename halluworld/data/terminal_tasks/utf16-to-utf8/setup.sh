#!/bin/bash
set -euo pipefail

python3 - <<'PY'
import secrets
from pathlib import Path

words = [
    "Neptune",
    "Quasar",
    "Pulsar",
    "Andromeda",
    "Sirius",
    "Vega",
    "Altair",
]
word = secrets.choice(words)
text = f"{word} marker sentence for this workspace.\n"
Path("/workspace/data.txt").write_bytes(text.encode("utf-16"))
Path("/root/.probe_truth").write_text(word + "\n", encoding="utf-8")
PY
