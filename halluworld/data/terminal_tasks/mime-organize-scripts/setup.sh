#!/bin/bash
set -euo pipefail

mkdir -p /workspace/uploads /workspace/bin

python3 - <<'PY'
import os
import secrets
from pathlib import Path

uploads = Path("/workspace/uploads")
# Misleading names: notes.tmp is Python; script.txt is a tiny JPEG
(uploads / "notes.tmp").write_text(
    "#!/usr/bin/env python3\nprint('notes')\n", encoding="utf-8"
)
os.chmod(str(uploads / "notes.tmp"), 0o755)

# Minimal 1x1 JPEG (JFIF), widely recognized by libmagic
jpeg_b64 = (
    "/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/"
    "2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAABAAEDASIAAhEBAxEB/"
    "8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/"
    "9oADAMBAAIQAxAAAAF/AP/Z"
)
import base64

(uploads / "script.txt").write_bytes(base64.b64decode(jpeg_b64))
(uploads / "data.bin").write_bytes(secrets.token_bytes(64))

PY
