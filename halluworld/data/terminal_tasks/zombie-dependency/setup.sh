#!/bin/bash
set -euo pipefail

python3 -m pip install --no-cache-dir requests

python3 - <<'PY'
import pathlib
import site

site_packages = pathlib.Path(site.getsitepackages()[0])
sessions_path = site_packages / "requests" / "sessions.py"
text = sessions_path.read_text()

class_start = text.find("class Session(")
if class_start == -1:
    raise RuntimeError("Could not locate class Session in requests/sessions.py")

next_class = text.find("\nclass ", class_start + 1)
if next_class == -1:
    next_class = len(text)

text = text[:class_start] + "\n# Session definition removed intentionally.\n" + text[next_class:]
sessions_path.write_text(text)
PY

cat > /workspace/main.py <<'PY'
from requests import Session

session = Session()
print("session-ok", isinstance(session, Session))
PY
