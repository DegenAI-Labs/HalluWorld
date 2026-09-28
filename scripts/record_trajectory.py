#!/usr/bin/env python3
"""Interactive trajectory recorder for HalluWorld levels.

Walk through a level yourself, then plant probes at exact steps.
Supports multi-room trajectories: swap to a different level file at any point
(e.g. foyer → kitchen → foyer-changed) using the Change Room panel.
Saves a JSON trajectory that can be replayed by scripts/run_trajectory_eval.py.

Usage
-----
New trajectory:
    conda run -n halluworld python record_trajectory.py \\
        --level levels/h1_foyer.txt --seed 42

Specify output path:
    conda run -n halluworld python record_trajectory.py \\
        --level levels/h1_foyer.txt --seed 42 \\
        --out trajectories/foyer_s42.json

Resume / extend an existing trajectory:
    conda run -n halluworld python record_trajectory.py \\
        --resume trajectories/foyer_s42.json

Then open http://localhost:5050 in your browser.
VS Code auto-forwards the port if you're on a remote machine.

Browser keyboard shortcuts
--------------------------
W / ↑       Move forward
A / ←       Turn left
D / →       Turn right
T / E       Toggle  (interact with door, bookshelf, etc.)
G           Pick up
F           Drop
Z           Undo last action
P           Focus the probe question field
Ctrl+S      Save trajectory to disk

Multi-room trajectory format
-----------------------------
{
  "segments": [
    {"level_file": str(LEVELS_DIR / "h1_foyer.txt"),         "seed": 42, "actions": [...]},
    {"level_file": str(LEVELS_DIR / "h2_kitchen.txt"),        "seed": 42, "actions": [...]},
    {"level_file": str(LEVELS_DIR / "h1_foyer_changed.txt"),  "seed": 42, "actions": [...]}
  ],
  "probes": [
    {"segment": 0, "step": 5, "probe_type": "presence", "question": "...",
     "ground_truth": "true", "metadata": {}},
    ...
  ]
}

Old single-room format (level_file / seed / actions at top level) is still
accepted on --resume and is automatically upgraded to the segments format.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from flask import Flask, jsonify, request  # noqa: E402

from halluworld.tracks.grid.envs.ascii_env import AsciiEnv  # noqa: E402
from halluworld.tracks.grid.serializers.memory import MemorySerializer  # noqa: E402
from halluworld.data import LEVELS_DIR

app = Flask(__name__)
_ser = MemorySerializer()

# ── Global state (single-user tool — no concurrency needed) ───────────────────
#
# segments: list of {level_file, seed, actions}  — one entry per room visit
# current_segment: index into segments of the room we are currently in
# probes: list of {segment, step, probe_type, question, ground_truth, metadata}
#
_state: dict = {
    "segments": [],       # list[dict]
    "current_segment": 0,
    "probes": [],
    "out_path": "",
    "env": None,          # current AsciiEnv (for the active segment)
}

# ── Env helpers ───────────────────────────────────────────────────────────────

def _build_env(level_file: str, seed: int, actions: list[int]) -> AsciiEnv:
    """Create a fresh env and replay *actions* to reconstruct state."""
    env = AsciiEnv.from_file(level_file)
    env.reset(seed=seed)
    for act in actions:
        env.step(act)
    return env


def _current_seg() -> dict:
    return _state["segments"][_state["current_segment"]]


def _current_step() -> int:
    return len(_current_seg()["actions"])




_DIR_CHAR: dict[int, str] = {0: ">", 1: "v", 2: "<", 3: "^"}
_COLOR_INIT: dict[str, str] = {
    "red": "r", "blue": "b", "green": "g", "yellow": "y",
    "purple": "p", "grey": "e", "orange": "o",
}
_ACTION_LABEL: dict[int, str] = {
    0: "←L", 1: "R→", 2: "▲F", 3: "↑G", 4: "↓D", 5: "⊕T",
}


def _cell_token(cell) -> str:
    """Two-character ASCII token for a non-agent grid cell."""
    t = cell.type
    ci = _COLOR_INIT.get(getattr(cell, "color", ""), "?").upper()

    if t == "wall":
        return "##"
    if t == "door":
        state = "O" if cell.is_open else ("L" if cell.is_locked else "D")
        return f"{state}{ci}"
    if t in ("key", "ball", "box"):
        return f"{t[0].upper()}{ci}"
    if t == "fire":
        return "~~" if getattr(cell, "active", True) else ".f"
    if t == "torch":
        return "yH" if getattr(cell, "is_lit", False) else "eH"
    if t == "flood":
        return "FF" if getattr(cell, "flooded", False) else ".f"
    if t == "noticeboard":
        return "NB"
    if t == "signpost":
        return "SG"
    if t == "boulder":
        return "BO"
    if t == "diningtable":
        return "DT"
    if t == "bookshelf":
        return "BS"
    if t == "candelabra":
        return "CB"
    if t == "npc":
        return "qN"
    if t == "lockedchest":
        return "LC"
    if t == "hiddencompartment":
        return "hC" if not getattr(cell, "revealed", False) else "HC"
    if t == "water":
        return "WW"
    if t == "river":
        return "~~"
    if t == "firesource":
        return "FS"
    if t == "pressureplate":
        return "PP"
    if t == "zonedoor":
        return "ZZ"
    if t == "elevated":
        return "ET"
    if t == "mud":
        return "MM"
    if t == "tree":
        return "TT"
    if t == "darkzone":
        return "DZ"
    return t[:2].upper()


def _render_grid(env: AsciiEnv) -> str:
    """Return the full grid as a monospace ASCII string for the browser panel."""
    ax, ay = int(env.agent_pos[0]), int(env.agent_pos[1])
    agent_ch = _DIR_CHAR.get(int(env.agent_dir), "A")
    rows = []
    for y in range(env.height):
        row = []
        for x in range(env.width):
            if x == ax and y == ay:
                row.append(f" {agent_ch}")
            else:
                cell = env.grid.get(x, y)
                row.append(" ." if cell is None else _cell_token(cell))
        rows.append(" ".join(row))
    return "\n".join(rows)


def _current_obs() -> str:
    step = _current_step()
    return _ser.serialize(_state["env"], step=step)


def _state_snapshot() -> dict:
    seg = _current_seg()
    env = _state["env"]
    n_seg = len(_state["segments"])
    cur = _state["current_segment"]
    # Detect adjacent zone doors to hint the user
    zone_doors = []
    zd_map = getattr(env, "_zone_doors", {})
    for (zx, zy), zd in zd_map.items():
        zone_doors.append({
            "pos": [zx, zy],
            "zone_name": zd.zone_name,
            "direction": zd.direction,
            "file_path": zd.file_path,
        })
    return {
        "step":             _current_step(),
        "segment":          cur,
        "n_segments":       n_seg,
        "level_file":       seg["level_file"],
        "seed":             seg["seed"],
        "out_path":         _state["out_path"],
        "grid":             _render_grid(env),
        "obs":              _current_obs(),
        "agent_pos":        list(map(int, env.agent_pos)),
        "agent_dir":        int(env.agent_dir),
        "probes":           _state["probes"],
        "zone_doors":       zone_doors,
        "segment_summaries": [
            {"level_file": s["level_file"], "seed": s["seed"],
             "n_actions": len(s["actions"])}
            for s in _state["segments"]
        ],
    }


# ── Flask routes ──────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return _HTML_PAGE


@app.route("/state")
def get_state():
    return jsonify(_state_snapshot())


@app.route("/action", methods=["POST"])
def take_action():
    data = request.get_json(force=True)
    act = data.get("action", -1)
    if act not in range(6):
        return jsonify({"error": f"invalid action {act!r}"}), 400
    _state["env"].step(int(act))
    _current_seg()["actions"].append(int(act))
    return get_state()


@app.route("/undo", methods=["POST"])
def undo():
    seg = _current_seg()
    if not seg["actions"]:
        # Can't undo past a room transition
        return jsonify({**_state_snapshot(), "warning": "At segment start — undo stops here."})
    # Remove probes planted at the step about to be erased
    cur_seg_idx = _state["current_segment"]
    cur_step = len(seg["actions"])
    _state["probes"] = [
        p for p in _state["probes"]
        if not (p["segment"] == cur_seg_idx and p["step"] == cur_step)
    ]
    seg["actions"].pop()
    _state["env"] = _build_env(seg["level_file"], seg["seed"], seg["actions"])
    return get_state()


@app.route("/change_room", methods=["POST"])
def change_room():
    """Start a new segment (room transition).

    Body: {level_file: str, seed: int}
    Creates a new segment and makes it current.
    If the caller is mid-trajectory on the last segment, it appends a new one.
    If the caller is NOT on the last segment (resuming and extending), all
    segments after the current one are dropped first.
    """
    data = request.get_json(force=True)
    level_file = (data.get("level_file") or "").strip()
    if not level_file:
        return jsonify({"error": "level_file is required"}), 400
    if not os.path.exists(level_file):
        return jsonify({"error": f"File not found: {level_file}"}), 400
    seed = int(data.get("seed", 0))

    cur = _state["current_segment"]
    # Drop any segments that come after the current position
    _state["segments"] = _state["segments"][: cur + 1]
    # Drop probes that belong to dropped segments
    _state["probes"] = [p for p in _state["probes"] if p["segment"] <= cur]

    new_seg = {"level_file": level_file, "seed": seed, "actions": []}
    _state["segments"].append(new_seg)
    _state["current_segment"] = len(_state["segments"]) - 1
    _state["env"] = _build_env(level_file, seed, [])
    return get_state()


@app.route("/probe", methods=["POST"])
def add_probe():
    data = request.get_json(force=True)
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "question is required"}), 400
    probe = {
        "segment":    _state["current_segment"],
        "step":       _current_step(),
        "probe_type": data.get("probe_type", "presence"),
        "question":   question,
        "ground_truth": data.get("ground_truth", ""),
        "metadata":   data.get("metadata", {}),
    }
    _state["probes"].append(probe)
    return jsonify({"ok": True, "probe": probe})


@app.route("/delete_probe", methods=["POST"])
def delete_probe():
    data = request.get_json(force=True)
    idx = int(data.get("index", -1))
    if 0 <= idx < len(_state["probes"]):
        _state["probes"].pop(idx)
        return jsonify({"ok": True})
    return jsonify({"error": "index out of range"}), 400


@app.route("/save", methods=["POST"])
def save():
    data = request.get_json(force=True) or {}
    out_path = (data.get("path") or "").strip() or _state["out_path"]
    if not out_path:
        seg0 = _state["segments"][0]
        stem = Path(seg0["level_file"]).stem
        out_path = f"trajectories/{stem}_s{seg0['seed']}.json"
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    trajectory = {
        "segments": _state["segments"],
        "probes":   _state["probes"],
    }
    with open(out_path, "w") as f:
        json.dump(trajectory, f, indent=2)
    _state["out_path"] = out_path
    n_actions = sum(len(s["actions"]) for s in _state["segments"])
    return jsonify({
        "ok":        True,
        "path":      out_path,
        "n_segments": len(_state["segments"]),
        "n_actions": n_actions,
        "n_probes":  len(_state["probes"]),
    })


# ── HTML page (self-contained, no external deps) ─────────────────────────────

_HTML_PAGE = """<!DOCTYPE html>
<html>
<head>
  <title>HalluWorld Trajectory Recorder</title>
  <meta charset="utf-8">
  <style>
    *{box-sizing:border-box;margin:0;padding:0}
    body{font-family:monospace;background:#1e1e1e;color:#d4d4d4;height:100vh;display:flex;flex-direction:column;overflow:hidden}
    /* Header */
    #hdr{background:#252526;padding:7px 14px;display:flex;align-items:center;gap:14px;border-bottom:1px solid #3c3c3c;flex-shrink:0;flex-wrap:wrap}
    #hdr h1{font-size:13px;color:#ccc}
    .meta{font-size:11px;color:#888}
    .badge{background:#0e639c;padding:2px 9px;border-radius:3px;font-size:12px;color:#fff}
    .badge-seg{background:#5a3e8a;padding:2px 9px;border-radius:3px;font-size:12px;color:#fff}
    .saved-path{color:#4ec9b0;font-size:11px;margin-left:auto}
    /* Main */
    #main{display:flex;flex:1;overflow:hidden}
    /* Left panel */
    #left{width:430px;flex-shrink:0;display:flex;flex-direction:column;border-right:1px solid #3c3c3c}
    #grid-wrap{flex:1;padding:10px;overflow:auto;background:#1e1e1e}
    #grid{font-size:12.5px;line-height:1.55;white-space:pre;color:#d4d4d4}
    #ctrl{padding:10px;background:#252526;border-top:1px solid #3c3c3c;display:flex;flex-direction:column;gap:6px}
    #ctrl h3{font-size:10px;text-transform:uppercase;color:#888;margin-bottom:2px}
    .btn-row{display:flex;gap:5px;justify-content:center}
    button{background:#3c3c3c;color:#d4d4d4;border:1px solid #555;padding:5px 9px;cursor:pointer;font-family:monospace;font-size:12px;border-radius:3px;transition:background .1s}
    button:hover{background:#4c4c4c}
    button:active{background:#555}
    .btn-primary{background:#0e639c;border-color:#0e639c}
    .btn-primary:hover{background:#1177bb}
    .btn-danger{background:#7a2222;border-color:#7a2222}
    .btn-danger:hover{background:#9b2b2b}
    .btn-success{background:#2d7a2d;border-color:#2d7a2d}
    .btn-success:hover{background:#3a963a}
    .btn-room{background:#5a3e8a;border-color:#5a3e8a}
    .btn-room:hover{background:#6e4fad}
    .hint{font-size:10px;color:#555;text-align:center;line-height:1.5}
    /* Right panel */
    #right{flex:1;display:flex;flex-direction:column;overflow:hidden}
    #obs-wrap{flex:1;padding:10px;overflow:auto}
    #obs{font-size:12px;line-height:1.6;white-space:pre-wrap;color:#d4d4d4}
    /* Segment breadcrumb */
    #seg-bar{background:#2a2040;border-bottom:1px solid #3c3c3c;padding:5px 10px;font-size:11px;color:#aaa;display:flex;align-items:center;gap:8px;flex-shrink:0;overflow-x:auto;white-space:nowrap}
    .seg-crumb{padding:2px 8px;border-radius:3px;background:#3c3c3c;cursor:default}
    .seg-crumb.active{background:#5a3e8a;color:#fff}
    .seg-arrow{color:#555}
    /* Zone door hint */
    #zdoors{background:#1a2a1a;border-bottom:1px solid #3c3c3c;padding:4px 10px;font-size:11px;color:#5d9c5d;flex-shrink:0;display:none}
    /* Change room panel */
    #room-panel{border-top:1px solid #3c3c3c;background:#1e1a2a;padding:10px;flex-shrink:0;display:none}
    #room-panel.open{display:block}
    #room-panel h3{font-size:10px;text-transform:uppercase;color:#a080d0;margin-bottom:7px}
    .rf{display:grid;grid-template-columns:1fr 80px;gap:6px;align-items:end}
    /* Probe panel */
    #probe-panel{border-top:1px solid #3c3c3c;background:#252526;padding:10px;display:flex;flex-direction:column;gap:8px;flex-shrink:0}
    #probe-panel h3{font-size:10px;text-transform:uppercase;color:#888}
    .pf{display:grid;grid-template-columns:1fr 1fr;gap:6px}
    .pf .full{grid-column:1/-1}
    label{font-size:10px;color:#888;display:block;margin-bottom:2px}
    input,select,textarea{background:#3c3c3c;color:#d4d4d4;border:1px solid #555;padding:4px 7px;font-family:monospace;font-size:12px;width:100%;border-radius:3px}
    input:focus,select:focus,textarea:focus{outline:none;border-color:#0e639c}
    textarea{resize:vertical;min-height:54px}
    #probe-list{max-height:130px;overflow-y:auto;display:flex;flex-direction:column;gap:3px}
    .pi{display:flex;align-items:flex-start;gap:7px;padding:5px 7px;background:#2a2a2a;border-radius:3px;font-size:11px}
    .pi-seg{color:#a080d0;font-weight:bold;min-width:36px;flex-shrink:0}
    .pi-step{color:#569cd6;font-weight:bold;min-width:44px;flex-shrink:0}
    .pi-type{color:#9cdcfe;min-width:68px;flex-shrink:0}
    .pi-q{flex:1;color:#d4d4d4;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
    .pi-gt{color:#4ec9b0;min-width:56px;flex-shrink:0;text-align:right}
    .pi-del{cursor:pointer;color:#666;padding:0 3px;flex-shrink:0}
    .pi-del:hover{color:#f66}
    /* Status bar */
    #status{background:#007acc;color:#fff;padding:2px 12px;font-size:11px;flex-shrink:0}
  </style>
</head>
<body>
<div id="hdr">
  <h1>HalluWorld Trajectory Recorder</h1>
  <span class="meta" id="lbl-level">—</span>
  <span class="meta">seed&nbsp;<b id="lbl-seed">—</b></span>
  <span class="badge">step&nbsp;<span id="lbl-step">0</span></span>
  <span class="badge-seg">seg&nbsp;<span id="lbl-seg">0</span></span>
  <span class="saved-path" id="lbl-path"></span>
</div>

<div id="main">
  <!-- LEFT: grid + action buttons -->
  <div id="left">
    <div id="grid-wrap"><pre id="grid">Loading…</pre></div>
    <div id="ctrl">
      <h3>Actions</h3>
      <div class="btn-row">
        <button onclick="act(2)">▲ Forward (W)</button>
      </div>
      <div class="btn-row">
        <button onclick="act(0)">◄ Turn L (A)</button>
        <button onclick="act(5)">⊕ Toggle (T)</button>
        <button onclick="act(1)">Turn R (D) ►</button>
      </div>
      <div class="btn-row">
        <button onclick="act(3)">↑ Pick Up (G)</button>
        <button onclick="act(4)">↓ Drop (F)</button>
      </div>
      <div class="btn-row" style="margin-top:4px">
        <button onclick="doUndo()" class="btn-danger" style="flex:1">⎌ Undo (Z)</button>
        <button onclick="doSave()" class="btn-success" style="flex:1">💾 Save (Ctrl+S)</button>
      </div>
      <button onclick="focusProbe()" class="btn-primary">＋ Plant Probe (P)</button>
      <button onclick="toggleRoomPanel()" class="btn-room" id="btn-change-room">🚪 Change Room</button>
      <div class="hint">W A D = turn+move &nbsp;·&nbsp; T = toggle &nbsp;·&nbsp; G/F = pick/drop<br>Z = undo &nbsp;·&nbsp; P = probe &nbsp;·&nbsp; Ctrl+S = save</div>
    </div>
  </div>

  <!-- RIGHT: breadcrumb + obs + panels -->
  <div id="right">
    <div id="seg-bar"><!-- filled by JS --></div>
    <div id="zdoors"><!-- zone door hints --></div>
    <div id="obs-wrap"><pre id="obs">Loading…</pre></div>

    <!-- Change room panel -->
    <div id="room-panel">
      <h3>Change Room — new segment</h3>
      <div class="rf">
        <div>
          <label>Level file (relative path)</label>
          <input id="r-file" type="text" placeholder=str(LEVELS_DIR / "h2_kitchen.txt")>
        </div>
        <div>
          <label>Seed</label>
          <input id="r-seed" type="number" value="0" min="0">
        </div>
      </div>
      <div style="margin-top:6px;display:flex;gap:6px">
        <button onclick="doChangeRoom()" class="btn-room" style="flex:1">Confirm Room Change</button>
        <button onclick="toggleRoomPanel()" style="flex:0">Cancel</button>
      </div>
      <div style="margin-top:5px;font-size:10px;color:#6a6a8a">
        Creates a new segment. Actions after this point go into the new room.<br>
        Undo does not cross segment boundaries.
      </div>
    </div>

    <!-- Probe panel -->
    <div id="probe-panel">
      <h3>Plant probe — seg&nbsp;<span id="lbl-probe-seg">0</span> step&nbsp;<span id="lbl-probe-step">0</span>&nbsp;(<span id="lbl-probe-count">0</span> planted)</h3>
      <div class="pf">
        <div>
          <label>Probe Type</label>
          <select id="p-type">
            <option value="presence">presence (yes / no)</option>
            <option value="count">count (integer)</option>
            <option value="attribute">attribute (freeform)</option>
          </select>
        </div>
        <div>
          <label>Ground Truth</label>
          <input id="p-gt" type="text" placeholder="true / false / 3 / red key">
        </div>
        <div class="full">
          <label>Question</label>
          <textarea id="p-q" placeholder="Ask a question about the current observation…"></textarea>
        </div>
        <div class="full">
          <button onclick="doPlant()" class="btn-primary" style="width:100%">Plant Probe (Enter when focused)</button>
        </div>
      </div>
      <h3 style="margin-top:2px">Planted probes</h3>
      <div id="probe-list"></div>
    </div>
  </div>
</div>

<div id="status">Ready — use WASD to navigate, P to plant a probe, Ctrl+S to save.</div>

<script>
let _seg = 0;
let _step = 0;

function status(msg, color) {
  const s = document.getElementById('status');
  s.textContent = msg;
  s.style.background = color || '#007acc';
}

async function fetchState() {
  const r = await fetch('/state');
  const d = await r.json();
  apply(d);
  return d;
}

function apply(d) {
  _seg  = d.segment;
  _step = d.step;
  document.getElementById('lbl-step').textContent = d.step;
  document.getElementById('lbl-seg').textContent  = d.segment + '/' + (d.n_segments - 1);
  document.getElementById('lbl-probe-step').textContent = d.step;
  document.getElementById('lbl-probe-seg').textContent  = d.segment;
  document.getElementById('lbl-level').textContent = d.level_file.replace('levels/', '');
  document.getElementById('lbl-seed').textContent  = d.seed;
  document.getElementById('lbl-path').textContent  = d.out_path ? '→ ' + d.out_path : '';
  document.getElementById('grid').textContent = d.grid;
  document.getElementById('obs').textContent  = d.obs;
  renderBreadcrumb(d.segment_summaries, d.segment);
  renderZoneDoors(d.zone_doors);
  renderProbes(d.probes);
}

function renderBreadcrumb(summaries, active) {
  const bar = document.getElementById('seg-bar');
  bar.innerHTML = '';
  summaries.forEach((s, i) => {
    if (i > 0) {
      const arr = document.createElement('span');
      arr.className = 'seg-arrow';
      arr.textContent = '→';
      bar.appendChild(arr);
    }
    const crumb = document.createElement('span');
    crumb.className = 'seg-crumb' + (i === active ? ' active' : '');
    const name = s.level_file.split('/').pop().replace('.txt','');
    crumb.textContent = `[${i}] ${name} (${s.n_actions}a)`;
    crumb.title = s.level_file + ' seed=' + s.seed;
    bar.appendChild(crumb);
  });
}

function renderZoneDoors(doors) {
  const el = document.getElementById('zdoors');
  if (!doors || doors.length === 0) { el.style.display = 'none'; return; }
  el.style.display = 'block';
  el.textContent = '🚪 Zone doors: ' + doors.map(d =>
    `${d.zone_name} (${d.direction}) → ${d.file_path}`
  ).join('  |  ');
}

function renderProbes(probes) {
  document.getElementById('lbl-probe-count').textContent = probes.length;
  const el = document.getElementById('probe-list');
  el.innerHTML = '';
  probes.forEach((p, i) => {
    const div = document.createElement('div');
    div.className = 'pi';
    div.innerHTML =
      `<span class="pi-seg">s${p.segment}</span>` +
      `<span class="pi-step">step&nbsp;${p.step}</span>` +
      `<span class="pi-type">${p.probe_type}</span>` +
      `<span class="pi-q" title="${escHtml(p.question)}">${escHtml(p.question)}</span>` +
      `<span class="pi-gt">${escHtml(String(p.ground_truth))}</span>` +
      `<span class="pi-del" onclick="delProbe(${i})">✕</span>`;
    el.appendChild(div);
  });
}

function escHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

async function act(a) {
  status('…');
  const r = await fetch('/action', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({action:a})});
  const d = await r.json();
  if (d.error) { status(d.error, '#8b2222'); return; }
  apply(d);
  status(`Seg ${d.segment}  Step ${d.step}`, '#007acc');
}

async function doUndo() {
  const r = await fetch('/undo', {method:'POST'});
  const d = await r.json();
  apply(d);
  const warn = d.warning ? `  ⚠ ${d.warning}` : '';
  status(`Undone → step ${d.step}${warn}`, '#6a6a00');
}

async function doSave() {
  const r = await fetch('/save', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({})});
  const d = await r.json();
  if (d.ok) {
    document.getElementById('lbl-path').textContent = '→ ' + d.path;
    status(`Saved: ${d.n_segments} seg / ${d.n_actions} actions / ${d.n_probes} probes → ${d.path}`, '#2d7a2d');
  } else {
    status('Save failed: ' + (d.error || '?'), '#8b2222');
  }
}

function toggleRoomPanel() {
  const p = document.getElementById('room-panel');
  p.classList.toggle('open');
  if (p.classList.contains('open')) document.getElementById('r-file').focus();
}

async function doChangeRoom() {
  const f = document.getElementById('r-file').value.trim();
  const s = parseInt(document.getElementById('r-seed').value) || 0;
  if (!f) { status('Level file is required', '#8b2222'); return; }
  const r = await fetch('/change_room', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({level_file: f, seed: s})
  });
  const d = await r.json();
  if (d.error) { status('Error: ' + d.error, '#8b2222'); return; }
  document.getElementById('room-panel').classList.remove('open');
  document.getElementById('r-file').value = '';
  apply(d);
  status(`Room changed → segment ${d.segment}: ${f}`, '#5a3e8a');
}

async function doPlant() {
  const qt = document.getElementById('p-type').value;
  const qq = document.getElementById('p-q').value.trim();
  const qg = document.getElementById('p-gt').value.trim();
  if (!qq) { status('Question is required', '#8b2222'); return; }
  const r = await fetch('/probe', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body:JSON.stringify({probe_type:qt, question:qq, ground_truth:qg})
  });
  const d = await r.json();
  if (d.ok) {
    document.getElementById('p-q').value = '';
    document.getElementById('p-gt').value = '';
    status(`Probe planted at seg ${d.probe.segment} step ${d.probe.step}`, '#2d7a2d');
    fetchState();
  } else {
    status('Error: ' + d.error, '#8b2222');
  }
}

async function delProbe(idx) {
  await fetch('/delete_probe', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({index:idx})});
  fetchState();
}

function focusProbe() {
  document.getElementById('p-q').focus();
}

// Keyboard shortcuts — only active when not typing in a form field
document.addEventListener('keydown', e => {
  const tag = document.activeElement.tagName.toLowerCase();
  const inForm = (tag === 'input' || tag === 'textarea' || tag === 'select');

  // Ctrl+S always saves
  if (e.ctrlKey && e.key === 's') { e.preventDefault(); doSave(); return; }

  // Enter in the question textarea plants the probe
  if (inForm && e.key === 'Enter' && tag === 'textarea' && !e.shiftKey) {
    e.preventDefault(); doPlant(); return;
  }

  if (inForm) {
    // Enter in the room file input confirms the room change
    if (e.key === 'Enter') { doChangeRoom(); return; }
    if (e.key === 'Escape') document.activeElement.blur();
    return;
  }

  switch (e.key) {
    case 'w': case 'ArrowUp':    e.preventDefault(); act(2); break;
    case 'a': case 'ArrowLeft':  e.preventDefault(); act(0); break;
    case 'd': case 'ArrowRight': e.preventDefault(); act(1); break;
    case 't': case 'e':          e.preventDefault(); act(5); break;
    case 'g':                    e.preventDefault(); act(3); break;
    case 'f':                    e.preventDefault(); act(4); break;
    case 'z':                    e.preventDefault(); doUndo(); break;
    case 'p':                    e.preventDefault(); focusProbe(); break;
  }
});

fetchState();
</script>
</body>
</html>"""


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive HalluWorld trajectory recorder."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--level", metavar="PATH",
                      help="Level .txt file to load (new trajectory).")
    mode.add_argument("--resume", metavar="PATH",
                      help="Trajectory JSON to resume / extend.")
    parser.add_argument("--seed",  type=int, default=0,
                        help="Episode seed (only used with --level, default 0).")
    parser.add_argument("--out",   metavar="PATH", default="",
                        help="Output JSON path (auto-generated if omitted).")
    parser.add_argument("--port",  type=int, default=5050,
                        help="Port to serve on (default 5050).")
    args = parser.parse_args()

    if args.resume:
        resume_path = args.resume
        print(f"Resuming from {resume_path} …")
        with open(resume_path) as f:
            traj = json.load(f)

        # Support old single-room format: {level_file, seed, actions, probes}
        if "segments" not in traj:
            print("  (upgrading single-room format to segments format)")
            traj = {
                "segments": [{
                    "level_file": traj["level_file"],
                    "seed":       traj["seed"],
                    "actions":    traj.get("actions", []),
                }],
                "probes": [
                    # Old probes had no "segment" key — assign them to segment 0
                    {"segment": 0, **{k: v for k, v in p.items() if k != "segment"}}
                    for p in traj.get("probes", [])
                ],
            }

        _state["segments"]        = traj["segments"]
        _state["probes"]          = traj.get("probes", [])
        _state["current_segment"] = len(_state["segments"]) - 1
        _state["out_path"]        = args.out or resume_path

        seg = _state["segments"][_state["current_segment"]]
        n_actions = sum(len(s["actions"]) for s in _state["segments"])
        print(f"  Segments : {len(_state['segments'])}")
        print(f"  Actions  : {n_actions} total")
        print(f"  Probes   : {len(_state['probes'])}")
        print(f"  Resuming in segment {_state['current_segment']}: {seg['level_file']} seed={seg['seed']}")
        if seg["actions"]:
            print(f"  Replaying {len(seg['actions'])} actions in current segment …")
    else:
        _state["segments"] = [{"level_file": args.level, "seed": args.seed, "actions": []}]
        _state["current_segment"] = 0
        _state["out_path"] = args.out
        seg = _state["segments"][0]
        print(f"  Level  : {seg['level_file']}")
        print(f"  Seed   : {seg['seed']}")

    seg = _state["segments"][_state["current_segment"]]
    _state["env"] = _build_env(seg["level_file"], seg["seed"], seg["actions"])

    print(f"\nOpen http://localhost:{args.port} in your browser.\n")
    app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()
