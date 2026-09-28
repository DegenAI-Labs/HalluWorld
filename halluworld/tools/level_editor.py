#!/usr/bin/env python3
"""HalluWorld Level Editor — terminal-based ASCII grid editor.

Creates and edits .txt level files compatible with AsciiEnv.

Usage:
    python halluworld/tools/level_editor.py              # new empty grid
    python halluworld/tools/level_editor.py levels/my_level.txt

Controls:
    Arrow keys      Move cursor
    # or w          Place wall  (see below for wind)
    Space or .      Place floor
    a               Place agent start
    g or *          Place goal
    k               Place key   (current color)
    d               Place door  (current color)
    b               Place ball  (current color)
    x               Place box   (current color)
    K / D / B / X   Set object type without placing
    1-6             Set color: 1=red 2=green 3=blue 4=yellow 5=purple 6=grey
    W               Toggle wind on cursor column (prompts for row offset)
    s               Save   (prompts for filename if new)
    S               Save as (always prompts)
    l               Load
    n               New grid (prompts for dimensions)
    r               Resize current grid (preserves content, pads/trims borders)
    q               Quit
    ?               Show help overlay
"""

import curses
import sys
import os
from pathlib import Path

# Make the repo root importable when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from halluworld.tracks.grid.envs.ascii_env import (
    LevelSpec,
    load_level,
    CHAR_TO_COLOR,
)

# ── constants ─────────────────────────────────────────────────────────────────

COLOR_CYCLE = list("rgbype")  # in 1-6 key order
COLOR_NAMES: dict[str, str] = {
    "r": "red", "g": "green", "b": "blue",
    "y": "yellow", "p": "purple", "e": "grey",
}
OBJ_NAMES: dict[str, str] = {"K": "key", "D": "door", "B": "ball", "X": "box"}

# Cell width in display columns (cell chars + trailing space)
CELL_W = 3


# ── grid helpers ──────────────────────────────────────────────────────────────

def make_empty_grid(rows: int, cols: int) -> list[list[str]]:
    """Wall border, empty floor inside, goal pre-placed at bottom-right interior."""
    grid = []
    for r in range(rows):
        row = []
        for c in range(cols):
            row.append("##" if (r == 0 or r == rows - 1 or c == 0 or c == cols - 1) else "..")
        grid.append(row)
    # Default goal at bottom-right interior cell
    grid[rows - 2][cols - 2] = "**"
    return grid


def grid_to_text(grid: list[list[str]], metadata: dict[str, str]) -> str:
    lines: list[str] = []
    for k, v in metadata.items():
        lines.append(f"{k}: {v}")
    if metadata:
        lines.append("")
    for row in grid:
        lines.append(" ".join(row))
    return "\n".join(lines)


# ── editor ────────────────────────────────────────────────────────────────────

class LevelEditor:
    def __init__(
        self,
        stdscr: "curses._CursesWindow",
        grid: list[list[str]] | None = None,
        filename: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> None:
        self.stdscr = stdscr
        self.grid = grid or make_empty_grid(9, 10)
        self.filename = filename
        self.metadata = metadata or {
            "agent_view_size": "7",
            "see_through_walls": "false",
            "mission": "navigate the gridworld and reach the goal",
        }
        self.cur_r = 1
        self.cur_c = 1
        self.color = "r"
        self.obj = "K"
        self.modified = False
        self.flash: str = ""
        # wind: col → row-offset (parsed from / synced to metadata["wind"])
        self._wind: dict[int, int] = self._parse_wind_meta()

        self._init_colors()

    # ── setup ──────────────────────────────────────────────────────────────

    # ── wind helpers ───────────────────────────────────────────────────────

    def _parse_wind_meta(self) -> dict[int, int]:
        raw = self.metadata.get("wind", "").strip()
        if not raw:
            return {}
        result: dict[int, int] = {}
        for part in raw.split(","):
            part = part.strip()
            if ":" in part:
                try:
                    col_s, off_s = part.split(":", 1)
                    result[int(col_s)] = int(off_s)
                except ValueError:
                    pass
        return result

    def _sync_wind_meta(self) -> None:
        if self._wind:
            self.metadata["wind"] = ",".join(
                f"{col}:{off}" for col, off in sorted(self._wind.items())
            )
        else:
            self.metadata.pop("wind", None)

    def _init_colors(self) -> None:
        curses.start_color()
        curses.use_default_colors()
        # pairs indexed by color char index (1-6) → 1..6
        _curses_colors = [
            curses.COLOR_RED, curses.COLOR_GREEN, curses.COLOR_BLUE,
            curses.COLOR_YELLOW, curses.COLOR_MAGENTA, curses.COLOR_WHITE,
        ]
        for i, cc in enumerate(_curses_colors, start=1):
            curses.init_pair(i, cc, -1)
        curses.init_pair(7, curses.COLOR_CYAN, -1)    # agent
        curses.init_pair(8, curses.COLOR_WHITE, -1)   # wall / default

    def _cell_attr(self, cell: str) -> int:
        if cell == "##":
            return curses.color_pair(8) | curses.A_BOLD
        if cell == "..":
            return curses.color_pair(8) | curses.A_DIM
        if cell[0] == "A":
            return curses.color_pair(7) | curses.A_BOLD
        if cell == "**":
            return curses.color_pair(4) | curses.A_BOLD       # yellow
        if len(cell) == 2 and cell[1] in "KDBX":
            idx = COLOR_CYCLE.index(cell[0]) + 1 if cell[0] in COLOR_CYCLE else 8
            return curses.color_pair(idx) | curses.A_BOLD
        return curses.color_pair(8)

    # ── properties ─────────────────────────────────────────────────────────

    @property
    def rows(self) -> int:
        return len(self.grid)

    @property
    def cols(self) -> int:
        return len(self.grid[0]) if self.grid else 0

    # ── drawing ────────────────────────────────────────────────────────────

    def draw(self) -> None:
        self.stdscr.erase()
        h, w = self.stdscr.getmaxyx()

        # ── title bar ──
        title = f" HalluWorld Level Editor  {self.cols}×{self.rows}"
        if self.filename:
            title += f"  [{self.filename}]"
        if self.modified:
            title += "  *"
        self._put(0, 0, title[: w - 1], curses.A_REVERSE)

        # ── column header ──
        header_chars: list[str] = []
        for c in range(self.cols):
            col_label = str(c)
            if c in self._wind:
                marker = "v" if self._wind[c] > 0 else "^"
                col_label = marker + col_label
            header_chars.append(f"{col_label:<{CELL_W}}")
        header = "     " + "".join(header_chars)
        self._put(1, 0, header[: w - 1], curses.color_pair(7) | curses.A_DIM)

        # ── grid rows ──
        grid_top = 2
        for r in range(self.rows):
            scr_r = grid_top + r
            if scr_r >= h - 5:
                break
            self._put(scr_r, 0, f"{r:<4}", curses.A_DIM)
            for c in range(self.cols):
                cell = self.grid[r][c]
                x = 4 + c * CELL_W
                if x + CELL_W >= w:
                    break
                attr = self._cell_attr(cell)
                if r == self.cur_r and c == self.cur_c:
                    attr |= curses.A_REVERSE
                self._put(scr_r, x, cell, attr)

        # ── status bar ──
        sep_row = h - 5
        color_name = COLOR_NAMES.get(self.color, self.color)
        obj_name = OBJ_NAMES.get(self.obj, self.obj)
        cur_cell = self.grid[self.cur_r][self.cur_c] if self._in_bounds(self.cur_r, self.cur_c) else ".."
        mission = self.metadata.get("mission", "")
        mission_preview = mission if len(mission) <= 40 else mission[:37] + "..."
        status = (
            f"  Color: {self.color}={color_name}   "
            f"Object: {self.obj}={obj_name}   "
            f"Cursor: ({self.cur_c}, {self.cur_r})   "
            f"Cell: {cur_cell}"
        )
        wind_summary = ""
        if self._wind:
            parts = ", ".join(
                f"col{c}:{off:+d}" for c, off in sorted(self._wind.items())
            )
            wind_summary = f"   wind: {parts}"
        hint1 = "  #=wall  .=floor  a=agent  g=goal  k/d/b/x=place obj  1-6=color  K/D/B/X=type  W=wind-col"
        hint2 = f"  s=save  S=save-as  l=load  n=new  r=resize  m=mission  q=quit  ?=help   mission: {mission_preview}{wind_summary}"

        try:
            self.stdscr.addstr(sep_row, 0, "─" * (w - 1), curses.A_DIM)
            if self.flash:
                self._put(sep_row + 1, 0, f"  {self.flash}"[: w - 1], curses.color_pair(4) | curses.A_BOLD)
                self.flash = ""
            else:
                self._put(sep_row + 1, 0, status[: w - 1], curses.A_BOLD)
            self._put(sep_row + 2, 0, hint1[: w - 1], curses.A_DIM)
            self._put(sep_row + 3, 0, hint2[: w - 1], curses.A_DIM)
        except curses.error:
            pass

        self.stdscr.refresh()

    def _put(self, r: int, c: int, text: str, attr: int = 0) -> None:
        h, w = self.stdscr.getmaxyx()
        if r < 0 or r >= h or c < 0:
            return
        text = text[: w - c - 1]  # guard against writing to last col
        try:
            self.stdscr.addstr(r, c, text, attr)
        except curses.error:
            pass

    # ── in-bounds ──────────────────────────────────────────────────────────

    def _in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.rows and 0 <= c < self.cols

    # ── tile placement ─────────────────────────────────────────────────────

    def place(self, cell: str) -> None:
        if not self._in_bounds(self.cur_r, self.cur_c):
            return
        # Ensure only one agent
        if cell[0] == "A":
            for r in range(self.rows):
                for c in range(self.cols):
                    if self.grid[r][c][0] == "A":
                        self.grid[r][c] = ".."
        self.grid[self.cur_r][self.cur_c] = cell
        self.modified = True

    # ── file I/O ───────────────────────────────────────────────────────────

    def prompt(self, msg: str) -> str:
        h, w = self.stdscr.getmaxyx()
        curses.echo()
        curses.curs_set(1)
        self._put(h - 1, 0, msg[: w - 1], curses.A_REVERSE)
        self.stdscr.clrtoeol()
        self.stdscr.refresh()
        try:
            result = self.stdscr.getstr(h - 1, min(len(msg), w - 2), 80).decode("utf-8").strip()
        except Exception:
            result = ""
        curses.noecho()
        curses.curs_set(0)
        return result

    def save(self, always_prompt: bool = False) -> None:
        filepath = self.filename
        if filepath is None or always_prompt:
            filepath = self.prompt("Save to: ")
        if not filepath:
            self.flash = "Save cancelled."
            return
        try:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            Path(filepath).write_text(grid_to_text(self.grid, self.metadata))
            self.filename = filepath
            self.modified = False
            self.flash = f"Saved → {filepath}"
        except Exception as e:
            self.flash = f"Save error: {e}"

    def load(self, filepath: str | None = None) -> None:
        if filepath is None:
            filepath = self.prompt("Load file: ")
        if not filepath:
            return
        try:
            spec = load_level(filepath)
            self.grid = [row[:] for row in spec.cells]
            self.filename = filepath
            self.metadata = dict(spec.metadata)
            self._wind = self._parse_wind_meta()
            self.modified = False
            self.cur_r = min(self.cur_r, self.rows - 1)
            self.cur_c = min(self.cur_c, self.cols - 1)
            self.flash = f"Loaded: {filepath}"
        except Exception as e:
            self.flash = f"Load error: {e}"

    def new_grid(self) -> None:
        if self.modified:
            ans = self.prompt("Unsaved changes. Continue? (y/n): ").lower()
            if ans != "y":
                return
        dims = self.prompt("Dimensions as 'rows cols' (e.g. 9 10): ")
        try:
            parts = dims.split()
            rows, cols = int(parts[0]), int(parts[1])
            if rows < 3 or cols < 3:
                raise ValueError("Min size is 3×3")
            self.grid = make_empty_grid(rows, cols)
            self.cur_r, self.cur_c = 1, 1
            self.filename = None
            self.modified = False
            self.flash = f"New {cols}×{rows} grid."
        except Exception as e:
            self.flash = f"Bad dimensions: {e}"

    def edit_mission(self) -> None:
        current = self.metadata.get("mission", "")
        val = self.prompt(f"Mission [{current}]: ")
        if val:
            self.metadata["mission"] = val
            self.modified = True
            self.flash = f"Mission set: {val[:50]}"
        else:
            self.flash = "Mission unchanged."

    def toggle_wind_column(self) -> None:
        col = self.cur_c
        if col in self._wind:
            del self._wind[col]
            self._sync_wind_meta()
            self.modified = True
            self.flash = f"Wind removed from col {col}."
            return
        raw = self.prompt(f"Wind offset for col {col} (e.g. -1 = north, +1 = south) [-1]: ")
        if raw == "":
            offset = -1
        else:
            try:
                offset = int(raw)
            except ValueError:
                self.flash = f"Invalid offset: {raw!r}"
                return
        if offset == 0:
            self.flash = "Offset 0 has no effect — skipped."
            return
        self._wind[col] = offset
        self._sync_wind_meta()
        self.modified = True
        direction = "north" if offset < 0 else "south"
        self.flash = f"Wind col {col}: {offset:+d} ({direction})."

    def resize_grid(self) -> None:
        dims = self.prompt(f"New dimensions as 'rows cols' (current: {self.rows} {self.cols}): ")
        try:
            parts = dims.split()
            new_rows, new_cols = int(parts[0]), int(parts[1])
            if new_rows < 3 or new_cols < 3:
                raise ValueError("Min size is 3×3")

            # Build new grid: copy existing content, pad with '..' or trim
            new_grid: list[list[str]] = []
            for r in range(new_rows):
                row: list[str] = []
                for c in range(new_cols):
                    on_border = (r == 0 or r == new_rows - 1 or c == 0 or c == new_cols - 1)
                    if r < self.rows and c < self.cols:
                        row.append(self.grid[r][c])
                    elif on_border:
                        row.append("##")
                    else:
                        row.append("..")
                new_grid.append(row)

            # Re-wall the new border (don't leave old border tiles floating)
            for c in range(new_cols):
                new_grid[0][c] = "##"
                new_grid[new_rows - 1][c] = "##"
            for r in range(new_rows):
                new_grid[r][0] = "##"
                new_grid[r][new_cols - 1] = "##"

            old = f"{self.cols}×{self.rows}"
            self.grid = new_grid
            self.cur_r = min(self.cur_r, new_rows - 1)
            self.cur_c = min(self.cur_c, new_cols - 1)
            self.modified = True
            self.flash = f"Resized {old} → {new_cols}×{new_rows}"
        except Exception as e:
            self.flash = f"Bad dimensions: {e}"

    # ── help overlay ───────────────────────────────────────────────────────

    def show_help(self) -> None:
        lines = [
            "─── HalluWorld Level Editor ───",
            "",
            "Navigation:  Arrow keys",
            "",
            "Place tiles:",
            "  # or w     Wall (##)",
            "  Space or . Floor (..)",
            "  a          Agent start (A.)",
            "  g or *     Goal (**)",
            "  k          Key   (current color)",
            "  d          Door  (current color, locked if matching key exists)",
            "  b          Ball  (current color)",
            "  x          Box   (current color)",
            "",
            "Set type without placing (uppercase):",
            "  K  D  B  X",
            "",
            "Set color:",
            "  1=red  2=green  3=blue  4=yellow  5=purple  6=grey",
            "",
            "Files:",
            "  s   Save       S   Save as",
            "  l   Load       n   New grid   r   Resize   m   Edit mission",
            "",
            "Wind dynamics:",
            "  W   Toggle wind on cursor column (prompts for row offset)",
            "      Offset: -1 = north (row--), +1 = south (row++)",
            "      Wind columns shown as ^N or vN in column header",
            "      Saved as 'wind: col:offset,...' in level metadata",
            "",
            "  q   Quit   ?   This help",
            "",
            "Cell format: 2 chars, e.g. rK=red key, bD=blue door",
            "",
            "Press any key to close...",
        ]
        h, w = self.stdscr.getmaxyx()
        bw = min(max(len(l) for l in lines) + 4, w - 2)
        bh = min(len(lines) + 2, h - 2)
        sr = max(0, h // 2 - bh // 2)
        sc = max(0, w // 2 - bw // 2)

        for i, line in enumerate(lines):
            r = sr + 1 + i
            if r >= h - 1:
                break
            try:
                self.stdscr.addstr(r, sc + 1, line[: bw - 2].ljust(bw - 2), curses.A_NORMAL)
            except curses.error:
                pass
        self.stdscr.refresh()
        self.stdscr.getch()

    # ── main loop ──────────────────────────────────────────────────────────

    def run(self) -> None:
        curses.curs_set(0)
        curses.noecho()

        while True:
            self.draw()
            key = self.stdscr.getch()

            # Navigation
            if key == curses.KEY_UP:
                self.cur_r = max(0, self.cur_r - 1)
            elif key == curses.KEY_DOWN:
                self.cur_r = min(self.rows - 1, self.cur_r + 1)
            elif key == curses.KEY_LEFT:
                self.cur_c = max(0, self.cur_c - 1)
            elif key == curses.KEY_RIGHT:
                self.cur_c = min(self.cols - 1, self.cur_c + 1)

            # Color selection 1-6
            elif ord("1") <= key <= ord("6"):
                self.color = COLOR_CYCLE[key - ord("1")]
                self.flash = f"Color → {self.color} ({COLOR_NAMES[self.color]})"

            # Set object type (uppercase, no placement)
            elif key in [ord(c) for c in "KDBX"]:
                self.obj = chr(key)
                self.flash = f"Object type → {self.obj} ({OBJ_NAMES[self.obj]})"

            # Tile placement
            elif key in (ord("#"), ord("w")):
                self.place("##")
            elif key in (ord(" "), ord(".")):
                self.place("..")
            elif key == ord("a"):
                self.place("A.")
            elif key in (ord("g"), ord("*")):
                self.place("**")
            elif key in [ord(c) for c in "kdbx"]:
                type_map = {"k": "K", "d": "D", "b": "B", "x": "X"}
                self.obj = type_map[chr(key)]
                self.place(f"{self.color}{self.obj}")

            # File ops
            elif key == ord("s"):
                self.save()
            elif key == ord("S"):
                self.save(always_prompt=True)
            elif key == ord("l"):
                self.load()
            elif key == ord("n"):
                self.new_grid()
            elif key == ord("r"):
                self.resize_grid()
            elif key == ord("m"):
                self.edit_mission()
            elif key == ord("W"):
                self.toggle_wind_column()

            # Quit
            elif key == ord("q"):
                if self.modified:
                    ans = self.prompt("Save before quit? (y/n/c=cancel): ").lower()
                    if ans == "y":
                        self.save()
                        break
                    elif ans == "n":
                        break
                    # 'c' = stay
                else:
                    break

            elif key == ord("?"):
                self.show_help()


# ── entry point ───────────────────────────────────────────────────────────────

def _main(stdscr: "curses._CursesWindow", args: list[str], initial_grid: list[list[str]] | None = None) -> None:
    editor = LevelEditor(stdscr, grid=initial_grid)
    if args:
        editor.load(args[0])
    editor.run()


if __name__ == "__main__":
    # Only prompt for size when starting a new level (no file argument given)
    initial_grid = None
    if not sys.argv[1:]:
        try:
            raw = input("Grid size as 'rows cols' [default 10 10]: ").strip()
            if raw:
                parts = raw.split()
                rows, cols = int(parts[0]), int(parts[1])
                if rows < 3 or cols < 3:
                    raise ValueError("Min size is 3x3")
            else:
                rows, cols = 10, 10
            initial_grid = make_empty_grid(rows, cols)
            print(f"Starting {cols}×{rows} grid...")
        except (ValueError, IndexError) as e:
            print(f"Invalid input ({e}), using default 10×10.")
            initial_grid = make_empty_grid(10, 10)
        except (EOFError, KeyboardInterrupt):
            sys.exit(0)

    curses.wrapper(_main, sys.argv[1:], initial_grid)
