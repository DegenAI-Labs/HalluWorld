"""The frozen release must remain connected to all four domain runtimes."""

import subprocess
import sys
from pathlib import Path

from halluworld.release_contract import (
    check_release,
    read_released_trajectories,
    released_trajectory_actions,
    selected_probe_timesteps,
)

REPO = Path(__file__).resolve().parent.parent


def test_all_four_release_domains_are_wired(bank_dir):
    reports = {report.domain: report for report in check_release()}

    assert set(reports) == {"grid", "chess", "innav", "terminal"}
    assert (reports["grid"].questions, reports["grid"].levels) == (443, 33)
    assert (reports["chess"].questions, reports["chess"].levels) == (350, 7)
    assert (reports["innav"].questions, reports["innav"].levels) == (117, 33)
    assert reports["innav"].details["trajectories"] == 294
    assert (reports["terminal"].questions, reports["terminal"].levels) == (529, 110)
    assert reports["terminal"].details["truncations"] >= 27


def test_every_released_trajectory_can_reconstruct_selected_probe_states(bank_dir):
    for trace in read_released_trajectories():
        actions = released_trajectory_actions(trace)
        selected = selected_probe_timesteps(trace["steps_taken"])
        assert not selected or max(selected) < len(actions), trace["file"]


def test_trajectory_packer_retains_action_sequence():
    source = (REPO / "scripts" / "build_trajectory_set.py").read_text()
    assert '"action_sequence": trace.get("action_sequence")' in source


def test_release_check_cli_succeeds(bank_dir):
    result = subprocess.run(
        [sys.executable, "-m", "halluworld", "release", "check"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "OK   grid" in result.stdout
    assert "OK   chess" in result.stdout
    assert "OK   innav" in result.stdout
    assert "OK   terminal" in result.stdout


def test_every_terminal_task_is_present_in_retained_checkout(bank_dir):
    from halluworld.questions import read_bank

    tasks = {
        record.task_or_level
        for record in read_bank(bank_dir / "terminal.jsonl.gz")
    }
    retained = REPO / "external" / "terminal-bench" / "original-tasks"
    assert not sorted(task for task in tasks if not (retained / task).is_dir())


def test_terminal_generation_paths_resolve_from_checkout(monkeypatch, tmp_path):
    from halluworld.tracks.terminal import run_tb_task

    assert run_tb_task.REPO_ROOT == REPO
    assert run_tb_task.TB_ROOT == REPO / "external" / "terminal-bench"
    assert run_tb_task.TASKS_DIR.is_dir()
    assert run_tb_task.JUDGE_SCRIPT.is_file()

    trajectory = tmp_path / "trajectory.json"
    trajectory.write_text("{}")
    monkeypatch.setattr(run_tb_task, "build_trajectory", lambda *args: trajectory)
    monkeypatch.setattr(run_tb_task, "runtime_probe_answering_enabled", lambda: False)
    monkeypatch.setattr(run_tb_task, "evaluate_probes", lambda path: (1, 1, 0))
    run_tb_task.postprocess_task(tmp_path, "task", True, "model")
