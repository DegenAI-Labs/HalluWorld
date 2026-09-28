from __future__ import annotations

import argparse
import os

import numpy as np
from PIL import Image

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall
from minigrid.minigrid_env import MiniGridEnv


class SimpleEnv(MiniGridEnv):
    def __init__(
        self,
        size=10,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        max_steps: int | None = None,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "grand mission"

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Generate vertical separation wall
        for i in range(0, height):
            self.grid.set(5, i, Wall())

        # Place the door and key
        self.grid.set(5, 6, Door(COLOR_NAMES[0], is_locked=True))
        self.grid.set(3, 6, Key(COLOR_NAMES[0]))

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "grand mission"


def run_headless(n_steps: int = 20, save_dir: str = "frames"):
    """Run the env headlessly, saving frames as PNGs."""
    os.makedirs(save_dir, exist_ok=True)
    env = SimpleEnv(render_mode="rgb_array")
    obs, _ = env.reset(seed=42)

    frame = env.render()
    Image.fromarray(frame).save(os.path.join(save_dir, "step_000.png"))
    print(f"Saved initial frame → {save_dir}/step_000.png")

    for step in range(1, n_steps + 1):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        frame = env.render()
        path = os.path.join(save_dir, f"step_{step:03d}.png")
        Image.fromarray(frame).save(path)
        print(f"Step {step:3d} | action={action} reward={reward:.2f} | saved {path}")
        if terminated or truncated:
            print("Episode finished, resetting.")
            obs, _ = env.reset()

    env.close()
    print(f"\nAll frames saved to ./{save_dir}/")


def run_human():
    """Run with a GUI window — requires a display (Xvfb or X11 forwarding).

    Server-side virtual display:
        Xvfb :99 -screen 0 1400x900x24 &
        DISPLAY=:99 python test_env.py --mode human

    SSH X11 forwarding (from local machine):
        ssh -X user@server
        python test_env.py --mode human
    """
    from minigrid.manual_control import ManualControl

    env = SimpleEnv(render_mode="human")
    manual_control = ManualControl(env, seed=42)
    manual_control.start()


def main():
    parser = argparse.ArgumentParser(description="SimpleEnv test runner")
    parser.add_argument(
        "--mode",
        choices=["headless", "human"],
        default="headless",
        help="headless: save frames as PNGs (no display needed); "
             "human: open GUI window (requires display / Xvfb / X11 forwarding)",
    )
    parser.add_argument("--steps", type=int, default=20, help="Steps for headless mode")
    parser.add_argument("--save-dir", type=str, default="frames", help="Output dir for frames")
    args = parser.parse_args()

    if args.mode == "human":
        run_human()
    else:
        run_headless(n_steps=args.steps, save_dir=args.save_dir)


if __name__ == "__main__":
    main()