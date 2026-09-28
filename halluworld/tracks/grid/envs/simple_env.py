from __future__ import annotations

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Ball, Box, Door, Goal, Key, Wall
from minigrid.minigrid_env import MiniGridEnv

# MiniGrid color palette
_COLORS = COLOR_NAMES  # ['blue', 'green', 'grey', 'purple', 'red', 'yellow']

# Distractor object types beyond the key and ball
_DISTRACTOR_TYPES = [Ball, Box]


class SimpleEnv(MiniGridEnv):
    """Two-room gridworld with procedurally generated object colors and positions.

    Layout on every reset():
      - Left room (interior x: 1..wall_x-1) contains the key, ball, distractors, agent
      - Vertical wall at x=wall_x with a locked door at a random y position
      - Right room contains the goal
      - Everything except wall position is randomised each episode

    Config params (vary these for experiments):
        agent_view_size: int  — FOV width/height (7, 5, or 3 for V-axis experiments)
        num_objects: int      — extra distractor objects beyond key+ball
        see_through_walls: bool
    """

    def __init__(
        self,
        size: int = 10,
        agent_view_size: int = 7,
        num_objects: int = 0,
        see_through_walls: bool = False,
        max_steps: int | None = None,
        **kwargs,
    ):
        self.num_objects = num_objects
        mission_space = MissionSpace(mission_func=self._gen_mission)
        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            agent_view_size=agent_view_size,
            see_through_walls=see_through_walls,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission() -> str:
        return "pick up the key, open the door, and reach the goal"

    def _gen_grid(self, width: int, height: int) -> None:
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # Dividing wall — fixed at x = width//2
        wall_x = width // 2
        for i in range(height):
            self.grid.set(wall_x, i, Wall())

        # Random door position on the dividing wall (avoid corners)
        door_y = self.np_random.integers(2, height - 2)
        key_color = self.np_random.choice(_COLORS)
        self.grid.set(wall_x, door_y, Door(key_color, is_locked=True))

        # Goal in the right room
        self.place_obj(Goal(), top=(wall_x + 1, 1), size=(width - wall_x - 2, height - 2))

        # Objects in the left room
        left_room = dict(top=(1, 1), size=(wall_x - 2, height - 2))

        # Key (matches door color)
        self.place_obj(Key(key_color), **left_room)

        # Ball with a different color
        ball_color = self.np_random.choice([c for c in _COLORS if c != key_color])
        self.place_obj(Ball(ball_color), **left_room)

        # Extra distractor objects
        distractor_colors = [c for c in _COLORS if c != key_color]
        for _ in range(self.num_objects):
            cls = _DISTRACTOR_TYPES[int(self.np_random.integers(0, len(_DISTRACTOR_TYPES)))]
            color = self.np_random.choice(distractor_colors)
            self.place_obj(cls(color), **left_room)

        # Agent placed randomly in the left room
        self.place_agent(top=(1, 1), size=(wall_x - 2, height - 2))

        self.mission = self._gen_mission()


def make_env(**kwargs) -> SimpleEnv:
    """Convenience constructor — pass any SimpleEnv __init__ kwargs."""
    return SimpleEnv(**kwargs)
