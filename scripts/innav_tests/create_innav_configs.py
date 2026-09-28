"""Create companion innav config files for each world.

Each world gets a .innav.json file with:
- Static probe location (where static benchmark asks probes)
- InNav start position (if different from world default)
- Navigation target locations (for sufficiency checking)
- Navigation parameters (max_steps, distance threshold, etc.)
"""

import json
from pathlib import Path
from halluworld.data import LEVELS_DIR


def create_innav_config(
    level_name: str,
    static_probe_location: tuple,
    innav_start_position: tuple = None,
    navigation_targets: list = None,
    max_steps: int = 50,
    sufficiency_distance: float = 4.0,
    notes: str = ""
):
    """Create innav config for a level."""

    config = {
        "level_name": level_name,
        "static_probe_location": list(static_probe_location),
        "innav_start_position": list(innav_start_position) if innav_start_position else None,
        "navigation_target_locations": [list(t) for t in navigation_targets] if navigation_targets else [list(static_probe_location)],
        "navigation_params": {
            "max_steps": max_steps,
            "min_steps": 10,
            "sufficiency_distance": sufficiency_distance,
            "min_distance_from_start": 5.0
        },
        "notes": notes
    }

    return config


def main():
    print("="*70)
    print("CREATE INNAV COMPANION CONFIGS")
    print("="*70)
    print()

    # Load static probe locations
    with open('static_probe_locations.json', 'r') as f:
        static_locations = json.load(f)

    configs = []

    # Corridor Gauntlet
    configs.append(create_innav_config(
        level_name="corridor_gauntlet",
        static_probe_location=(1, 2),
        innav_start_position=None,  # Use world default
        navigation_targets=[
            (1, 2),   # Static location
            (5, 2),   # Quarter way
            (10, 2),  # Halfway
            (15, 2),  # Three-quarters
        ],
        max_steps=50,
        sufficiency_distance=4.0,
        notes="Corridor world - agent navigates east along row 2"
    ))

    # Dense Array
    configs.append(create_innav_config(
        level_name="dense_array",
        static_probe_location=(5, 13),
        innav_start_position=None,  # Use world default
        navigation_targets=[
            (5, 13),  # Static location
            (7, 11),  # Northeast
            (9, 9),   # Further northeast
        ],
        max_steps=50,
        sufficiency_distance=4.0,
        notes="Dense array - agent navigates through object field"
    ))

    # Rotation Challenge
    configs.append(create_innav_config(
        level_name="rotation_challenge",
        static_probe_location=(6, 9),
        innav_start_position=None,  # Use world default
        navigation_targets=[
            (6, 9),   # Static location
            (8, 7),   # Northeast
            (10, 5),  # Further northeast
        ],
        max_steps=50,
        sufficiency_distance=4.0,
        notes="Rotation challenge - complex navigation with turns"
    ))

    # C2 Fire Crossing
    configs.append(create_innav_config(
        level_name="c2_fire_crossing",
        static_probe_location=(7, 11),
        innav_start_position=None,  # Use world default
        navigation_targets=[
            (7, 11),  # Static location
            (7, 8),   # North (past door)
            (7, 5),   # Further north (through river/fire)
        ],
        max_steps=100,  # Complex world needs more steps
        sufficiency_distance=4.0,
        notes="Fire crossing - dynamic obstacles (fire, river)"
    ))

    # C3 Flood Room
    configs.append(create_innav_config(
        level_name="c3_flood_room",
        static_probe_location=(8, 14),
        innav_start_position=None,  # Use world default
        navigation_targets=[
            (8, 14),  # Static location
            (8, 11),  # North
            (8, 8),   # Further north (through flood)
        ],
        max_steps=100,  # Complex world needs more steps
        sufficiency_distance=4.0,
        notes="Flood room - dynamic obstacles (flooding)"
    ))

    # Save each config to companion file
    for config in configs:
        level_name = config['level_name']
        output_file = fstr(LEVELS_DIR / "{level_name}.innav.json")

        with open(output_file, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"✅ Created: {output_file}")
        print(f"   Static location: {config['static_probe_location']}")
        print(f"   Targets: {len(config['navigation_target_locations'])} locations")
        print(f"   Max steps: {config['navigation_params']['max_steps']}")
        print()

    print("="*70)
    print(f"Created {len(configs)} innav companion configs")
    print("="*70)
    print()
    print("Each world now has:")
    print("  • level_name.txt         - Original world file (static benchmark)")
    print("  • level_name.innav.json - InNav config (navigation metadata)")
    print()


if __name__ == "__main__":
    main()
