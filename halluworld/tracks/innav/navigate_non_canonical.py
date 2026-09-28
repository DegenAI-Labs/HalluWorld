#!/usr/bin/env python3
"""Navigate non-canonical serializations for 16 worlds."""

import subprocess
import time

WORLDS_NEED_GRID = [
    'C1a_noboard',
    'C1a_persistent_chain',
    'C1b_continuous_chain',
    'C1b_noboard',
    'C2_fire_crossing',
    'C3_flood_room',
    'C5a_adversarial_board',
    'C6_flood_fire_escape',
    'M1_river_field',
    'M4_narrator',
    'U1_fog_of_war',
    'U2_oracle_high',
    'U2_oracle_low',
    'U2_oracle_mid',
    'U4_amnesiac',
]

# C4_forking_paths also needs it but might hit boulder bug - try anyway
WORLDS_NEED_GRID.append('C4_forking_paths')

SEEDS = [999, 1000, 1001, 1002, 1003]

print("=" * 80)
print("NAVIGATING NON-CANONICAL (GRID) FOR 16 WORLDS")
print("=" * 80)
print()

launched = 0

# Launch in batches to avoid overwhelming
batch_size = 8  # 8 worlds at a time

for i in range(0, len(WORLDS_NEED_GRID), batch_size):
    batch = WORLDS_NEED_GRID[i:i+batch_size]

    print(f"Batch {i//batch_size + 1}/{(len(WORLDS_NEED_GRID) + batch_size - 1)//batch_size}:")

    for world in batch:
        for seed in SEEDS:
            print(f"  Launching {world} seed {seed} (grid)...")

            cmd = [
                'python', 'run_innav_eval.py',
                '--models', 'gpt-5.4-mini',
                '--levels', world,
                '--serializer', 'grid',
                '--episodes', '1',
                '--seed', str(seed),
                '--prenavigate',
                '--navigation-model', 'gpt-5.4-mini',
                '--trace-dir', 'navigation_traces',
                '--output', f'nav_{world}_grid_seed{seed}.csv'
            ]

            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            launched += 1
            time.sleep(0.2)

    print(f"  Launched {len(batch) * len(SEEDS)} navigation jobs")
    print(f"  Waiting 3s before next batch...")
    time.sleep(3)

print()
print("=" * 80)
print(f"✅ Launched {launched} navigation jobs total")
print("=" * 80)
print()
print("Navigation running in background.")
print("Check progress with: ps aux | grep prenavigate | wc -l")
