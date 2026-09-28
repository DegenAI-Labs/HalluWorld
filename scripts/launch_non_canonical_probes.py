#!/usr/bin/env python3
"""Launch non-canonical probes for all 31 worlds × 13 models = 403 probes."""

import subprocess
import time

WORLDS_NON_CANONICAL = [
    # Memory canonical → grid non-canonical
    ('C1a_noboard', 'grid', 4),
    ('C1a_persistent_chain', 'grid', 5),
    ('C1b_continuous_chain', 'grid', 5),
    ('C1b_noboard', 'grid', 5),
    ('C2_fire_crossing', 'grid', 5),
    ('C3_flood_room', 'grid', 5),
    ('C4_forking_paths', 'grid', 5),
    ('C5a_adversarial_board', 'grid', 4),
    ('C6_flood_fire_escape', 'grid', 5),
    ('M1_river_field', 'grid', 5),
    ('M4_narrator', 'grid', 5),
    ('U1_fog_of_war', 'grid', 5),
    ('U2_oracle_high', 'grid', 5),
    ('U2_oracle_low', 'grid', 5),
    ('U2_oracle_mid', 'grid', 5),
    ('U4_amnesiac', 'grid', 5),

    # Symbolic canonical → grid non-canonical
    ('M2_witness_stand', 'grid', 5),
    ('M3_incident_report', 'grid', 5),
    ('P4b_delta_perception', 'grid', 5),
    ('X1_facility_3zone', 'grid', 5),
    ('X2_facility_5zone', 'grid', 5),
    ('X3_facility_7zone', 'grid', 5),
    ('X4_compound_witness', 'grid', 5),
    ('X5_facility_tour', 'grid', 5),
    ('X6_return_visit', 'grid', 5),
    ('X7_dragon_keep', 'grid', 5),

    # Grid canonical → symbolic non-canonical
    ('P1_dense_array', 'symbolic', 5),
    ('P2_corridor_gauntlet', 'symbolic', 6),
    ('P3_rotation_challenge', 'symbolic', 5),
    ('P4_harder_array', 'symbolic', 5),
    ('P5_object_permanence', 'symbolic', 5),
]

MODELS = [
    'claude-opus-4-6',
    'claude-sonnet-4-6',
    'deepseek-ai/DeepSeek-V3-0324',
    'gpt-4o',
    'gpt-4o-mini',
    'gpt-5.4',
    'gpt-5.4-mini',
    'gpt-5.5',
    'moonshotai/Kimi-K2.6',
    'o3',
    'o3-mini',
    'o4-mini',
    'zai-org/GLM-5',
]

# Model groups for rate limiting
RATE_LIMIT_GROUPS = {
    'openai': ['gpt-4o', 'gpt-4o-mini', 'gpt-5.4', 'gpt-5.4-mini', 'gpt-5.5', 'o3', 'o3-mini', 'o4-mini'],
    'anthropic': ['claude-opus-4-6', 'claude-sonnet-4-6'],
    'other': ['deepseek-ai/DeepSeek-V3-0324', 'moonshotai/Kimi-K2.6', 'zai-org/GLM-5'],
}

print("=" * 80)
print("LAUNCHING NON-CANONICAL PROBES")
print(f"31 worlds × 13 models = 403 probes")
print("=" * 80)
print()

launched = 0

# Launch by group to manage rate limits
for group, models in RATE_LIMIT_GROUPS.items():
    batch_size = 4 if group == 'openai' else 6

    print(f"\n{'='*80}")
    print(f"{group.upper()} GROUP ({len(models)} models)")
    print(f"{'='*80}")

    for model in models:
        model_safe = model.replace('/', '-').replace('.', '')

        print(f"\nLaunching {model}:")
        batch_count = 0

        for world, serializer, trace_count in WORLDS_NON_CANONICAL:
            print(f"  {world} ({serializer}, {trace_count} traces)")

            cmd = [
                'python', 'run_innav_eval.py',
                '--models', model,
                '--levels', world,
                '--serializer', serializer,
                '--episodes', str(trace_count),
                '--seed', '999',
                '--trace-dir', 'navigation_traces',
                '--navigation-model', 'gpt-5.4-mini',
                '--output', f'probe_{world}_{serializer}_{model_safe}.csv'
            ]

            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            launched += 1
            batch_count += 1
            time.sleep(0.3)

            # Pause between batches
            if batch_count % batch_size == 0 and batch_count < len(WORLDS_NON_CANONICAL):
                time.sleep(2)

        print(f"  → Launched {len(WORLDS_NON_CANONICAL)} probes for {model}")
        time.sleep(1)

print()
print("=" * 80)
print(f"✅ Launched {launched} non-canonical probes in background")
print("=" * 80)
print()
print("Probes running. Check progress with:")
print("  ps aux | grep run_innav_eval.py | wc -l")
