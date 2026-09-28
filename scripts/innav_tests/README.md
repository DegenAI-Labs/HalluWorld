# Egocentric Test Scripts

One-time test scripts used during egocentric mode development.
These are NOT part of the core workflow - just for development/debugging.

## Core Workflow (Use These)

For actual egocentric evaluation, use:
- `../../run_egocentric_eval.py` - Main entrypoint for egocentric evaluation
- `../../navigate_until_sufficient.py` - Navigation with sufficiency logic (imported by egocentric.py)

## Development/Test Scripts (This Folder)

These were used during development and are kept for reference:

**Action Parsing Tests:**
- `test_action_parsing.py` - Tests shared action parsing utility
- `test_llm_fallback.py` - Tests GPT-4o-mini fallback for action parsing

**Trace Workflow Tests:**
- `test_save_load_workflow.py` - End-to-end trace save/load test
- `test_trace_workflow.py` - Trace workflow validation

**Navigation Tests:**
- `test_navigation_grid.py` - Grid-based navigation testing
- `test_all_worlds_navigation.py` - Multi-world navigation test
- `test_world_by_world.py` - Individual world testing
- `diagnose_navigation.py` - Navigation diagnostic tool

**Setup Scripts (One-time use):**
- `prenavigate_and_save_traces.py` - Superseded by `--prenavigate` flag
- `find_valid_probe_windows.py` - Sufficiency validator development
- `extract_static_probe_locations.py` - Extract probe locations from levels
- `create_egocentric_configs.py` - Generate .egocentric.json configs (already done)
