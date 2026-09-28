#!/bin/bash
# Smoke test for innav evaluation
# DO NOT assume success from completion - inspect traces manually!

set -e  # Exit on error

echo "=== InNav Mode Smoke Test ==="
echo ""

# 1. Activation check
if [[ "$VIRTUAL_ENV" != *"py_halluworld_benchmark"* ]]; then
    echo "❌ ERROR: Virtual environment not activated!"
    echo "Run: source ~/.pyenv/versions/py_halluworld_benchmark/bin/activate"
    exit 1
fi

echo "✅ Virtual environment: $VIRTUAL_ENV"
echo ""

# 2. Run smoke test (1 episode, 3 probes)
echo "Running smoke test: 1 episode, P1_dense_array, 3 probe timesteps"
echo "Command: python run_innav_eval.py --models gpt-4o-mini --levels P1_dense_array --episodes 1 --seed 42 --probe-timesteps 3 --max-steps 50 --output smoke_ego.csv"
echo ""

python run_innav_eval.py \
  --models gpt-4o-mini \
  --levels P1_dense_array \
  --episodes 1 \
  --seed 42 \
  --probe-timesteps 3 \
  --max-steps 50 \
  --output smoke_ego.csv

echo ""
echo "=== Test completed (but READ TRACES before assuming success!) ==="
echo ""

# 3. Quick inspection (NO pandas, just Unix tools)
echo "--- CSV Structure ---"
head -n 1 smoke_ego.csv
echo ""

echo "--- Row Count ---"
tail -n +2 smoke_ego.csv | wc -l
echo ""

echo "--- Sample Rows ---"
head -n 5 smoke_ego.csv | column -t -s','
echo ""

echo "--- Accuracy Summary (manual calc) ---"
echo "InNav scores:"
tail -n +2 smoke_ego.csv | cut -d',' -f11 | awk '{sum+=$1; n++} END {if (n>0) print "  Total:", sum, "/ Count:", n, "→ Avg:", sum/n}'

echo "Static scores:"
tail -n +2 smoke_ego.csv | cut -d',' -f13 | awk '{sum+=$1; n++} END {if (n>0) print "  Total:", sum, "/ Count:", n, "→ Avg:", sum/n}'
echo ""

echo "=== NEXT: Manual Trace Inspection ==="
echo ""
echo "1. Check episode navigation makes sense:"
echo "   cat smoke_ego.csv | cut -d',' -f1,2,3,4,5,6"
echo ""
echo "2. Check action responses (what did agent say?):"
echo "   # Open CSV in less/vim and look at action_response column"
echo ""
echo "3. Check probe responses:"
echo "   cat smoke_ego.csv | cut -d',' -f8,9,10,11,12,13 | column -t -s','"
echo "   Columns: probe_type | ground_truth | innav_response | innav_score | static_response | static_score"
echo ""
echo "4. Verify probe framing was clear:"
echo "   # Look for 'You have been navigating' in innav_response context"
echo "   # Static should NOT have navigation context"
echo ""
echo "5. Check for ActionParseError:"
echo "   grep -i 'error' smoke_ego.csv"
echo ""

echo "✅ Smoke test script complete!"
echo "⚠️  Remember: Passing != Working. Inspect traces manually!"
