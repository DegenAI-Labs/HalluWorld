#!/bin/bash
# Quick progress checker for experiments

echo "📊 Experiment 1 Progress Check"
echo "==============================="
echo ""

cd results_varun_exp1 2>/dev/null || mkdir -p results_varun_exp1

for file in gpt4omini_seed*.csv gpt4o_seed*.csv gpt5mini_seed*.csv glm5_seed*.csv; do
  if [ -f "$file" ]; then
    lines=$(wc -l < "$file")
    echo "✅ $file: $lines lines"
  else
    echo "⏳ $file: pending..."
  fi
done

echo ""
echo "Expected: ~240 lines per file (header + ~60 episodes × ~4 probes)"
