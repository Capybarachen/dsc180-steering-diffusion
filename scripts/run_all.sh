#!/usr/bin/env bash
# Every figure and table of the checkpoint, one target after another, each with its own log in logs/.
# A failed target is reported and the next one still runs. Use it in a background job on DSMLP (see the README).
#
#   bash scripts/run_all.sh                  # samples trajectories fid2d sweep subsets mnist
#   bash scripts/run_all.sh sweep mnist      # only these targets
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs
targets=("$@")
if [[ ${#targets[@]} -eq 0 ]]; then
  targets=(samples trajectories fid2d sweep subsets mnist)
fi
failed=()
for target in "${targets[@]}"; do
  echo "=== $target, started $(date)"
  python run.py "$target" 2>&1 | tee "logs/$target.log"
  status=${PIPESTATUS[0]}
  if [[ $status -ne 0 ]]; then
    echo "!!! $target failed (exit $status): see logs/$target.log"
    failed+=("$target")
  fi
done
echo "=== finished $(date)"
if [[ ${#failed[@]} -gt 0 ]]; then
  echo "failed: ${failed[*]}"
  exit 1
fi
