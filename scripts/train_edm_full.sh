#!/usr/bin/env bash
# The encouraged full run (end of part 6): EDM's own train.py trains EDM's full CIFAR-10 network (DDPM++, 55.7M
# parameters) on all 50,000 training images, unconditionally, with the course notebook's settings.
#
#   python run.py edm-export            # first: downloads EDM (with its PyTorch 2.2+ fix) and writes the dataset
#   bash scripts/train_edm_full.sh      # then: a new run in training-runs/
#   bash scripts/train_edm_full.sh --resume="$PWD/training-runs/<run>/training-state-<kimg>.pt"   # continue a run
#   python run.py edm-analyze --run-dir training-runs/<run>                                       # loss and FID
#
# Extra arguments go to train.py (the last value of an option wins). Run it as a background job, not in a notebook.
# Every 10 ticks (500,000 images) it saves network-snapshot-<kimg>.pkl (about 0.2 GB) and, with --dump=10 below,
# training-state-<kimg>.pt (about 0.7 GB: the network and Adam's state), which --resume needs. EDM's default dumps
# state only every 500 ticks or at the end, so without --dump a stopped run could not be resumed. Delete older
# training-state files to save space; keep the newest.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EDM_DIR="${EDM_DIR:-$ROOT/downloads/edm-008a4e5316c8e3bfe61a62f874bddba254295afb}"
DATA="$ROOT/outputs/cifar10-32x32-n50000.zip"
if [[ ! -f "$EDM_DIR/train.py" || ! -f "$DATA" ]]; then
  echo "Run 'python run.py edm-export' first: it downloads EDM and writes $DATA" >&2
  exit 1
fi
cd "$EDM_DIR"
exec torchrun --standalone --nproc_per_node=1 train.py --outdir="$ROOT/training-runs" --data="$DATA" \
  --cond=0 --arch=ddpmpp --batch=512 --batch-gpu=64 --duration=10 --snap=10 --dump=10 "$@"
