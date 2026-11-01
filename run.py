"""DSC 180 D32 checkpoint: every figure and table, one target each.

    python run.py samples        parts 1–3: unconditional, conditional and CFG sample grids
    python run.py trajectories   parts 1 and 3: trajectory figures (unconditional, w = 1, w = 4) and their metrics
    python run.py fid2d          part 4: FID in two dimensions, and the classifier on real test images
    python run.py sweep          part 5: label accuracy, FID and FID_class for w = 0, 1, 1.5, 2, 3, 4
    python run.py subsets        part 6: train on 100 and 1,000 images with label dropout; snapshots, FID,
                                 nearest training images, CFG grids
    python run.py mnist          MNIST CFG network: train, then accuracy and FID for w = 0, 1, 2, 4
    python run.py edm-export     full run: write CIFAR-10 in EDM's format and print the train.py command
    python run.py edm-analyze --run-dir training-runs/<run>
                                 full run: loss curve and FID across its snapshots
    python run.py all            samples, trajectories, fid2d, sweep, subsets and mnist, in that order

Figures go to figures/, tables to results/ (CSV, plus Markdown to paste into the report). Downloads go to
downloads/ and generated images, snapshots and caches to outputs/, so a rerun reuses them.
Without an NVIDIA GPU every target runs in QUICK mode, cut down to test the code; its numbers mean nothing.
"""
import argparse
import os
import time

ALL = ["samples", "trajectories", "fid2d", "sweep", "subsets", "mnist"]
TARGETS = ALL + ["edm-export", "edm-analyze", "all"]
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("target", choices=TARGETS, help="what to make (see above)")
    parser.add_argument("--quick", action="store_true", help="cut every step down to test the code (the default "
                                                             "without an NVIDIA GPU)")
    parser.add_argument("--full", action="store_true", help="full size even without an NVIDIA GPU (slow)")
    parser.add_argument("--device", help="cuda, cuda:1, mps or cpu (default: the best available)")
    parser.add_argument("--cls", default="cat", choices=CLASSES, help="trajectories: the class requested in part 3 "
                                                                       "(default cat)")
    parser.add_argument("--seed", type=int, default=0, help="trajectories: the seed of the starting noise (default 0, "
                                                            "as in part 1)")
    parser.add_argument("--mnist-steps", type=int, help="mnist: training updates (default 30,000; the course's "
                                                        "generator was trained for 96,000)")
    parser.add_argument("--run-dir", nargs="+", help="edm-analyze: the run folder that EDM's train.py writes, e.g. "
                                                     "training-runs/00000-cifar10-32x32-n50000-uncond-ddpmpp-edm-...; "
                                                     "for a resumed run, its folders in order")
    args = parser.parse_args()
    if args.quick and args.full:
        parser.error("choose --quick or --full, not both")
    if args.target == "edm-analyze" and not args.run_dir:
        parser.error("edm-analyze needs --run-dir")

    # These settings are read when the modules below are imported.
    if args.quick or args.full:
        os.environ["LAB_QUICK"] = "1" if args.quick else "0"
    if args.device:
        os.environ["LAB_DEVICE"] = args.device
    if args.mnist_steps:
        os.environ["LAB_MNIST_STEPS"] = str(args.mnist_steps)

    import matplotlib
    matplotlib.use("Agg")                       # figures are saved to files, never shown
    from src.common import DEVICE, QUICK
    print(f"device {DEVICE}, QUICK = {QUICK}" + ("  (test run: the numbers mean nothing)" if QUICK else ""),
          flush=True)

    def run(target):
        t0 = time.time()
        print(f"\n===== {target}", flush=True)
        if target == "samples":
            from src.cifar_parts import samples
            samples()
        elif target == "trajectories":
            from src.cifar_parts import trajectories
            trajectories(cls=args.cls, seed=args.seed)
        elif target == "fid2d":
            from src.cifar_parts import classifier_check, fid_2d
            fid_2d()
            classifier_check()
        elif target == "sweep":
            from src.cifar_parts import sweep
            sweep()
        elif target == "subsets":
            from src.cifar_train import subsets
            subsets()
        elif target == "mnist":
            from src import mnist_cfg
            mnist_cfg.run()
        elif target == "edm-export":
            from src.cifar_train import edm_export
            edm_export()
        elif target == "edm-analyze":
            from src.cifar_train import edm_analyze
            edm_analyze(args.run_dir)
        print(f"===== {target} done in {(time.time() - t0) / 60:.1f} min", flush=True)

    for target in ALL if args.target == "all" else [args.target]:
        run(target)


if __name__ == "__main__":
    main()
