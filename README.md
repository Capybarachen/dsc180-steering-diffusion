# Steering diffusion models: Q1 checkpoint (DSC 180, D32)

Code, figures and tables for my Q1 checkpoint in DSC 180 D32, *Steering Diffusion Models* (UC San Diego, fall 2026).
It generates CIFAR-10 images with NVIDIA's pretrained EDM models (Karras et al., 2022), steers them with
classifier-free guidance (CFG) and measures them with label accuracy and FID; trains EDM's network with label dropout
on small subsets of CIFAR-10; and trains a CFG network on MNIST.

- **Report:** _link to the report_
- **Notes:** [NOTES.md](NOTES.md): predictions, observations and answers to the questions, part by part

## Quick start

On a machine with an NVIDIA GPU (a DSMLP GPU or Colab):

```bash
git clone https://github.com/Capybarachen/dsc180-steering-diffusion.git
cd <this-repo>
pip install -r requirements.txt     # on DSMLP: pip install --user -r requirements.txt
python run.py samples               # parts 1–3 sample grids; the first run downloads about 720 MB
python run.py all                   # every figure and table below except the encouraged full run
```

Each command writes its figures to `figures/` and its tables to `results/` (a CSV with the raw numbers and a
Markdown table to paste into the report). Downloads go to `downloads/`, and generated images, snapshots and caches
to `outputs/`, so a rerun reuses them. Long steps print an estimate of how long they will take.
`python run.py --help` lists the targets and options.

## Every figure and table, and the command that makes it

**This week: generation and sampling** (course notebook parts 1–5, NVIDIA's pretrained networks)

| Figure or table | File | Command |
|---|---|---|
| 30 unconditional samples, seed 0 (part 1) | `figures/p1_uncond_samples.png` | `python run.py samples` |
| Conditional samples: one class per row, the same 8 starting noises in every row (part 2) | `figures/p2_cond_samples.png` | `python run.py samples` |
| CFG with two networks toward cat, w = 0, 1, 1.5, 2, 3, 4, same noise per column (part 3) | `figures/p3_cfg_cat.png` | `python run.py samples` |
| Trajectory of the unconditional model: noisy image and D(x; σ) at 8 of the 18 noise levels (part 1) | `figures/p1_trajectory_uncond.png` | `python run.py trajectories` |
| Trajectories toward cat for the conditional model (w = 1) and CFG at w = 4, same seed (part 3) | `figures/p3_trajectory_cat_w1.png`, `figures/p3_trajectory_cat_w4.png`, and `*_all_levels.png` with all 18 levels | `python run.py trajectories` |
| Part 3's questions in numbers: classifier's probability of the requested class in D(x; σ), distance from D(x; σ) to the final image, and share of D(x; σ)'s pixels outside [-1, 1], at each noise level for w = 0, 1, 4 (100 images) | `figures/p3_trajectory_metrics.png`, `results/p3_trajectory_metrics.md` | `python run.py trajectories` |
| FID in two dimensions (part 4) | `figures/p4_fid_2d.png` | `python run.py fid2d` |
| Classifier accuracy on 10,000 real test images: the label accuracy of real images, for reference (part 4) | `results/p4_classifier_check.json` | `python run.py fid2d` |
| **The CFG sweep table:** label accuracy, FID and FID_class for w = 0, 1, 1.5, 2, 3, 4 at 1,000 images per w (part 5) | `results/p5_cfg_sweep_n1000.md` (and `.csv`, `.json`) | `python run.py sweep` |
| The CFG sweep plots: accuracy, FID and FID_class against w, and the classes the unconditional model makes (part 5) | `figures/p5_cfg_sweep.png` | `python run.py sweep` |
| The first 10 images for each w, one per class, same noise per column (part 5) | `figures/p5_sweep_samples.png` | `python run.py sweep` |

`python run.py trajectories --cls dog --seed 3` makes part 3's figures for another class or starting noise.

**By the checkpoint: training**

| Figure or table | File | Command |
|---|---|---|
| Training loss, and FID across 5 snapshots, for 100 and 1,000 training images (part 6) | `figures/p6_loss_fid.png`, `results/p6_fid_snapshots.md` | `python run.py subsets` |
| Unconditional samples from the same noise at each snapshot (part 6) | `figures/p6_progress_n100.png`, `figures/p6_progress_n1000.png` | `python run.py subsets` |
| Samples next to their nearest training images, and median distances against real test images (part 6) | `figures/p6_nearest_n100.png`, `figures/p6_nearest_n1000.png`, `results/p6_nearest.md` | `python run.py subsets` |
| CFG grids with one network, w = 0, 1, 2, 4, with label accuracy (part 6) | `figures/p6_cfg_n100.png`, `figures/p6_cfg_n1000.png`, `results/p6_cfg_accuracy.md` | `python run.py subsets` |
| **MNIST CFG network:** label accuracy and FID on the lab classifier's features for w = 0, 1, 2, 4, for our network, the lab's released generator and real test digits | `results/mnist_cfg.md` (and `.csv`, `.json`) | `python run.py mnist` |
| MNIST: training loss, and accuracy and FID against w | `figures/mnist_cfg_sweep.png` | `python run.py mnist` |
| MNIST samples, w = 0, 1, 2, 4, one requested digit per column | `figures/mnist_cfg_samples.png`, `figures/mnist_course_samples.png` | `python run.py mnist` |
| Encouraged: EDM's full network trained on all 50,000 images: loss curve and FID across snapshots | `figures/full_run_loss_fid.png`, `results/full_run_fid.md` | [the full run](#the-encouraged-full-run) |

## Setup

- **Python 3.10–3.13 and PyTorch 2.2 or later with a CUDA GPU.** `requirements.txt` adds torchvision (MNIST),
  pandas and pyarrow (CIFAR-10 from Hugging Face), Pillow, and requests, click and psutil (EDM's code and
  `train.py`).
- **QUICK mode.** As in the course notebook, every target runs in QUICK mode when there is no NVIDIA GPU: each step
  is cut down far enough to test the code, and the numbers mean nothing. `--quick` forces it on a GPU (a fast test of
  the whole pipeline: `python run.py all --quick`), and `--full` turns it off.
- **Other settings** (environment variables): `LAB_DEVICE` (for example `cuda:1`), `LAB_N_EVAL` (images per w in
  part 5, default 1,000), `LAB_STEPS` (updates per training set in part 6, default 20,000), `LAB_MNIST_STEPS`
  (MNIST updates, default 30,000; also `--mnist-steps`), `LAB_MNIST_N_EVAL` (MNIST samples per w, default 10,000),
  and `EDM_DIR` (your own clone of EDM instead of the download).

### On DSMLP

Log in with `ssh <username>@dsmlp-login.ucsd.edu` and start a container with a GPU (for example
`launch-scratch.sh -g 1`; `-h` lists the options). Clone the repository there and install the requirements. The long
targets (`sweep`, `subsets`, `mnist` and the full run) should run in a background container, which keeps running
after you log out (DSMLP's launch scripts have a background option, `-b`; check `-h`). In it, run
`bash scripts/run_all.sh`, which runs every target in order, writes each log to `logs/<target>.log`, and carries on
past a failed target. `kubectl get pods` lists your containers, `kubectl logs -f <pod>` follows one, and
`kubectl delete pod <pod>` releases its GPU when you are done.

### On Colab

Choose a GPU runtime (Runtime → Change runtime type), then in a cell:

```
!git clone https://github.com/Capybarachen/dsc180-steering-diffusion.git
%cd <this-repo>
!pip install -q -r requirements.txt
!python run.py samples
```

Colab deletes the files when the runtime ends, so download `figures/` and `results/` (or commit them) before it
does. `notebooks/cifar10_edm_lab.ipynb` is the course notebook with part 3's cell filled in; it shows the same CIFAR-10
figures inline but does not save them.

## Data and model downloads

Each target downloads what it needs the first time, checks every file against its SHA-256 checksum, and resumes an
interrupted download when you rerun it. The CIFAR-10 targets download about 720 MB in all.

| What | Source | Used by |
|---|---|---|
| EDM's code at commit `008a4e5` | [NVlabs/edm](https://github.com/NVlabs/edm) | every CIFAR-10 target (EDM's pickles need its code), part 6, the full run |
| EDM's pretrained CIFAR-10 networks, unconditional and conditional (`edm-cifar10-32x32-{uncond,cond}-vp.pkl`) | NVIDIA, `nvlabs-fi-cdn.nvidia.com/edm/pretrained` | parts 1–5 |
| EDM's FID reference: Inception statistics of the 50,000 CIFAR-10 training images (`cifar10-32x32.npz`) | NVIDIA, `nvlabs-fi-cdn.nvidia.com/edm/fid-refs` | FID |
| The Inception-v3 feature detector of EDM's `fid.py` (`inception-2015-12-05.pkl`) | NVIDIA NGC (StyleGAN3's metrics) | FID |
| A pretrained CIFAR-10 ResNet-56, which scores label accuracy and never guides sampling | [chenyaofo/pytorch-cifar-models](https://github.com/chenyaofo/pytorch-cifar-models) | label accuracy |
| CIFAR-10, 50,000 training and 10,000 test images | Hugging Face's copy, [`uoft-cs/cifar10`](https://huggingface.co/datasets/uoft-cs/cifar10) at a fixed revision; the original `cifar-10-python.tar.gz` is used instead if it is in `downloads/` | parts 4–6, the full run |
| MNIST, 60,000 training and 10,000 test digits | `torchvision.datasets.MNIST` | `mnist` |
| The MNIST guidance lab's frozen digit classifier (`evaluator.pt`, 1.9 MB) and released generator (`generator.pt`, 7.6 MB) | the course's Google Drive, as in the lab | `mnist` |

If the Google Drive download fails, copy `evaluator.pt` and `generator.pt` from the lab's
`dsc180-d32-mnist-guidance-lab.zip` into `downloads/mnist_lab/`.

**Three small fixes to EDM's code.** The downloaded copy of EDM is edited so that it runs with current Python and
PyTorch (`fix_edm` in `src/edm.py`):
1. `torch_utils/misc.py`: `super().__init__(dataset)` becomes `super().__init__()`, the course notebook's fix for
   PyTorch 2.2 and later.
2. `dnnlib/util.py`: `from distutils.util import strtobool` gets a fallback, because Python 3.12 removed distutils.
   The course notebook's in-process shim cannot reach `train.py`, which runs in its own process.
3. `training/training_loop.py`: the `torch.load` that `--resume` uses gets `weights_only=False`. Since PyTorch 2.6
   `torch.load` refuses the pickled network inside a training state by default, so resuming would fail.

A clone that you point to with `EDM_DIR` is not edited; the code prints the changes instead.

## The encouraged full run

EDM's own `train.py` trains EDM's full CIFAR-10 network (DDPM++, 55.7M parameters, the same architecture as the
pretrained networks) on all 50,000 training images, unconditionally, with the course notebook's settings
(`--batch=512 --batch-gpu=64 --duration=10`, so at most 10 million training images).

```bash
python run.py edm-export          # downloads EDM and writes outputs/cifar10-32x32-n50000.zip, EDM's dataset format
bash scripts/train_edm_full.sh    # run this in a background container: it trains for many hours
python run.py edm-analyze --run-dir training-runs/00000-cifar10-32x32-n50000-uncond-ddpmpp-edm-gpus1-batch512-fp32
```

`train.py` prints its progress every tick (50,000 training images) and writes its loss to `stats.jsonl` in the run
folder. Every 10 ticks it saves `network-snapshot-<kimg>.pkl` (about 0.2 GB) and, with the script's `--dump=10`,
`training-state-<kimg>.pt` (about 0.7 GB), which is what `--resume` needs; EDM's default saves that state only every
500 ticks, so a stopped run could not be resumed. Delete older training-state files to save space. The first snapshot
is the untrained network.

- **Stop it at any time.** To continue, pass the newest training state:
  `bash scripts/train_edm_full.sh --resume="$PWD/training-runs/<run>/training-state-<kimg>.pt"`. A resumed run writes
  a new folder, so give `edm-analyze` both folders in order: `--run-dir training-runs/00000-... training-runs/00001-...`.
- **`edm-analyze` can run while training continues.** It measures FID on 1,000 unconditional samples from each
  snapshot and caches each value in `outputs/`, so a rerun only measures new snapshots.
- EDM's full schedule (200 million images) took about two days on 8 V100 GPUs, so a few hours on one GPU covers a
  small part of it: expect FID to keep falling but to stay above the pretrained network's.

## Reading the numbers

- **FID gets better (lower) with more samples.** With 1,000 samples our FIDs are much higher than published ones (EDM
  reports 1.97 unconditional and 1.79 conditional with 50,000). Compare runs only with each other, at the same number
  of samples and against the same reference. Every table states its number of samples.
- **FID ignores the labels.** It compares all generated images with all 50,000 training images, so conditional and CFG
  runs request balanced classes (image *i* requests class *i* mod 10). **FID_class** compares the images of each
  requested class with the training images of that class, and averages over the classes. With 100 images per class
  it is much higher than the all-class FID; compare it across w, not with the other FID.
- **With 100 or 1,000 training images the model memorizes** (part 6): its samples become copies of training images.
  So part 6 tests the training pipeline; the full run shows real training.
- **MNIST's FID uses the lab classifier's 128 features**, against the 60,000 training digits. Real test digits come
  from other writers, so their FID (it should come out near 5.6 at 10,000 digits) is a reference point, not a floor:
  a model trained on the training digits can score lower.

## Repository layout

```
run.py                     every figure and table, one target each
src/common.py              device, QUICK mode, folders, checked downloads, FID's Gaussian fit and distance, saving
src/edm.py                 EDM: downloads and fixes, models, CIFAR-10, the sampler, D and CFG, the classifier, FID
src/cifar_parts.py         parts 1–5: sample grids, trajectories, FID in 2D, the CFG sweep
src/cifar_train.py         part 6 and the full run: training with label dropout, snapshots, nearest images, analysis
src/mnist_cfg.py           the MNIST CFG network: training and measuring
src/mnist_guidance_lab.py  the MNIST guidance lab's helper code (U-Net, sampler, classifier), unchanged
scripts/run_all.sh         every target in order, with logs: for a background job
scripts/train_edm_full.sh  the full run with EDM's train.py
notebooks/                 the course's CIFAR-10 notebook, with part 3's cell filled in
figures/, results/         the outputs, committed
NOTES.md                   my notes
```

`downloads/`, `outputs/`, `training-runs/` and `logs/` are not committed (see `.gitignore`).

## Credits

- T. Karras, M. Aittala, T. Aila, S. Laine, "Elucidating the Design Space of Diffusion-Based Generative Models",
  NeurIPS 2022 ([arXiv:2206.00364](https://arxiv.org/abs/2206.00364)), and its code and pretrained networks,
  [NVlabs/edm](https://github.com/NVlabs/edm) (CC BY-NC-SA 4.0). This repository downloads EDM's code at run time
  and does not redistribute it.
- J. Ho, T. Salimans, "Classifier-Free Diffusion Guidance", 2022 ([arXiv:2207.12598](https://arxiv.org/abs/2207.12598)).
- The ResNet-56 classifier from [chenyaofo/pytorch-cifar-models](https://github.com/chenyaofo/pytorch-cifar-models).
- The DSC 180 D32 course materials (Yusu Wang, Qingsong Wang): the CIFAR-10 EDM notebook, whose code `src/edm.py`,
  `src/cifar_parts.py` and `src/cifar_train.py` follow closely, and the MNIST guidance lab, whose helper code is
  `src/mnist_guidance_lab.py`.
