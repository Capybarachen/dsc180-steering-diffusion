# Notes: DSC 180 D32, Q1 checkpoint

Before running each part I write my prediction; after it, what happened and which figure or table shows it. The
command behind every figure and table is in the [README](README.md).

## Run log

| Date | Where (GPU) | Command | Time | Notes |
|---|---|---|---|---|
| | | | | |

## Reference values

- **EDM's released conditional network has `label_dropout = 0`** (`python run.py samples` prints it). It was never
  trained without its label, so it has no "no label" input, and CFG in parts 3 and 5 takes D(x; σ) from the separate
  unconditional network: CFG with two networks.
- **The MNIST lab's frozen classifier** reads 99.32% of the 10,000 MNIST test digits correctly, with pixels scaled to
  [-1, 1] as in the lab (the `test_acc` stored in `evaluator.pt`, reproduced exactly by an independent check).
- **FID of the 10,000 real MNIST test digits** against the 60,000 training digits, on that classifier's features:
  5.6 (computed independently before the GPU runs; 6.7–7.5 with 5,000 digits, 17–22 with 1,000). The "real test
  digits" row of `results/mnist_cfg.md` should match it.

## Part 1: unconditional samples, and inside the sampler

**Before running:** what will D(x; σ) look like at σ = 80? (In code session 2, the MNIST denoiser's guess at high noise
was an average digit.) At which noise level do I expect the content of the final image to be settled?

- Prediction:
- What happened (`figures/p1_uncond_samples.png`, `figures/p1_trajectory_uncond.png`):

## Part 2: conditional samples

**Before running:** what do I expect to stay the same down a column?

- Prediction:
- What happened (`figures/p2_cond_samples.png`):

## Part 3: CFG with two networks

**Before running:** what should change as w goes from 1 to 4?

- Prediction:
- What happened (`figures/p3_cfg_cat.png`):

### Trajectories with a condition

Class: cat; seed 0, the same starting noise as part 1's trajectory. Figures: `figures/p3_trajectory_cat_w1.png` and
`figures/p3_trajectory_cat_w4.png` (8 noise levels), and `*_all_levels.png` (all 18). Numbers:
`figures/p3_trajectory_metrics.png` and `results/p3_trajectory_metrics.md`, over 100 images with balanced requests,
where w = 0 (the unconditional network) is the baseline.

1. **At which noise level does the class first show in D(x; σ)? Is it earlier with w = 4?**
   - Where to look: the first σ at which `p_requested` for w = 1 and w = 4 rises clearly above the w = 0 baseline
     (about 0.1), and the first column where the estimate looks like a cat. The classifier was trained on clean
     images, so its reading of the blurry estimates at high noise is only indicative.
   - Answer:
2. **From which noise level on does the image barely change?**
   - Where to look: `rms_to_final`, the distance from D(x; σ) to the final image, and the last columns of the figures.
   - Answer:
3. **How do the w = 4 estimates differ from the w = 1 ones at low noise?**
   - Where to look: the last columns of the two figures, and `out_of_range`, the share of the estimate's pixels
     outside [-1, 1], which are clipped when the image is saved.
   - Answer:

## Part 4: label accuracy and FID

- Classifier accuracy on 10,000 real test images (`results/p4_classifier_check.json`):
- **Before running** the 2D figure: which panel has the largest FID, and which term causes it?
  - Prediction:
  - What happened (`figures/p4_fid_2d.png`):

## Part 5: the CFG sweep, 1,000 images per w

**Before running:** a sketch of how I expect accuracy and FID to move as w grows.

- Prediction:
- Table (paste `results/p5_cfg_sweep_n1000.md`):
- What happened (`figures/p5_cfg_sweep.png`, `figures/p5_sweep_samples.png`):
- **Which w would I choose, and what does each measure miss?** (Look at the images, not only the numbers.)

## Part 6: EDM's network trained on 100 and on 1,000 images, with CFG

Setup: EDM's `EDMPrecond` around a 7.0M-parameter SongUNet (64 channels, [1, 2, 2], 2 blocks), EDM's loss, Adam
with learning rate 2e-4, batch 64, 20,000 updates per training set, EMA 0.999 (0.9 for the first 1,000 updates),
label dropout 0.1. Five snapshots per run, each measured with FID on 1,000 unconditional samples.

- **Loss and FID across snapshots** (`figures/p6_loss_fid.png`, `results/p6_fid_snapshots.md`): what does FID do
  across the snapshots, and why?
- **Is it memorization?** (`figures/p6_nearest_n100.png`, `figures/p6_nearest_n1000.png`, `results/p6_nearest.md`):
  how close are the samples to their nearest training image, compared with real test images?
- **CFG with one network.** Before running: does w > 1 raise label accuracy here? Compare with part 5.
  - Prediction:
  - What happened (`figures/p6_cfg_n100.png`, `figures/p6_cfg_n1000.png`, `results/p6_cfg_accuracy.md`):

## MNIST CFG network

Setup: the lab's U-Net (64 channels, 1.9M parameters) trained from scratch on the 60,000 MNIST training digits, on the
cosine path, predicting the clean digit; the label replaced by 10 ("no label") for a random 10% of examples. Adam with
learning rate 2e-4, batch 128, 30,000 updates (the lab's released generator had 96,000), EMA 0.999 with the course's
warm-up. Sampling: the lab's sampler, 16 DDIM steps, u + w (c − u). Measured on 10,000 samples per w (1,000 per
digit), with the same starting noise for every w and for both networks.

- **Before running:** how will label accuracy and FID move as w goes 0 → 1 → 2 → 4?
  - Prediction:
- Table (paste `results/mnist_cfg.md`):
- What happened (`figures/mnist_cfg_sweep.png`, `figures/mnist_cfg_samples.png`, `figures/mnist_course_samples.png`):
- How does our network compare with the lab's generator?

## Encouraged: the full EDM run

- GPU, hours, million training images reached:
- Loss and FID across snapshots (`figures/full_run_loss_fid.png`, `results/full_run_fid.md`):

## For the report (due Friday, November 6)

- [ ] Title and abstract
- [ ] Introduction: the problem, related work, and the data (CIFAR-10, MNIST, and data for any attribute I plan to
      study)
- [ ] Link to this repository
- [ ] Preliminary results: this week's figures and tables, each with a sentence on what it shows

## Questions for Yusu Wang and Qingsong Wang

-
