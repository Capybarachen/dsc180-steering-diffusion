"""Parts 1–5 of the course notebook: the sample grids, the trajectory figures (with part 3's "your turn"), the
classifier and FID, and the CFG sweep. Each function saves its figures in figures/ and its tables in results/."""
import time

import numpy as np
import matplotlib.pyplot as plt
import torch

from .common import OUT, QUICK, frechet, gaussian, how_long, save_fig, write_json, write_table
from . import edm
from .edm import CLASSES, D, D_cfg, edm_sample, show_grid, show_trajectory, to_uint8

W = [0, 1, 1.5, 2, 3, 4]    # part 3's grid and part 5's sweep


# ------------------------------------------------------------------ parts 1–3: sample grids
def samples():
    """Part 1: 30 unconditional samples. Part 2: one class per row from the same noise. Part 3: CFG toward cat."""
    uncond, cond = edm.networks()
    x = edm_sample(lambda x, s: D(uncond, x, s), n=30, seed=0)
    show_grid({f"images {10 * k + 1}–{10 * k + 10}": to_uint8(x[10 * k:10 * k + 10]) for k in range(3)},
              title="unconditional model, seed 0", name="p1_uncond_samples")

    rows = {}
    for c in range(10):
        labels = torch.full((8,), c)
        rows[CLASSES[c]] = to_uint8(edm_sample(lambda x, s: D(cond, x, s, labels), n=8, seed=1))
    show_grid(rows, title="conditional model: one class per row, the same starting noise in each column",
              name="p2_cond_samples")

    source = (edm.EDM_DIR / "training" / "networks.py").read_text().splitlines()
    k = next(i for i, line in enumerate(source) if "self.training and self.label_dropout" in line)
    print("EDM, training/networks.py, inside SongUNet.forward:")
    print("\n".join(source[k - 2:k + 3]))
    print("\nlabel_dropout stored in the released conditional network:", cond.model.label_dropout)

    labels = torch.full((8,), CLASSES.index("cat"))
    show_grid({f"w = {w:g}": to_uint8(edm_sample(lambda x, s: D_cfg(x, s, labels, w), n=8, seed=2)) for w in W},
              title="CFG toward cat: the same starting noise in each column", name="p3_cfg_cat")


# ------------------------------------------------------------------ parts 1 and 3: trajectories
def trajectories(cls="cat", seed=0):
    """Part 1's trajectory figure (unconditional), and part 3's: the conditional model (w = 1) and CFG at w = 4,
    toward one class, from the same seed. Each conditional figure comes twice: at the 8 noise levels of part 1, and
    at all 18. Then the trajectory metrics, which put numbers on part 3's three questions."""
    uncond, _ = edm.networks()
    _, history = edm_sample(lambda x, s: D(uncond, x, s), n=3, seed=seed, record=True)
    show_trajectory(history, title="unconditional model: the noisy image and the denoiser's estimate at each noise level",
                    name="p1_trajectory_uncond")

    labels = torch.full((3,), CLASSES.index(cls))
    for w in (1, 4):
        _, history = edm_sample(lambda x, s: D_cfg(x, s, labels, w), n=3, seed=seed, record=True)
        what = "conditional model (w = 1)" if w == 1 else f"CFG with two networks (w = {w})"
        show_trajectory(history, title=f"{what}, toward {cls}: the noisy image and the denoiser's estimate",
                        name=f"p3_trajectory_{cls}_w{w}")
        show_trajectory(history, steps=tuple(range(len(history))), title=f"{what}, toward {cls}: all 18 noise levels",
                        name=f"p3_trajectory_{cls}_w{w}_all_levels")
    trajectory_metrics(seed=seed)


@torch.no_grad()
def trajectory_metrics(n=None, seed=0, ws=(0, 1, 4)):
    """For n images with balanced requests (image i requests class i mod 10), at each of the 18 noise levels:
    - p_requested: the classifier's probability of the requested class, read from the estimate D(x; σ)
      (question 1: when does the class show?)
    - rms_to_final: the RMS pixel distance from D(x; σ) to the final image (question 2: when does it stop changing?)
    - out_of_range: the share of D(x; σ)'s pixels outside [-1, 1] (question 3: what does w = 4 do at low noise?)
    At w = 0 the request is ignored, so its p_requested is a chance baseline, about 0.1."""
    n = n or (20 if QUICK else 100)
    labels = torch.arange(n) % 10
    rows = []
    for w in ws:
        x_final, history = edm_sample(lambda x, s: D_cfg(x, s, labels, w), n=n, seed=seed, record=True)
        x_final = x_final.cpu()
        for k, (s, _, estimate) in enumerate(history):
            probs = edm.classifier_logits(to_uint8(estimate)).softmax(1)
            rows.append({"w": w, "step": k, "sigma": s,
                         "p_requested": probs[torch.arange(n), labels].mean().item(),
                         "accuracy": (probs.argmax(1) == labels).float().mean().item(),
                         "rms_to_final": (estimate - x_final).pow(2).mean((1, 2, 3)).sqrt().mean().item(),
                         "out_of_range": (estimate.abs() > 1).float().mean().item()})
        print(f"trajectory metrics, w = {w:g}: done", flush=True)
    write_table("p3_trajectory_metrics", rows,
                caption=f"Trajectory metrics: {n} images, image i requests class i mod 10, seed {seed}. "
                        "The classifier reads the estimate D(x; σ) at each noise level.",
                formats={"sigma": "{:.3g}", "p_requested": "{:.3f}", "accuracy": "{:.0%}",
                         "rms_to_final": "{:.3f}", "out_of_range": "{:.2%}"})

    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    panels = [("p_requested", "P(requested class | D(x; σ))", "When does the class show?"),
              ("rms_to_final", "RMS distance from D(x; σ) to the final image", "When does the image stop changing?"),
              ("out_of_range", "share of D(x; σ)'s pixels outside [-1, 1]", "How far does the estimate overshoot?")]
    for ax, (key, ylabel, title) in zip(axes, panels):
        for w in ws:
            sel = [r for r in rows if r["w"] == w]
            ax.plot([r["sigma"] for r in sel], [r[key] for r in sel], marker="o", ms=3,
                    label="w = 0 (unconditional)" if w == 0 else f"w = {w:g}")
        ax.set(xscale="log", xlabel="noise level σ (sampling runs left to right)", ylabel=ylabel, title=title)
        ax.invert_xaxis()
    axes[0].axhline(0.1, color="0.6", lw=0.8, ls="--")
    axes[0].legend()
    fig.suptitle(f"Inside the sampler: {n} images with balanced requests, seed {seed}")
    fig.tight_layout()
    save_fig(fig, "p3_trajectory_metrics")


# ------------------------------------------------------------------ part 4: the classifier and FID
def classifier_check():
    """The classifier's accuracy on real test images: the label accuracy of real images, for reference. (Not a
    ceiling: strong CFG can make images that the classifier reads more easily than real ones.)"""
    test_images, test_labels = edm.load_cifar10("test")
    n = 1000 if QUICK else 10_000
    accuracy = (edm.classify(test_images[:n]) == test_labels[:n]).float().mean().item()
    print(f"classifier accuracy on {n} real test images: {accuracy:.1%}")
    write_json("p4_classifier_check", {"classifier": "ResNet-56 (chenyaofo/pytorch-cifar-models)",
                                       "real_test_images": n, "accuracy": accuracy})


def fid_2d():
    """FID in two dimensions: a "real" cloud of 100,000 points against three "generated" clouds."""
    g = torch.Generator().manual_seed(0)
    A = torch.tensor([[1.0, 0.0], [0.8, 0.5]], dtype=torch.float64)    # gives the cloud its tilted shape

    def cloud(n, shift=(0.0, 0.0), spread=1.0):
        return (torch.randn(n, 2, generator=g, dtype=torch.float64) * spread) @ A.T + torch.tensor(shift, dtype=torch.float64)

    def ellipse(ax, mu, cov, color):
        """Draw the Gaussian's two-standard-deviation ellipse."""
        e, V = torch.linalg.eigh(cov)
        t = torch.linspace(0, 2 * torch.pi, 200, dtype=torch.float64)
        pts = mu + 2 * torch.stack([t.cos(), t.sin()], 1) @ (V * e.clamp(min=0).sqrt()).T
        ax.plot(pts[:, 0], pts[:, 1], color=color, lw=2)

    real = cloud(100_000)
    mu_r, s_r = gaussian(real)
    cases = {"same distribution, 30 points": cloud(30),
             "shifted mean, 5,000 points": cloud(5000, shift=(1.5, 0.0)),
             "too little variety, 5,000 points": cloud(5000, spread=0.4)}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharex=True, sharey=True)
    for ax, (name, pts) in zip(axes, cases.items()):
        mu_g, s_g = gaussian(pts)
        ax.scatter(*real[:3000].T, s=2, color="0.75", label="real")
        ax.scatter(*pts.T, s=14 if len(pts) < 100 else 3, color="tab:orange", label="generated")
        ellipse(ax, mu_r, s_r, "0.3")
        ellipse(ax, mu_g, s_g, "tab:orange")
        total, mean_term = frechet(mu_r, s_r, mu_g, s_g), float(((mu_r - mu_g) ** 2).sum())
        ax.set_title(f"{name}\nFID {total:.2f} = mean term {mean_term:.2f} + spread term {total - mean_term:.2f}",
                     fontsize=10)
    axes[0].legend(loc="upper left")
    save_fig(fig, "p4_fid_2d")


# ------------------------------------------------------------------ part 5: the CFG sweep
def generate(w, n=None):
    """n images with balanced requested classes (image i requests class i mod 10); saved in outputs/ so that a
    rerun reuses them. Image i starts from the same noise for every w."""
    n = n or edm.N_EVAL
    path = OUT / f"cfg_w{w:g}_n{n}.npz"
    if path.exists():
        saved = np.load(path)
        return torch.from_numpy(saved["images"]), torch.from_numpy(saved["labels"])
    labels, images, t0 = torch.arange(n) % 10, [], time.time()
    for b in range(0, n, edm.BATCH):
        lab = labels[b:b + edm.BATCH]
        images.append(to_uint8(edm_sample(lambda x, s: D_cfg(x, s, lab, w), n=len(lab), seed=1000 + b)))
        if b == 0:
            print(f"w = {w:g}: {how_long((time.time() - t0) * n / edm.BATCH)} for {n} images", flush=True)
    images = torch.cat(images)
    np.savez(path, images=images.numpy(), labels=labels.numpy())
    return images, labels


def sweep():
    """Label accuracy, FID (all classes, against EDM's 50,000-image reference) and FID_class (each class against
    the training images of that class, averaged) for w = 0, 1, 1.5, 2, 3, 4."""
    n_eval = edm.N_EVAL
    train_images, train_labels = edm.load_cifar10("train")
    n_ref = 1000 if QUICK else len(train_images)            # real images for FID_class: all 50,000 on a GPU
    print(f"Inception features of {n_ref} training images, for FID_class", flush=True)
    train_features = edm.features(train_images[:n_ref])
    class_ref = [gaussian(train_features[train_labels[:n_ref] == c]) for c in range(10)]

    def fid_class(feats, labels):
        """FID of each class's images against the real training images of that class, averaged over the classes."""
        return float(np.mean([frechet(*gaussian(feats[labels == c]), *class_ref[c]) for c in range(10)]))

    results, firsts = {}, {}
    for w in W:
        images, labels = generate(w)
        predicted, feats = edm.classify(images), edm.features(images)
        results[w] = {"accuracy": (predicted == labels).float().mean().item(),
                      "fid": frechet(*gaussian(feats), *edm.fid_reference()),
                      "fid_class": fid_class(feats, labels) if w > 0 else None}   # w = 0 requests no class
        if w == 0:
            histogram = torch.bincount(predicted, minlength=10)
        firsts[f"w = {w:g}"] = images[:10]                  # one image per requested class, the same noise per column
        per_class = f"{results[w]['fid_class']:7.1f}" if w > 0 else "      –"
        print(f"w = {w:<4g} accuracy {results[w]['accuracy']:6.1%}   FID {results[w]['fid']:7.2f}   "
              f"FID_class {per_class}", flush=True)

    write_json(f"p5_cfg_sweep_n{n_eval}", {
        "n_images": n_eval, "fid_class_real_images_per_class": n_ref // 10,
        "results": {f"{w:g}": r for w, r in results.items()},
        "w0_predicted_class_histogram": dict(zip(CLASSES, histogram.tolist()))})
    write_table(f"p5_cfg_sweep_n{n_eval}",
                [{"w": w, "images": n_eval, "label accuracy": r["accuracy"], "FID": r["fid"],
                  "FID_class": r["fid_class"]} for w, r in results.items()],
                caption=f"CFG with two networks, {n_eval} images per w ({n_eval // 10} per class). FID against all "
                        f"50,000 CIFAR-10 training images; FID_class against the training images of each class.",
                formats={"label accuracy": "{:.1%}", "FID": "{:.2f}", "FID_class": "{:.1f}"})

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes[0, 0].plot(W, [results[w]["accuracy"] for w in W], marker="o")
    axes[0, 0].set(xlabel="w", ylabel="label accuracy", title="Does the image show the requested class?")
    axes[0, 1].plot(W, [results[w]["fid"] for w in W], marker="o")
    axes[0, 1].set(xlabel="w", ylabel=f"FID ({n_eval} images, all classes)", title="How close to real images?")
    axes[1, 0].plot(W[1:], [results[w]["fid_class"] for w in W[1:]], marker="o", color="tab:green")
    axes[1, 0].set(xlabel="w", ylabel=f"FID_class ({n_eval // 10} images per class)",
                   title="How close to real images of the requested class?")
    axes[1, 1].bar(CLASSES, histogram.numpy())
    axes[1, 1].set(title="w = 0: classes the unconditional model makes", ylabel="images")
    axes[1, 1].tick_params(axis="x", rotation=60)
    fig.tight_layout()
    save_fig(fig, "p5_cfg_sweep")
    show_grid(firsts, title="The CFG sweep: images 1–10 for each w (image i requests class i mod 10: airplane, "
                            "automobile, …, truck)", name="p5_sweep_samples")
