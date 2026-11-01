"""The MNIST CFG network: the guidance lab's U-Net, trained from scratch on MNIST's 60,000 training digits with the
label replaced by 10 ("no label") for a random 10% of examples. Then sampled with w = 0, 1, 2, 4 and measured with
the lab's frozen digit classifier: label accuracy, and FID on the classifier's 128 features.

Course convention (the lab's mnist_guidance_lab.py): x0 is noise at t = 0, x1 is a digit in [-1, 1] at t = 1,
x_t = alpha_t x1 + sigma_t x0 on the cosine path (alpha = sin(pi t / 2), sigma = cos(pi t / 2)), and the network
predicts the clean digit x1 from (x_t, t, label). The sampler mixes the two predictions as u + w (c - u).
"""
import copy
import json
import os
import time

import numpy as np
import matplotlib.pyplot as plt
import torch

from .common import DEVICE, DL, OUT, QUICK, fetch, frechet, gaussian, how_long, save_fig, write_json, write_table
from . import mnist_guidance_lab as mg

LAB_FILES = {   # the lab's weights, exactly as its notebook fetches them: file: (address, SHA-256)
    "generator.pt": ("https://drive.google.com/uc?export=download&id=1ET8_47MTKGwFr2hbG7eEyQwfLhqw3ql3",
                     "ec413c1df3fdaf1321059a9fbde2265fa10c6499b6c7e0b9031b218e74f99bb6"),
    "evaluator.pt": ("https://drive.google.com/uc?export=download&id=1RkocEim9ejow8n76WI0LGpNF18K5FxlW",
                     "05e955d1c14856991785c408f9a818b5d887cdc2c5142d63eac134367f50b7fe"),
}
LAB_DIR = DL / "mnist_lab"      # the lab's generator.pt and evaluator.pt (or copy them here from the lab's zip)
MNIST_DIR = DL / "mnist"        # torchvision's copy of MNIST
RUN_DIR = OUT / "mnist_cfg"

W = [0, 1, 2, 4]
STEPS = int(os.environ.get("LAB_MNIST_STEPS", 100 if QUICK else 30_000))   # the course's generator: 96,000
BATCH = 32 if QUICK else 128
LR = 2e-4
EMA = 0.999                     # with the course's warm-up: 0.9 for the first 1,000 updates
LABEL_DROPOUT = 0.1             # the CFG switch: 10% of training labels become 10, "no label"
CH = 64                         # the lab's U-Net: 64 channels, 1.9M parameters
N_EVAL = int(os.environ.get("LAB_MNIST_N_EVAL", 200 if QUICK else 10_000))   # samples per w, balanced over digits
SAMPLE_STEPS = 16               # the lab's sampler: 16 DDIM steps
SEED = 0


# ------------------------------------------------------------------ data and the lab's networks
def load_mnist(train=True):
    """MNIST digits in [-1, 1] with shape (n, 1, 28, 28), and their labels: 60,000 for train, 10,000 for test.
    The scaling is the course's (load_digits in the lab: pixels / 255 * 2 - 1)."""
    from torchvision.datasets import MNIST
    ds = MNIST(str(MNIST_DIR), train=train, download=True)
    return ds.data.float().div(255).mul(2).sub(1)[:, None].contiguous(), ds.targets.long().clone()


def load_evaluator():
    """The lab's frozen digit classifier. It judges samples; it never steers them."""
    fetch({"evaluator.pt": LAB_FILES["evaluator.pt"]}, LAB_DIR)
    ck = torch.load(LAB_DIR / "evaluator.pt", map_location="cpu", weights_only=True)
    evaluator = mg.Evaluator()
    evaluator.load_state_dict(ck["state"])
    return evaluator.to(DEVICE).eval().requires_grad_(False)


def load_unet(path):
    """A U-Net saved as {"state", "ch", ...}: the lab's generator.pt or our own checkpoint."""
    ck = torch.load(path, map_location="cpu", weights_only=True)
    net = mg.UNet(ch=int(ck["ch"]))
    net.load_state_dict(ck["state"])
    return net.to(DEVICE).eval().requires_grad_(False)


def load_course_generator():
    """The lab's released generator (96,000 updates), for comparison; None if it cannot be downloaded."""
    try:
        fetch({"generator.pt": LAB_FILES["generator.pt"]}, LAB_DIR)
    except (OSError, RuntimeError) as e:
        print(f"skipping the course generator ({e}); copy generator.pt from the lab's zip into {LAB_DIR} to "
              f"compare with it", flush=True)
        return None
    return load_unet(LAB_DIR / "generator.pt")


# ------------------------------------------------------------------ training
def train(steps=STEPS, seed=SEED):
    """Train the lab's U-Net from scratch with 10% label dropout. Returns the averaged (EMA) network and the losses.
    A finished run in outputs/mnist_cfg/ is loaded instead of trained again."""
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    ckpt, loss_file = RUN_DIR / f"unet_steps{steps}_seed{seed}.pt", RUN_DIR / f"losses_steps{steps}_seed{seed}.json"
    if ckpt.exists() and loss_file.exists():
        print(f"MNIST: loading the finished run from {ckpt.name}", flush=True)
        return load_unet(ckpt), json.loads(loss_file.read_text())

    x_train, y_train = load_mnist(train=True)
    x_train, y_train = x_train.to(DEVICE), y_train.to(DEVICE)
    torch.manual_seed(seed)
    net = mg.UNet(ch=CH).to(DEVICE).train()
    ema = copy.deepcopy(net).eval().requires_grad_(False)       # averaged weights, used for sampling
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    g = torch.Generator().manual_seed(seed)                     # all randomness drawn on the CPU, then moved
    losses, t0 = [], time.time()
    print(f"MNIST: training the lab's U-Net ({sum(p.numel() for p in net.parameters()):,} parameters) for {steps:,} "
          f"updates of {BATCH} digits", flush=True)
    for step in range(1, steps + 1):
        idx = torch.randint(len(x_train), (BATCH,), generator=g).to(DEVICE)
        x1, y = x_train[idx], y_train[idx]
        y[torch.rand(BATCH, generator=g).to(DEVICE) < LABEL_DROPOUT] = 10      # "no label" for 10% of examples
        t = torch.rand(BATCH, generator=g).to(DEVICE)
        x0 = torch.randn(x1.shape, generator=g).to(DEVICE)
        alpha, sigma = mg.alpha_sigma(t[:, None, None, None])
        loss = ((net(alpha * x1 + sigma * x0, t, y) - x1) ** 2).mean()   # predict the clean digit
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            decay = 0.9 if step <= 1000 else EMA                # the course's EMA, with its warm-up
            for p_ema, p in zip(ema.parameters(), net.parameters()):
                p_ema.lerp_(p, 1 - decay)
        losses.append(loss.item())
        if step == 100:
            print(f"MNIST: {how_long((time.time() - t0) / 100 * steps)} for {steps:,} updates", flush=True)
        if step % 5000 == 0 or step == steps:
            print(f"MNIST update {step:,}: loss {np.mean(losses[-1000:]):.4f}, {how_long(time.time() - t0)} so far",
                  flush=True)
    torch.save({"state": {k: v.detach().cpu() for k, v in ema.state_dict().items()}, "ch": CH, "steps": steps,
                "seed": seed, "ema": EMA, "label_dropout": LABEL_DROPOUT, "batch": BATCH, "lr": LR}, ckpt)
    loss_file.write_text(json.dumps(losses))
    return ema, losses


# ------------------------------------------------------------------ measuring
@torch.no_grad()
def classifier_features(evaluator, x, bs=1000):
    """The frozen classifier's 128 features for each digit in [-1, 1]."""
    return torch.cat([evaluator.features(x[i:i + bs].to(DEVICE)).double().cpu() for i in range(0, len(x), bs)])


def scores(evaluator, images, labels, ref, class_ref, per_class=True):
    """Label accuracy and confidence, FID against all training digits, FID_class (each digit against the training
    digits of that digit, averaged), and the lab's pixel spread (variation within each requested digit)."""
    accuracy, confidence = mg.classification(evaluator, images, labels, DEVICE)
    feats = classifier_features(evaluator, images)
    digits = labels.unique().tolist()
    return {"label accuracy": accuracy, "confidence": confidence,
            "FID": frechet(*gaussian(feats), *ref),
            "FID_class": float(np.mean([frechet(*gaussian(feats[labels == c]), *class_ref[c]) for c in digits]))
            if per_class else None,
            "pixel spread": float(np.mean([images[labels == c].var(dim=0, unbiased=False).mean().item() for c in digits]))}


def sweep(net, evaluator, ref, class_ref, model, n=N_EVAL, seed=SEED):
    """Sample n digits for each w (image i requests digit i mod 10, the same starting noise for every w) and score
    them. Returns the table rows and the first ten samples for each w."""
    labels = torch.arange(n) % 10
    rows, firsts = [], {}
    for w in W:
        images, nfe = mg.sample(net, n, labels, DEVICE, steps=SAMPLE_STEPS, w=float(w), seed=seed, batch=1000)
        row = {"model": model, "w": w, "images": n, "network calls per image": nfe}
        row.update(scores(evaluator, images, labels, ref, class_ref, per_class=w > 0))   # w = 0 requests no digit
        rows.append(row)
        firsts[f"w = {w:g}"] = images[:10]
        print(f"{model}, w = {w:g}: accuracy {row['label accuracy']:.1%}, FID {row['FID']:.2f}", flush=True)
    return rows, firsts


def digit_grid(rows, title, name):
    """rows: {row label: ten digits in [-1, 1]}, column j requesting digit j."""
    fig, axes = plt.subplots(len(rows), 10, figsize=(11.5, 1.25 * len(rows) + 0.6), squeeze=False)
    for r, (label, images) in enumerate(rows.items()):
        for j, ax in enumerate(axes[r]):
            ax.imshow(images[j, 0].numpy(), cmap="gray", vmin=-1, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[r, 0].set_ylabel(label, rotation=0, ha="right", va="center")
    for j in range(10):
        axes[0, j].set_title(f"request {j}", fontsize=9)
    fig.suptitle(title)
    fig.tight_layout()
    save_fig(fig, name)


def run(steps=STEPS):
    """Train (or load) our network, then measure it, the course's generator, and real test digits."""
    net, losses = train(steps)
    evaluator = load_evaluator()
    x_train, y_train = load_mnist(train=True)
    n_ref = 6000 if QUICK else len(x_train)
    print(f"MNIST: classifier features of {n_ref:,} training digits, the FID reference", flush=True)
    train_feats = classifier_features(evaluator, x_train[:n_ref])
    ref = gaussian(train_feats)
    class_ref = [gaussian(train_feats[y_train[:n_ref] == c]) for c in range(10)]

    ours = f"ours, {steps:,} updates"
    rows, firsts = sweep(net, evaluator, ref, class_ref, ours)
    digit_grid(firsts, f"Our MNIST CFG network ({steps:,} updates): the same starting noise in each column",
               "mnist_cfg_samples")
    course_net = load_course_generator()
    if course_net is not None:
        course_rows, course_firsts = sweep(course_net, evaluator, ref, class_ref, "course generator, 96,000 updates")
        rows += course_rows
        digit_grid(course_firsts, "The lab's released generator (96,000 updates): the same starting noise as ours",
                   "mnist_course_samples")

    x_test, y_test = load_mnist(train=False)
    n_test = min(N_EVAL, len(x_test))
    real = {"model": "real test digits", "w": None, "images": n_test, "network calls per image": None}
    real.update(scores(evaluator, x_test[:n_test], y_test[:n_test], ref, class_ref))
    rows.append(real)
    print(f"real test digits: accuracy {real['label accuracy']:.1%}, FID {real['FID']:.2f}", flush=True)

    write_table("mnist_cfg", rows,
                caption=f"MNIST CFG: {N_EVAL:,} samples per w ({N_EVAL // 10:,} per digit), {SAMPLE_STEPS} DDIM steps. "
                        f"Label accuracy and confidence from the lab's frozen classifier; FID on its 128 features "
                        f"against the {n_ref:,} MNIST training digits; FID_class per digit, averaged. Real test digits "
                        f"come from other writers, so they are a reference point, not a floor.",
                formats={"label accuracy": "{:.1%}", "confidence": "{:.3f}", "FID": "{:.2f}", "FID_class": "{:.2f}",
                         "pixel spread": "{:.4f}"})
    write_json("mnist_cfg", {"training": {"updates": steps, "batch": BATCH, "lr": LR, "ema": EMA,
                                          "label_dropout": LABEL_DROPOUT, "channels": CH, "seed": SEED,
                                          "final_loss_mean_of_last_1000": float(np.mean(losses[-1000:]))},
                             "evaluation": {"samples_per_w": N_EVAL, "sampler_steps": SAMPLE_STEPS,
                                            "fid_reference_digits": n_ref},
                             "rows": rows})

    fig, axes = plt.subplots(1, 3, figsize=(16, 3.8))
    k = min(200, len(losses))
    axes[0].plot(range(k - 1, len(losses)), np.convolve(losses, np.ones(k) / k, mode="valid"))
    axes[0].set(yscale="log", xlabel="update", ylabel=f"loss, average of {k} updates", title="Training loss (ours)")
    for model in dict.fromkeys(r["model"] for r in rows if r["w"] is not None):
        sel = [r for r in rows if r["model"] == model]
        axes[1].plot([r["w"] for r in sel], [r["label accuracy"] for r in sel], marker="o", label=model)
        axes[2].plot([r["w"] for r in sel], [r["FID"] for r in sel], marker="o", label=model)
    axes[1].axhline(real["label accuracy"], color="0.5", ls="--", lw=1, label="real test digits")
    axes[2].axhline(real["FID"], color="0.5", ls="--", lw=1, label="real test digits")
    axes[1].set(xlabel="w", ylabel="label accuracy", title="Does the digit match the request?")
    axes[2].set(xlabel="w", ylabel=f"FID ({N_EVAL:,} samples, classifier features)", title="How close to real digits?")
    axes[1].legend()
    axes[2].legend()
    fig.tight_layout()
    save_fig(fig, "mnist_cfg_sweep")
