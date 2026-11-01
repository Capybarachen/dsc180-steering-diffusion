"""Part 6 of the course notebook: train EDM's network (a 7.0M-parameter version) for CFG, with 10% label dropout,
on 100 and on 1,000 CIFAR-10 images; measure saved snapshots with FID; compare samples with their nearest training
images; and test CFG with this one network. Then the encouraged full run with EDM's own train.py: write the dataset,
and plot the run's loss and FID across its snapshots."""
import copy
import io
import json
import os
import pickle
import time
import zipfile
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from .common import DEVICE, OUT, QUICK, ROOT, how_long, save_fig, write_table
from . import edm
from .edm import D, edm_sample, show_grid, to_uint8

SIZES = [100, 1000]          # the two training sets, balanced over the classes
STEPS = int(os.environ.get("LAB_STEPS", 500 if QUICK else 20_000))   # updates per training set
SNAPSHOTS = 2 if QUICK else 5    # evenly spaced snapshots per run; each is saved and measured with FID
N_SNAPSHOT = 100 if QUICK else 1000   # unconditional samples per snapshot for FID
BATCH_TRAIN = 64
LABEL_DROPOUT = 0.1          # the CFG switch: 10% of training labels become "no label"


def subset(n):
    """The first n // 10 training images of each class: images in [-1, 1], labels, and their indices."""
    train_images, train_labels = edm.load_cifar10("train")
    idx = torch.cat([torch.nonzero(train_labels == c).flatten()[:n // 10] for c in range(10)])
    return train_images[idx].float() / 127.5 - 1, train_labels[idx], idx


def make_net():
    """EDM's EDMPrecond around a smaller SongUNet (EDM's own CIFAR-10 network: 128 channels, [2, 2, 2], 4 blocks)."""
    EDMPrecond, _ = edm.edm_modules()
    return EDMPrecond(img_resolution=32, img_channels=3, label_dim=10, model_type="SongUNet",
                      label_dropout=LABEL_DROPOUT,
                      model_channels=64, channel_mult=[1, 2, 2], num_blocks=2,
                      attn_resolutions=[16], dropout=0.0, embedding_type="positional", encoder_type="standard",
                      decoder_type="standard", channel_mult_noise=1, resample_filter=[1, 1])


def D_own(net, x, sigma, labels, w):
    """CFG with one network: D(x; σ) with no label, plus w times the difference the label makes."""
    d_uncond = D(net, x, sigma)
    if w == 0:
        return d_uncond
    return d_uncond + w * (D(net, x, sigma, labels) - d_uncond)


def own_samples(net, n, w=0, seed=10):
    """n uint8 samples requesting class i mod 10 for image i; w = 0 ignores the request (no label)."""
    labels, out = torch.arange(n) % 10, []
    for b in range(0, n, 500):
        lab = labels[b:b + 500]
        out.append(to_uint8(edm_sample(lambda x, s: D_own(net, x, s, lab, w), n=len(lab), seed=seed + b)))
    return torch.cat(out), labels


def train(n_train):
    """Train a CFG network on n_train images; save and measure a snapshot every STEPS // SNAPSHOTS updates.
    A finished run (its last snapshot and its metrics in outputs/) is loaded instead of trained again."""
    final = OUT / f"subset{n_train}_update{STEPS}.pt"
    metrics_file = OUT / f"subset{n_train}_steps{STEPS}_metrics.json"
    progress_file = OUT / f"subset{n_train}_steps{STEPS}_progress.npz"
    if final.exists() and metrics_file.exists() and progress_file.exists():
        print(f"{n_train} images: loading the finished run from {final.name}", flush=True)
        net = make_net().to(DEVICE).eval().requires_grad_(False)
        net.load_state_dict(torch.load(final, map_location=DEVICE))
        saved, progress = json.loads(metrics_file.read_text()), np.load(progress_file)
        return {"net": net, "losses": saved["losses"], "fids": {int(k): v for k, v in saved["fids"].items()},
                "progress": {int(k): torch.from_numpy(progress[k]) for k in progress.files}}

    x_train, y_train, _ = subset(n_train)
    torch.manual_seed(0)
    _, EDMLoss = edm.edm_modules()
    net = make_net().to(DEVICE).train()
    ema = copy.deepcopy(net).eval().requires_grad_(False)       # averaged weights, used for sampling
    opt = torch.optim.Adam(net.parameters(), lr=2e-4)
    loss_fn, gen = EDMLoss(), torch.Generator().manual_seed(0)
    losses, fids, progress, t0 = [], {}, {}, time.time()
    for step in range(1, STEPS + 1):
        batch = torch.randint(len(x_train), (BATCH_TRAIN,), generator=gen)
        labels = F.one_hot(y_train[batch], 10).float().to(DEVICE)   # the network drops 10% of these while training
        loss = loss_fn(net, x_train[batch].to(DEVICE), labels).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            decay = 0.9 if step <= 1000 else 0.999              # the course's EMA, with its warm-up
            for p_ema, p in zip(ema.parameters(), net.parameters()):
                p_ema.lerp_(p, 1 - decay)
        losses.append(loss.item())
        if step == 100:
            print(f"{n_train} images: {how_long((time.time() - t0) / 100 * STEPS)} for {STEPS} updates, "
                  f"plus FID at each snapshot", flush=True)
        if step % (STEPS // SNAPSHOTS) == 0:                    # a snapshot
            torch.save(ema.state_dict(), OUT / f"subset{n_train}_update{step}.pt")
            images = own_samples(ema, N_SNAPSHOT)[0]
            fids[step], progress[step] = edm.fid(images), images[:10]
            print(f"{n_train} images, update {step}: loss {np.mean(losses[-500:]):.3f}, FID {fids[step]:.1f}, "
                  f"{how_long(time.time() - t0)} so far", flush=True)
    metrics_file.write_text(json.dumps({"n_train": n_train, "steps": STEPS, "n_snapshot_samples": N_SNAPSHOT,
                                        "fids": fids, "losses": losses}))
    np.savez(progress_file, **{str(s): images.numpy() for s, images in progress.items()})
    return {"net": ema, "losses": losses, "fids": fids, "progress": progress}


def plot_runs(runs):
    """The loss curves and FID across snapshots, the snapshot table, and the samples at each snapshot."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
    for n, run in runs.items():
        k = min(200, len(run["losses"]))
        axes[0].plot(range(k - 1, len(run["losses"])), np.convolve(run["losses"], np.ones(k) / k, mode="valid"),
                     label=f"{n} training images")
        axes[1].plot(list(run["fids"]), list(run["fids"].values()), marker="o", label=f"{n} training images")
    axes[0].set(yscale="log", xlabel="update", ylabel=f"loss, average of {k} updates", title="Training loss")
    axes[1].set(xlabel="update", ylabel=f"FID ({N_SNAPSHOT} unconditional samples)", title="FID across snapshots")
    axes[0].legend()
    axes[1].legend()
    fig.tight_layout()
    save_fig(fig, "p6_loss_fid")
    write_table("p6_fid_snapshots",
                [{"training images": n, "update": s, "loss (mean of last 500 updates)": float(np.mean(run["losses"][max(0, s - 500):s])),
                  "FID": f} for n, run in runs.items() for s, f in run["fids"].items()],
                caption=f"Part 6: FID of {N_SNAPSHOT} unconditional samples (no label) from each snapshot's averaged "
                        f"weights, against all 50,000 CIFAR-10 training images.",
                formats={"loss (mean of last 500 updates)": "{:.4f}", "FID": "{:.1f}"})
    for n, run in runs.items():
        show_grid({f"update {s}": images for s, images in run["progress"].items()},
                  title=f"{n} training images: unconditional samples from the same noise at each snapshot",
                  name=f"p6_progress_n{n}")


def nearest(images, x_train):
    """Distance to, and index of, the nearest training image, in pixels scaled to [-1, 1]."""
    return torch.cdist(images.float().flatten(1) / 127.5 - 1, x_train.flatten(1)).min(1)


def memorization(runs):
    """For 100 samples from the last snapshot, the nearest training image in pixels; the same for 100 real test
    images, which are new images by definition."""
    train_images, _ = edm.load_cifar10("train")
    test_images, _ = edm.load_cifar10("test")
    rows = []
    for n, run in runs.items():
        x_train, _, idx = subset(n)
        samples = own_samples(run["net"], 100, seed=20)[0]
        d_gen, i_gen = nearest(samples, x_train)
        d_test, _ = nearest(test_images[:100], x_train)
        print(f"{n} training images, median distance to the nearest training image: "
              f"samples {d_gen.median():.1f}, real test images {d_test.median():.1f}")
        rows.append({"training images": n, "median distance, samples": d_gen.median().item(),
                     "median distance, real test images": d_test.median().item(),
                     "samples closer than every test image": (d_gen < d_test.min()).float().mean().item()})
        show_grid({"sample": samples[:10], "nearest training image": train_images[idx][i_gen[:10]]},
                  title=f"{n} training images, last snapshot", name=f"p6_nearest_n{n}")
    write_table("p6_nearest", rows,
                caption="Part 6: distance from 100 samples (last snapshot, no label) and from 100 real test images to "
                        "their nearest training image, in pixels scaled to [-1, 1] (3,072 values per image).",
                formats={"median distance, samples": "{:.1f}", "median distance, real test images": "{:.1f}",
                         "samples closer than every test image": "{:.0%}"})


def cfg_grids(runs):
    """CFG with one network: D(x; σ) with no label and D(x; σ, c) with the label, for w = 0, 1, 2, 4."""
    rows_out = []
    for n, run in runs.items():
        rows = {}
        for w in [0, 1, 2, 4]:
            x, labels = own_samples(run["net"], 100, w=w, seed=30)
            accuracy = (edm.classify(x) == labels).float().mean().item()
            rows[f"w = {w}, accuracy {accuracy:.0%}"] = x[:10]
            rows_out.append({"training images": n, "w": w, "images": 100, "label accuracy": accuracy})
        show_grid(rows, title=f"{n} training images: CFG with one network, the same starting noise in each column",
                  name=f"p6_cfg_n{n}")
    write_table("p6_cfg_accuracy", rows_out,
                caption="Part 6: label accuracy (ResNet-56) of 100 samples per w, image i requesting class i mod 10. "
                        "At w = 0 the request is ignored, so about 10% is chance.",
                formats={"label accuracy": "{:.0%}"})


def subsets():
    """Everything in part 6 up to the encouraged full run."""
    runs = {n: train(n) for n in SIZES}
    plot_runs(runs)
    memorization(runs)
    cfg_grids(runs)


# ------------------------------------------------------------------ the encouraged full run with EDM's train.py
def export_edm_dataset(n=50_000):
    """Write the first n training images in EDM's dataset format: the zip of PNGs that dataset_tool.py makes."""
    from PIL import Image
    train_images, train_labels = edm.load_cifar10("train")
    path, labels = (OUT / f"cifar10-32x32-n{n}.zip").resolve(), []
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as z:
        for i in range(n):
            name = f"{i // 1000:05d}/img{i:08d}.png"
            png = io.BytesIO()
            Image.fromarray(train_images[i].permute(1, 2, 0).numpy()).save(png, format="png", compress_level=0)
            z.writestr(name, png.getvalue())
            labels.append([name, int(train_labels[i])])
        z.writestr("dataset.json", json.dumps({"labels": labels}))
    return path


def edm_export():
    """Download EDM (with its fixes for current Python and PyTorch), write the dataset, and print the command for the
    full run."""
    edm.setup()
    data = export_edm_dataset()
    print(f"dataset: {data}")
    print("Start the full run in a terminal (a background job on DSMLP), not in a notebook:")
    print("  bash scripts/train_edm_full.sh")
    print(f"which runs, in {edm.EDM_DIR.resolve()}:")
    print(f"  torchrun --standalone --nproc_per_node=1 train.py --outdir={ROOT / 'training-runs'} --data={data} "
          f"--cond=0 --arch=ddpmpp --batch=512 --batch-gpu=64 --duration=10 --snap=10 --dump=10")


def edm_analyze(run_dirs):
    """The full run's loss from stats.jsonl, and FID on 1,000 unconditional samples from each saved snapshot.
    run_dirs: the run's folder, or its folders in order when it was resumed (each --resume writes a new folder).
    Each snapshot's FID is cached in outputs/, so a rerun while training continues only measures new snapshots."""
    edm.setup()                                              # snapshots are pickles that need EDM's code
    run_dirs = [Path(d) for d in ([run_dirs] if isinstance(run_dirs, (str, Path)) else run_dirs)]
    rows = []
    for d in run_dirs:
        for line in open(d / "stats.jsonl"):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:                     # a line that train.py is still writing
                pass
    rows = sorted((r for r in rows if "Loss/loss" in r and "Progress/kimg" in r), key=lambda r: r["Progress/kimg"]["mean"])
    n_images = 100 if QUICK else 1000
    cache_file = OUT / f"full_run_fid_n{n_images}.json"
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    snapshot_fid = {}
    for d in run_dirs:
        for path in sorted(d.glob("network-snapshot-*.pkl")):
            mimg = int(path.stem.split("-")[-1]) / 1000          # million training images seen
            if mimg in snapshot_fid:                            # a resumed run starts by saving the same network
                continue
            key = f"{d.name}/{path.name}/{int(path.stat().st_mtime)}"   # a new run with the same name is new
            if key not in cache:
                try:
                    with open(path, "rb") as f:
                        snap = pickle.load(f)["ema"].to(DEVICE).eval().requires_grad_(False)
                except (EOFError, pickle.UnpicklingError):
                    print(f"{d.name}/{path.name}: still being written; skipped", flush=True)
                    continue
                images = torch.cat([to_uint8(edm_sample(lambda x, s: D(snap, x, s), n=min(edm.BATCH, n_images - b),
                                                        seed=3000 + b))
                                    for b in range(0, n_images, edm.BATCH)])
                cache[key] = edm.fid(images)
                cache_file.write_text(json.dumps(cache, indent=2))
                print(f"{d.name}/{path.name}: FID {cache[key]:.1f}", flush=True)
            snapshot_fid[mimg] = cache[key]
    snapshot_fid = dict(sorted(snapshot_fid.items()))
    run_name = ", ".join(d.name for d in run_dirs)
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
    axes[0].plot([r["Progress/kimg"]["mean"] / 1000 for r in rows], [r["Loss/loss"]["mean"] for r in rows])
    axes[0].set(xlabel="million training images", ylabel="loss", title="Training loss, from stats.jsonl")
    axes[1].plot(list(snapshot_fid), list(snapshot_fid.values()), marker="o")
    axes[1].set(xlabel="million training images", ylabel=f"FID ({n_images} samples, log scale)", yscale="log",
                title="FID across snapshots (the first is the untrained network)")
    fig.tight_layout()
    save_fig(fig, "full_run_loss_fid")
    if snapshot_fid:
        write_table("full_run_fid", [{"million training images": m, "FID": f} for m, f in snapshot_fid.items()],
                    caption=f"Full run of EDM's train.py ({run_name}): FID of {n_images} unconditional samples "
                            f"per snapshot, against all 50,000 CIFAR-10 training images.",
                    formats={"FID": "{:.1f}"})
