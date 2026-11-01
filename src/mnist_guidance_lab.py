"""Helpers for the MNIST guidance assignment: the class-conditional U-Net, the frozen digit classifier (an
evaluator, never a guide), the sampler with classifier-free guidance, and two diagnostics.

Course convention: x0 is noise at t = 0, x1 is data at t = 1, x_t = alpha_t x1 + sigma_t x0 on the cosine path
(alpha = sin(pi t / 2), sigma = cos(pi t / 2)), and the network predicts the clean image x1 from (x_t, t, label).
Label 10 means "no request"; the sampler mixes the two guesses as u + w (c - u).

The model and sampler are the course's speedrun/common.py and speedrun/sample.py, unchanged in behavior.
"""
from __future__ import annotations

import hashlib
import math
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def alpha_sigma(t):
    """The cosine path: t = 0 is noise, t = 1 is data."""
    return torch.sin(t * torch.pi / 2), torch.cos(t * torch.pi / 2)


def alpha_sigma_dot(t):
    return (torch.pi / 2) * torch.cos(t * torch.pi / 2), -(torch.pi / 2) * torch.sin(t * torch.pi / 2)


# ------------------------------------------------------------------ the networks
class Evaluator(nn.Module):
    """A frozen MNIST classifier. It judges samples; it never steers them."""
    FEAT = 128

    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Linear(64 * 7 * 7, self.FEAT)
        self.head = nn.Linear(self.FEAT, 10)

    def features(self, x):
        return F.relu(self.fc(self.body(x).flatten(1)))

    def forward(self, x):
        f = self.features(x)
        return self.head(f), f


class Block(nn.Module):
    def __init__(self, cin, cout, emb):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.emb = nn.Linear(emb, cout)
        self.n1, self.n2 = nn.GroupNorm(8, cout), nn.GroupNorm(8, cout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, e):
        h = F.silu(self.n1(self.c1(x)))
        h = h + self.emb(e)[:, :, None, None]        # the time embedding (and the request) enter every block here
        h = F.silu(self.n2(self.c2(h)))
        return h + self.skip(x)


class UNet(nn.Module):
    """Class-conditional denoiser: predicts the clean image x1 from (x_t, t, label). Label 10 = no request."""

    def __init__(self, ch=64, emb=256):
        super().__init__()
        self.temb = nn.Sequential(nn.Linear(1, emb), nn.SiLU(), nn.Linear(emb, emb))
        self.cemb = nn.Embedding(11, emb)             # 10 digits + 1 "no request" label
        self.d1, self.d2, self.d3 = Block(1, ch, emb), Block(ch, ch * 2, emb), Block(ch * 2, ch * 2, emb)
        self.mid = Block(ch * 2, ch * 2, emb)
        self.u3 = Block(ch * 4, ch * 2, emb)
        self.u2 = Block(ch * 4, ch, emb)
        self.u1 = Block(ch * 2, ch, emb)
        self.out = nn.Conv2d(ch, 1, 3, padding=1)

    def forward(self, x, t, y):
        e = self.temb(t[:, None]) + self.cemb(y)     # the request is added to the time embedding
        h1 = self.d1(x, e)                           # 28 x 28
        h2 = self.d2(F.avg_pool2d(h1, 2), e)         # 14 x 14
        h3 = self.d3(F.avg_pool2d(h2, 2), e)         # 7 x 7
        m = self.mid(h3, e)
        u = self.u3(torch.cat([m, h3], 1), e)
        u = F.interpolate(u, size=h2.shape[-2:], mode="nearest")
        u = self.u2(torch.cat([u, h2], 1), e)
        u = F.interpolate(u, size=h1.shape[-2:], mode="nearest")
        u = self.u1(torch.cat([u, h1], 1), e)
        return self.out(u)


# ------------------------------------------------------------------ weights
def fetch(files: dict, folder=".") -> None:
    """Make sure each weight file is next to the notebook: files = {name: (url, sha256)}. Downloads what is
    missing or damaged and checks every file's hash."""
    for name, (url, digest) in files.items():
        path = Path(folder) / name
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            print("downloading", name)
            urllib.request.urlretrieve(url, path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"{name} does not match the course copy: delete it and run this cell again")


def load_models(folder=".", device="cpu"):
    """The class-conditional generator and the frozen evaluator, from generator.pt and evaluator.pt."""
    folder = Path(folder)
    ck = torch.load(folder / "generator.pt", map_location="cpu", weights_only=True)
    net = UNet(ch=ck["ch"]).to(device)
    net.load_state_dict(ck["state"])
    ck = torch.load(folder / "evaluator.pt", map_location="cpu", weights_only=True)
    evaluator = Evaluator().to(device)
    evaluator.load_state_dict(ck["state"])
    for model in (net, evaluator):
        model.eval().requires_grad_(False)
    return net, evaluator


# ------------------------------------------------------------------ sampling with guidance
@torch.no_grad()
def sample(net, n, labels, device, steps=16, w=2.0, eta=0.0, order="ddim", seed=0, batch=128):
    """n images for the requested labels. Guidance u + w (c - u): w = 0 ignores the request, w = 1 uses it as
    trained, w > 1 exaggerates it. Returns (images, network evaluations per image)."""
    if not isinstance(n, int) or n < 1 or not isinstance(steps, int) or steps < 1:
        raise ValueError("n and steps must be positive integers")
    if not isinstance(batch, int) or batch < 1:
        raise ValueError("batch must be a positive integer")
    if order not in ("ddim", "euler", "heun"):
        raise ValueError("order must be ddim, euler or heun")
    if not math.isfinite(w) or w < 0 or not math.isfinite(eta) or not 0 <= eta <= 1:
        raise ValueError("w must be nonnegative and eta must be in [0,1]")
    if eta and order != "ddim":
        raise ValueError("eta is supported only by DDIM; Euler/Heun are deterministic")
    labels = torch.as_tensor(labels, dtype=torch.long, device="cpu")
    if labels.shape != (n,) or (labels < 0).any() or (labels > 9).any():
        raise ValueError("supply n digit labels in [0,9]")
    g = torch.Generator(device="cpu").manual_seed(seed)       # one CPU noise bank: the same on every device
    initial = torch.randn(n, 1, 28, 28, generator=g)
    outs = []
    for i in range(0, n, batch):
        x, nfe = _sample_chunk(net, initial[i:i + batch], labels[i:i + batch], device, steps, w, eta, order,
                               seed + i + 1)
        outs.append(x.cpu())
    return torch.cat(outs), nfe


@torch.no_grad()
def _sample_chunk(net, initial, labels, device, steps, w, eta, order, noise_seed):
    g = torch.Generator(device="cpu").manual_seed(noise_seed)
    x, y = initial.to(device), labels.to(device)
    y_un = torch.full_like(y, 10)
    ts = torch.linspace(0, 1, steps + 1, device=device)
    nfe = 0

    def x1_hat(xt, t):
        nonlocal nfe
        tb = t.expand(xt.shape[0]).contiguous()
        if w in (0, 1):
            nfe += 1
            return net(xt, tb, y_un if w == 0 else y).clamp(-1, 1)
        out = net(torch.cat([xt, xt]), torch.cat([tb, tb]), torch.cat([y, y_un]))
        nfe += 2
        c, u = out.chunk(2)
        return (u + w * (c - u)).clamp(-1, 1)

    def velocity(xt, t):
        a, sg = alpha_sigma(t)
        x1 = x1_hat(xt, t)
        x0 = (xt - a * x1) / sg
        ad, sd = alpha_sigma_dot(t)
        return ad * x1 + sd * x0

    for i, (t, s) in enumerate(zip(ts[:-1], ts[1:])):
        if i == steps - 1:
            x = x1_hat(x, t)                          # the last step lands on the guess
        elif order in ("euler", "heun"):
            v = velocity(x, t)
            pred = x + (s - t) * v
            x = pred if order == "euler" else x + (s - t) * (v + velocity(pred, s)) / 2
        else:
            at, st = alpha_sigma(t)
            a_s, s_s = alpha_sigma(s)
            x1 = x1_hat(x, t)
            x0 = (x - at * x1) / st
            sig = eta * s_s / st * (1 - (at / a_s).square()).clamp_min(0).sqrt()
            keep = (s_s.square() - sig.square()).clamp_min(0).sqrt()
            x = a_s * x1 + keep * x0
            if eta:
                x += sig * torch.randn(x.shape, generator=g).to(device)
    return x.clamp(-1, 1), nfe


# ------------------------------------------------------------------ diagnostics and display
@torch.no_grad()
def classification(evaluator, x, y, device="cpu", bs=256):
    """Share of images the classifier reads as their requested digit, and its mean confidence in that digit."""
    ok, confidence = 0, 0.0
    for i in range(0, len(x), bs):
        logits, _ = evaluator(x[i:i + bs].to(device))
        labels = y[i:i + bs].to(device)
        ok += (logits.argmax(1) == labels).sum().item()
        confidence += logits.softmax(1).gather(1, labels[:, None]).sum().item()
    return ok / len(x), confidence / len(x)


def diagnostics(evaluator, images, labels, device="cpu") -> dict:
    """target_match: read as the requested digit. pixel_spread: mean per-pixel variance within each requested
    digit, a measure of variation (wanted or not), not of quality."""
    match, confidence = classification(evaluator, images, labels, device)
    spread = np.mean([float(images[labels == c].var(dim=0, unbiased=False).mean()) for c in labels.unique()])
    return dict(target_match=round(match, 3), confidence=round(confidence, 3), pixel_spread=round(float(spread), 4))


def grid(rows: dict, count: int = 10) -> None:
    """One row of images per entry of rows = {label: images}; the first `count` images of each."""
    fig, axes = plt.subplots(len(rows), count, figsize=(1.2 * count, 1.4 * len(rows)), squeeze=False)
    for r, (label, images) in enumerate(rows.items()):
        for j, ax in enumerate(axes[r]):
            ax.imshow(images[j, 0], cmap="gray", vmin=-1, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[r, 0].set_ylabel(label, rotation=0, ha="right", va="center")
    fig.tight_layout()
    plt.show()
