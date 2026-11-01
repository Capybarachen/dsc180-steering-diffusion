"""EDM on CIFAR-10: the course notebook's downloads, models, data, sampler, plots, classifier and FID
(cifar10_edm_lab.ipynb), as functions that download and load what they need the first time they are called.

Notation, as in the course: a noisy image is x = x1 + σ x0 for a clean image x1 and Gaussian noise x0, and
D(x; σ, c) is the network's estimate of the clean image for class c; D(x; σ) with no class.
"""
import functools
import io
import os
import pickle
import sys
import tarfile
import types
import zipfile
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from .common import DEVICE, DL, QUICK, download, fetch, frechet, gaussian, save_fig

EDM_COMMIT = "008a4e5316c8e3bfe61a62f874bddba254295afb"   # the NVlabs/edm version the course notebook was tested with
NV = "https://nvlabs-fi-cdn.nvidia.com/edm"
FILES = {   # file: (address, SHA-256)
    "edm-cifar10-32x32-uncond-vp.pkl": (f"{NV}/pretrained/edm-cifar10-32x32-uncond-vp.pkl",
                                        "4d5dcc1f1d0d41c8934ad21626eeddbdc0460182becf9fc059a0631b1eedb4da"),
    "edm-cifar10-32x32-cond-vp.pkl": (f"{NV}/pretrained/edm-cifar10-32x32-cond-vp.pkl",
                                      "b27385e4f2f8d4d1e2e2c03864c5a9574a0c58ba81eb11625ae6d15c401f4796"),
    "cifar10-32x32.npz": (f"{NV}/fid-refs/cifar10-32x32.npz",
                          "4ada108cdfd8e43409ed427f5f76631beac9b9617cdcaf416d41b1e161d85969"),
    "inception-2015-12-05.pkl": ("https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/"
                                 "files/metrics/inception-2015-12-05.pkl",
                                 "a31bd1d4522101109044ec28621a5fcb591ed6115e0d58104fc013fa01ef94f7"),
    "cifar10_resnet56-187c023a.pt": ("https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/"
                                     "cifar10_resnet56-187c023a.pt",
                                     "187c023aee0c9cf3093a682d9447a538cbaf5489f7ae78a7df4edf2246ce380b"),
}
TAR = DL / "cifar-10-python.tar.gz"     # the original CIFAR-10 archive: used if you already have it
if not TAR.exists():                    # otherwise Hugging Face's copy of the same images, a fixed version
    HF = "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/0b2714987fa478483af9968de7c934580d0bb9a2/plain_text"
    FILES["cifar10-train.parquet"] = (f"{HF}/train-00000-of-00001.parquet",
                                      "8428b53a88a11ac374111006708df51469e315a22ac6d66470afd9c78d2ae883")
    FILES["cifar10-test.parquet"] = (f"{HF}/test-00000-of-00001.parquet",
                                     "841389e6f2d64f28bf17310e430aebac20ec3ba611a3c5e231dc93c645ce84de")
EDM_DIR = Path(os.environ.get("EDM_DIR", DL / f"edm-{EDM_COMMIT}"))
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
N_EVAL = int(os.environ.get("LAB_N_EVAL", 200 if QUICK else 1000))   # images per setting in part 5
BATCH = 100 if QUICK else 500                                        # images per sampling batch
EDM_FIXES = [   # (file in EDM, text, replacement): EDM's code with current Python and PyTorch
    # PyTorch 2.2+: the course notebook's one-line fix to EDM's training sampler
    ("torch_utils/misc.py", "super().__init__(dataset)", "super().__init__()"),
    # Python 3.12+ without setuptools: distutils is gone. train.py runs in its own process, where the shim in
    # setup() cannot help, so the import gets a fallback with the same function.
    ("dnnlib/util.py", "\nfrom distutils.util import strtobool\n",
     "\ntry:\n    from distutils.util import strtobool\nexcept ImportError:  # Python 3.12+ without setuptools\n"
     "    def strtobool(v):\n        return int(str(v).lower() in ('y', 'yes', 't', 'true', 'on', '1'))\n"),
    # PyTorch 2.6+: torch.load refuses the pickled network in a training state unless told to, so --resume failed
    ("training/training_loop.py", "torch.load(resume_state_dump, map_location=torch.device('cpu'))",
     "torch.load(resume_state_dump, map_location=torch.device('cpu'), weights_only=False)"),
]


def fix_edm(edm_dir):
    """Apply EDM_FIXES to a copy of EDM's code; each fix is applied once. Returns the files it changed."""
    changed = []
    for name, old, new in EDM_FIXES:
        path = Path(edm_dir) / name
        text = path.read_text()
        if old in text and new not in text:
            path.write_text(text.replace(old, new, 1))
            changed.append(name)
    return changed


# ------------------------------------------------------------------ downloads and EDM's code
@functools.lru_cache(maxsize=None)
def setup():
    """The notebook's Downloads cell: every file checked against its SHA-256, EDM's code at the tested commit with
    its fixes for current Python and PyTorch, and EDM's folder on sys.path (its pickles need its code)."""
    fetch(FILES)
    if not EDM_DIR.exists():
        print("downloading EDM's code", flush=True)
        download(f"https://github.com/NVlabs/edm/archive/{EDM_COMMIT}.zip", DL / "edm.zip")
        with zipfile.ZipFile(DL / "edm.zip") as z:
            z.extractall(DL)
    if "EDM_DIR" in os.environ:                          # your own clone: we do not edit it for you
        for name, old, new in EDM_FIXES:
            if old in (EDM_DIR / name).read_text() and new not in (EDM_DIR / name).read_text():
                print(f"For train.py with current Python and PyTorch, in {EDM_DIR / name} replace\n  {old.strip()}\n"
                      f"with\n  {new.strip()}")
    else:
        fix_edm(EDM_DIR)
    try:
        import distutils.util  # noqa: F401              # EDM imports one helper from distutils, removed in Python 3.12
    except ImportError:
        shim = types.ModuleType("distutils.util")
        shim.strtobool = lambda v: int(str(v).lower() in ("y", "yes", "t", "true", "on", "1"))
        sys.modules["distutils"] = types.ModuleType("distutils")
        sys.modules["distutils"].util = shim
        sys.modules["distutils.util"] = shim
    if str(EDM_DIR) not in sys.path:
        sys.path.insert(0, str(EDM_DIR))
    return EDM_DIR


def edm_modules():
    """EDM's network and loss classes, used in part 6."""
    setup()
    from training.networks import EDMPrecond
    from training.loss import EDMLoss
    return EDMPrecond, EDMLoss


# ------------------------------------------------------------------ models and data
def load_edm(name):
    """A network from one of EDM's pickles: its `ema` entry, the averaged weights that EDM samples with."""
    setup()
    with open(DL / name, "rb") as f:
        return pickle.load(f)["ema"].to(DEVICE).eval().requires_grad_(False)


@functools.lru_cache(maxsize=None)
def networks():
    """EDM's pretrained CIFAR-10 networks: (unconditional D(x; σ), conditional D(x; σ, c))."""
    return load_edm("edm-cifar10-32x32-uncond-vp.pkl"), load_edm("edm-cifar10-32x32-cond-vp.pkl")


@functools.lru_cache(maxsize=None)
def load_cifar10(split):
    """CIFAR-10 as uint8 images of shape (n, 3, 32, 32) and integer labels: 50,000 for "train", 10,000 for "test"."""
    setup()
    if TAR.exists():
        names = [f"data_batch_{i}" for i in range(1, 6)] if split == "train" else ["test_batch"]
        images, labels = [], []
        with tarfile.open(TAR) as tar:
            for name in names:
                d = pickle.load(tar.extractfile(f"cifar-10-batches-py/{name}"), encoding="bytes")
                images.append(d[b"data"].reshape(-1, 3, 32, 32))
                labels += d[b"labels"]
        return torch.from_numpy(np.concatenate(images)), torch.tensor(labels)
    import pandas as pd                          # Hugging Face's copy: one PNG per row
    from PIL import Image
    table = pd.read_parquet(DL / f"cifar10-{split}.parquet")
    images = np.stack([np.asarray(Image.open(io.BytesIO(img["bytes"])).convert("RGB")) for img in table["img"]])
    return torch.from_numpy(images).permute(0, 3, 1, 2).contiguous(), torch.tensor(table["label"].to_numpy())


# ------------------------------------------------------------------ plotting
def to_uint8(x):
    """Network images in [-1, 1] to uint8 pixels, as EDM's generate.py saves them."""
    return (x * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu()


def show_grid(rows, title=None, name=None):
    """rows: {row label: uint8 images of shape (n, 3, 32, 32)}. Saved as figures/<name>.png."""
    ncol = max(len(v) for v in rows.values())
    fig, axes = plt.subplots(len(rows), ncol, figsize=(1.05 * ncol + 1.2, 1.1 * len(rows) + 0.4), squeeze=False)
    for r, (label, images) in enumerate(rows.items()):
        for j, ax in enumerate(axes[r]):
            ax.set_xticks([])
            ax.set_yticks([])
            if j < len(images):
                ax.imshow(images[j].permute(1, 2, 0).numpy())
            else:
                ax.axis("off")
        axes[r, 0].set_ylabel(label, rotation=0, ha="right", va="center", fontsize=9)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return save_fig(fig, name) if name else plt.show()


def show_trajectory(history, rows=3, steps=(0, 3, 6, 8, 10, 12, 14, 17), title=None, name=None):
    """For the first `rows` images: the noisy image (as the course's x_t = x / sqrt(1 + σ²)) above the estimate
    D(x; σ), at several of the sampler's steps. Saved as figures/<name>.png."""
    fig, axes = plt.subplots(2 * rows, len(steps), figsize=(1.25 * len(steps) + 1.4, 2.6 * rows), squeeze=False)
    for j, k in enumerate(steps):
        s, x, estimate = history[k]
        for i in range(rows):
            axes[2 * i, j].imshow(to_uint8(x[i] / (1 + s ** 2) ** 0.5).permute(1, 2, 0).numpy())
            axes[2 * i + 1, j].imshow(to_uint8(estimate[i]).permute(1, 2, 0).numpy())
        axes[0, j].set_title(f"σ = {s:.2g}", fontsize=9)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    for i in range(rows):
        axes[2 * i, 0].set_ylabel("noisy x", rotation=0, ha="right", va="center", fontsize=9)
        axes[2 * i + 1, 0].set_ylabel("D(x; σ)", rotation=0, ha="right", va="center", fontsize=9)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return save_fig(fig, name) if name else plt.show()


# ------------------------------------------------------------------ the sampler, the denoiser and CFG
def edm_sample(denoise, n, seed, steps=18, sigma_min=0.002, sigma_max=80.0, rho=7.0, record=False):
    """EDM's deterministic sampler (Heun, 18 noise levels, 35 network evaluations). denoise(x, sigma) returns the
    clean-image estimate. With record=True it also returns, for each step, the noise level σ, the noisy image x
    and the estimate D(x; σ)."""
    i = torch.arange(steps, dtype=torch.float64)
    sigmas = (sigma_max ** (1 / rho) + i / (steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    sigmas = sigmas.tolist() + [0.0]
    x = torch.randn(n, 3, 32, 32, generator=torch.Generator().manual_seed(seed)).to(DEVICE) * sigmas[0]
    history = []
    for s, s_next in zip(sigmas[:-1], sigmas[1:]):
        estimate = denoise(x, s)
        if record:
            history.append((s, x.cpu(), estimate.cpu()))
        d = (x - estimate) / s                               # direction from the clean estimate to x
        x_next = x + (s_next - s) * d                        # Euler step to the next noise level
        if s_next > 0:                                       # Heun: average with the direction there
            x_next = x + (s_next - s) * (d + (x_next - denoise(x_next, s_next)) / s_next) / 2
        x = x_next
    return (x, history) if record else x


@torch.no_grad()
def D(net, x, sigma, labels=None):
    """The clean-image estimate: D(x; σ) if labels is None, else D(x; σ, c) for the class indices in labels."""
    s = torch.full((len(x),), sigma, device=x.device)
    c = None if labels is None else F.one_hot(labels, 10).float().to(x.device)
    return net(x, s, c).float()


def D_cfg(x, sigma, labels, w):
    """CFG with two networks: D(x; σ) + w (D(x; σ, c) − D(x; σ))."""
    uncond, cond = networks()
    if w == 1:
        return D(cond, x, sigma, labels)                     # one network evaluation instead of two
    d_uncond = D(uncond, x, sigma)
    if w == 0:
        return d_uncond
    return d_uncond + w * (D(cond, x, sigma, labels) - d_uncond)


# ------------------------------------------------------------------ the classifier that scores label accuracy
class Block(torch.nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(cout)
        self.conv2 = torch.nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(cout)
        self.downsample = None
        if stride != 1 or cin != cout:
            self.downsample = torch.nn.Sequential(torch.nn.Conv2d(cin, cout, 1, stride, bias=False),
                                                  torch.nn.BatchNorm2d(cout))

    def forward(self, x):
        out = self.bn2(self.conv2(torch.relu(self.bn1(self.conv1(x)))))
        return torch.relu(out + (x if self.downsample is None else self.downsample(x)))


class ResNet56(torch.nn.Module):
    """The CIFAR-10 ResNet-56 of chenyaofo/pytorch-cifar-models, with matching parameter names."""
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(16)
        stage = lambda cin, cout, stride: torch.nn.Sequential(Block(cin, cout, stride),  # noqa: E731
                                                              *[Block(cout, cout, 1) for _ in range(8)])
        self.layer1, self.layer2, self.layer3 = stage(16, 16, 1), stage(16, 32, 2), stage(32, 64, 2)
        self.fc = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = self.layer3(self.layer2(self.layer1(torch.relu(self.bn1(self.conv1(x))))))
        return self.fc(x.mean((2, 3)))


MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
STD = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)


@functools.lru_cache(maxsize=None)
def classifier():
    """The pretrained ResNet-56. It only scores images; it never guides the sampler."""
    setup()
    net = ResNet56()
    net.load_state_dict(torch.load(DL / "cifar10_resnet56-187c023a.pt", map_location="cpu", weights_only=True))
    return net.to(DEVICE).eval()


@torch.no_grad()
def classifier_logits(images):
    """The classifier's 10 logits for each uint8 image (on the CPU)."""
    out = []
    for i in range(0, len(images), 500):
        x = (images[i:i + 500].float() / 255 - MEAN) / STD
        out.append(classifier()(x.to(DEVICE)).float().cpu())
    return torch.cat(out)


def classify(images):
    """The classifier's predicted class for each uint8 image."""
    return classifier_logits(images).argmax(1)


# ------------------------------------------------------------------ FID against EDM's reference
FID_DEVICE = DEVICE if DEVICE.type == "cuda" else torch.device("cpu")   # the detector is slow on Apple GPUs


@functools.lru_cache(maxsize=None)
def detector():
    """The Inception-v3 feature detector of EDM's fid.py (2048 features per image)."""
    setup()
    with open(DL / "inception-2015-12-05.pkl", "rb") as f:
        return pickle.load(f).to(FID_DEVICE)


@functools.lru_cache(maxsize=None)
def fid_reference():
    """EDM's reference: the mean and covariance of the features of the 50,000 CIFAR-10 training images."""
    setup()
    ref = np.load(DL / "cifar10-32x32.npz")
    return torch.from_numpy(ref["mu"]).double(), torch.from_numpy(ref["sigma"]).double()


@torch.no_grad()
def features(images):
    """2048 Inception features for each uint8 image."""
    net = detector()
    return torch.cat([net(images[i:i + 100].to(FID_DEVICE), return_features=True).double().cpu()
                      for i in range(0, len(images), 100)])


def fid(images):
    """FID of uint8 images against the 50,000 CIFAR-10 training images."""
    return frechet(*gaussian(features(images)), *fid_reference())
