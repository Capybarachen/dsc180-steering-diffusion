"""Shared by every part: the device, QUICK mode, the repository's folders, checked downloads, FID's Gaussian fit
and distance, and saving figures and tables. Importing this module downloads and loads nothing."""
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")   # before torch: Apple GPUs fall back to CPU

import csv
import hashlib
import json
import shutil
import urllib.error
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parent.parent
DL = ROOT / "downloads"     # code, models and data (not committed)
OUT = ROOT / "outputs"      # generated images, snapshots and caches (not committed)
FIG = ROOT / "figures"      # the figures for the report (committed)
RES = ROOT / "results"      # the tables for the report (committed)
for _folder in (DL, OUT, FIG, RES):
    _folder.mkdir(exist_ok=True)

if os.environ.get("LAB_DEVICE"):
    DEVICE = torch.device(os.environ["LAB_DEVICE"])
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

# QUICK cuts every step down far enough to test the code, not to give results. As in the course notebook it is on
# unless there is an NVIDIA GPU; LAB_QUICK=1 or LAB_QUICK=0 (run.py --quick or --full) overrides that.
if os.environ.get("LAB_QUICK") in ("0", "1"):
    QUICK = os.environ["LAB_QUICK"] == "1"
else:
    QUICK = DEVICE.type != "cuda"


def _cpu_limit():
    """CPUs this container may use. In a DSMLP or Kubernetes container PyTorch sees every core of the machine, and
    starting that many threads on a few allowed CPUs makes CPU linear algebra (FID's eigendecompositions) crawl."""
    try:                                                     # cgroup v2
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()[:2]
        if quota != "max":
            return max(1, int(quota) // int(period))
    except (OSError, ValueError):
        pass
    try:                                                     # cgroup v1
        quota = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read())
        period = int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read())
        if quota > 0:
            return max(1, quota // period)
    except (OSError, ValueError):
        pass
    return len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)


torch.set_num_threads(max(1, min(torch.get_num_threads(), _cpu_limit())))


# ------------------------------------------------------------------ downloads (the course notebook's code)
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(url, path, attempts=8):
    """Download with retries; an interrupted download resumes where it stopped."""
    part = Path(f"{path}.part")
    for _ in range(attempts):
        done = part.stat().st_size if part.exists() else 0
        request = urllib.request.Request(url, headers={"Range": f"bytes={done}-"} if done else {})
        try:
            with urllib.request.urlopen(request, timeout=60) as r, open(part, "ab" if r.status == 206 else "wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)
            break
        except urllib.error.HTTPError as e:
            if e.code == 416:                    # the partial file is already complete
                break
            print(f"  {e}; retrying", flush=True)
        except OSError as e:                     # a timeout or a dropped connection
            print(f"  {e}; retrying", flush=True)
    else:
        raise RuntimeError(f"could not download {url}; run this again to resume")
    part.replace(path)


def fetch(files, folder=DL):
    """files = {name: (address, SHA-256)}. Downloads what is missing or damaged into folder and checks every file."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name, (url, digest) in files.items():
        path = folder / name
        if path.exists() and sha256(path) == digest:
            continue
        print("downloading", name, flush=True)
        download(url, path)
        if sha256(path) != digest:
            path.unlink()
            raise RuntimeError(f"{name}: the checksum does not match; run this again")


def how_long(seconds):
    """A rough duration for progress messages."""
    return f"about {seconds / 60:.0f} min" if seconds >= 90 else f"about {seconds:.0f} s"


# ------------------------------------------------------------------ FID (the course notebook's code)
def gaussian(f):
    """The mean and covariance of a set of features, one row per image."""
    return f.mean(0), torch.cov(f.T)


def frechet(mu1, s1, mu2, s2):
    """The Fréchet distance between two Gaussians: FID when they are fitted to Inception (or other) features."""
    e, V = torch.linalg.eigh(s1)
    r = V @ torch.diag(e.clamp(min=0).sqrt()) @ V.T          # Σ1^(1/2)
    m = r @ s2 @ r                                           # has the same eigenvalues as Σ1 Σ2
    root_trace = torch.linalg.eigvalsh((m + m.T) / 2).clamp(min=0).sqrt().sum()   # tr((Σ1 Σ2)^(1/2))
    return float(((mu1 - mu2) ** 2).sum() + s1.trace() + s2.trace() - 2 * root_trace)


# ------------------------------------------------------------------ saving figures and tables
def save_fig(fig, name):
    """Save a figure as figures/<name>.png and close it."""
    path = FIG / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("figure:", path.relative_to(ROOT), flush=True)
    return path


def write_json(name, data):
    """Save data as results/<name>.json."""
    path = RES / f"{name}.json"
    path.write_text(json.dumps(data, indent=2) + "\n")
    print("table:", path.relative_to(ROOT), flush=True)
    return path


def write_table(name, rows, caption="", formats=None):
    """rows: a list of dicts with the same keys. Saves results/<name>.csv with the raw numbers and results/<name>.md,
    a Markdown table to paste into the report. formats: {column: format string}, for the Markdown table only."""
    formats = formats or {}
    columns = list(rows[0])
    with open(RES / f"{name}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, columns)
        writer.writeheader()
        writer.writerows(rows)

    def cell(column, value):
        if value is None:
            return "–"
        if column in formats:
            return formats[column].format(value)
        return f"{value:.4g}" if isinstance(value, float) else str(value)

    lines = [caption, ""] if caption else []
    lines += ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(cell(c, row[c]) for c in columns) + " |" for row in rows]
    (RES / f"{name}.md").write_text("\n".join(lines) + "\n")
    print("table:", (RES / f"{name}.csv").relative_to(ROOT), "and .md", flush=True)
