"""Shared helpers: project paths, cache locations, seeding, device choice, timing.

Import this module FIRST in every notebook/script. It sets cache environment
variables that transformers, datasets, gensim and NLTK read when they are imported.
"""
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

# ---- Paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"          # untouched downloads (parquet)
SPLITS = DATA / "splits"    # fixed train/test splits shared by all notebooks
RESULTS = ROOT / "results"  # metrics JSON written by src/evaluate.py
MODELS = ROOT / "models"    # our own trained models

# ---- Caches (all inside data/, which is gitignored) ------------------------
# setdefault: a value already set in the shell wins.
os.environ.setdefault("HF_HOME", str(DATA / "hf_cache"))
os.environ.setdefault("GENSIM_DATA_DIR", str(DATA / "gensim_data"))
os.environ.setdefault("NLTK_DATA", str(DATA / "nltk_data"))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # run unsupported MPS ops on CPU
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")   # silences a fork warning
# Some models ship only pytorch_model.bin; transformers would then download a second
# (safetensors) copy in a background thread. We have ~10 GB of disk, so turn that off.
os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "true")

# NLTK reads NLTK_DATA only on first import, so also add the path directly
# in case nltk was imported before this module.
try:
    import nltk
    if str(DATA / "nltk_data") not in nltk.data.path:
        nltk.data.path.insert(0, str(DATA / "nltk_data"))
except ImportError:
    pass

# ---- Global settings -------------------------------------------------------
SEED = 42
FAST = os.environ.get("NEWSMIND_FAST") == "1"  # NEWSMIND_FAST=1 -> small subsets


def set_seed(seed: int = SEED) -> None:
    """Make Python, NumPy and PyTorch random numbers repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)  # also seeds CUDA and MPS
    except ImportError:
        pass


def get_device():
    """Pick the fastest available device: CUDA GPU > Apple MPS > CPU."""
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@contextmanager
def timer(name: str, store: dict | None = None):
    """Time a block of code: `with timer("fit", t): model.fit(...)`.

    Prints the elapsed seconds and, if `store` is given, saves them in store[name].
    """
    start = time.perf_counter()
    yield
    secs = time.perf_counter() - start
    print(f"[{name}] {secs:.2f}s")
    if store is not None:
        store[name] = secs
