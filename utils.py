"""Small helpers shared by every module (no heavy imports here)."""
from __future__ import annotations

import os
import re


def progress(iterable, **kwargs):
    """tqdm progress bar if tqdm is installed, otherwise the plain iterable."""
    try:
        from tqdm.auto import tqdm
        return tqdm(iterable, **kwargs)
    except Exception:  # tqdm missing -> silently degrade
        return iterable


def get_device() -> str:
    """'cuda' if a GPU is visible to torch, else 'cpu'."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def stem_of(name) -> str:
    """File name / path -> stem without extension ('a/b/x.wav' -> 'x')."""
    return os.path.splitext(os.path.basename(str(name)))[0]


def free_gpu() -> None:
    """Release cached GPU memory between stages (safe to call without torch)."""
    try:
        import gc, torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def keys_hash(meta) -> str:
    """Short fingerprint of the ordered (split, stem) list. Cached arrays are only reused if it matches."""
    import hashlib
    s = "|".join(f"{a}/{b}" for a, b in zip(meta["split"], meta["stem"]))
    return hashlib.md5(s.encode()).hexdigest()[:10]
