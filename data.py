"""Locate and load the competition files.

Nothing here assumes a fixed folder layout: ``discover_layout`` walks DATA_DIR and finds train.csv / test.csv /
sample_submission.csv and the folders holding the audio. Anything it gets wrong can be overridden in CONFIG.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import stem_of

AUDIO_EXT = (".wav", ".flac", ".mp3", ".m4a", ".ogg")


def discover_layout(root: str) -> dict:
    """Walk ``root`` and report where the CSVs and the train/test audio folders are.

    Audio folder rule: among folders that contain audio files, pick the one whose name contains 'train'
    (resp. 'test') and that holds the most files. Falls back to any path component containing the word.
    """
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"DATA_DIR does not exist: {root!r}. On Kaggle look under /kaggle/input/ and fix CONFIG['DATA_DIR'].")
    out = dict(train_csv=None, test_csv=None, sample_csv=None, train_audio=None, test_audio=None)
    audio_dirs: dict[str, int] = {}
    for dp, _, files in os.walk(root):
        for f in files:
            lf = f.lower()
            full = os.path.join(dp, f)
            if lf == "train.csv":
                out["train_csv"] = out["train_csv"] or full
            elif lf == "test.csv":
                out["test_csv"] = out["test_csv"] or full
            elif lf in ("sample_submission.csv", "samplesubmission.csv"):
                out["sample_csv"] = out["sample_csv"] or full
            elif lf.endswith(AUDIO_EXT):
                audio_dirs[dp] = audio_dirs.get(dp, 0) + 1
    for split in ("train", "test"):
        by_name = [(n, d) for d, n in audio_dirs.items() if split in os.path.basename(d).lower()]
        by_path = [(n, d) for d, n in audio_dirs.items() if split in d.lower()]
        cands = by_name or by_path
        out[f"{split}_audio"] = max(cands)[1] if cands else None
    return out


def build_audio_index(directory: str | None) -> dict[str, str]:
    """{file stem: full path} for every audio file below ``directory`` (empty dict if None)."""
    idx: dict[str, str] = {}
    if not directory:
        return idx
    for dp, _, files in os.walk(directory):
        for f in files:
            if f.lower().endswith(AUDIO_EXT):
                idx.setdefault(stem_of(f), os.path.join(dp, f))
    return idx


def detect_columns(train: pd.DataFrame, id_col: str | None, label_col: str | None) -> tuple[str, str]:
    """Pick the file-name column and the numeric label column of train.csv (explicit CONFIG values win)."""
    cols = list(train.columns)
    if id_col is None:
        def looks_like_audio(s: pd.Series) -> bool:
            return s.astype(str).str.lower().str.contains(r"\.(?:wav|flac|mp3|m4a|ogg)$").mean() > 0.9
        def texty(c: str) -> bool:
            return pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])
        by_values = [c for c in cols if texty(c) and looks_like_audio(train[c])]
        by_name = [c for c in cols if any(k in c.lower() for k in ("file", "name", "audio", "id"))]
        by_dtype = [c for c in cols if texty(c)]
        id_col = (by_values or by_name or by_dtype or cols[:1])[0]
    if label_col is None:
        numeric = [c for c in cols if c != id_col and pd.api.types.is_numeric_dtype(train[c])]
        named = [c for c in numeric if any(k in c.lower() for k in ("label", "score", "grammar", "target"))]
        if not (named or numeric):
            raise ValueError(f"No numeric label column found in train.csv columns {cols}; set CONFIG['LABEL_COL'].")
        label_col = (named or numeric)[0]
    return id_col, label_col


def _build_split(raw: pd.DataFrame, id_col: str, label_col: str | None, split: str,
                 audio_dir: str | None, global_index: dict) -> pd.DataFrame:
    """One tidy table per split: split, id, stem, path (None if missing), label (NaN for test)."""
    local = build_audio_index(audio_dir)
    stems = raw[id_col].map(stem_of)
    paths = [local.get(s) or global_index.get(s) for s in stems]
    df = pd.DataFrame({"split": split, "id": raw[id_col].astype(str).values, "stem": stems.values, "path": paths})
    # SAFETY: test labels are random placeholders and are never read.
    df["label"] = raw[label_col].astype(float).values if (label_col is not None and split == "train") else np.nan
    return df


def load_competition(cfg, verbose: bool = True) -> dict:
    """Load train/test/sample_submission and resolve every audio path. Returns a dict of tables and column names."""
    layout = discover_layout(cfg.DATA_DIR)
    train_csv = cfg.TRAIN_CSV or layout["train_csv"]
    test_csv = cfg.TEST_CSV or layout["test_csv"]
    sample_csv = cfg.SAMPLE_SUB_CSV or layout["sample_csv"]
    train_audio = cfg.TRAIN_AUDIO_DIR or layout["train_audio"]
    test_audio = cfg.TEST_AUDIO_DIR or layout["test_audio"]
    for name, val in dict(train_csv=train_csv, test_csv=test_csv, sample_csv=sample_csv).items():
        if not val:
            raise FileNotFoundError(f"Could not find {name}; set it explicitly in CONFIG.")
    train_raw, test_raw, sample = pd.read_csv(train_csv), pd.read_csv(test_csv), pd.read_csv(sample_csv)
    id_col, label_col = detect_columns(train_raw, cfg.ID_COL, cfg.LABEL_COL)
    test_id_col = id_col if id_col in test_raw.columns else test_raw.columns[0]
    global_index = build_audio_index(cfg.DATA_DIR)          # fallback if split folders were not found
    train = _build_split(train_raw, id_col, label_col, "train", train_audio, global_index)
    test = _build_split(test_raw, test_id_col, None, "test", test_audio, global_index)
    if verbose:
        print("train.csv :", train_csv, train_raw.shape, "columns:", list(train_raw.columns))
        print("test.csv  :", test_csv, test_raw.shape, "columns:", list(test_raw.columns))
        print("sample sub:", sample_csv, sample.shape, "columns:", list(sample.columns))
        print("audio dirs: train =", train_audio, "| test =", test_audio)
        print(f"detected columns -> id: {id_col!r}, label: {label_col!r}")
        print(f"audio files found: train {train.path.notna().sum()}/{len(train)}, test {test.path.notna().sum()}/{len(test)}")
        print("label summary (train):\n", train.label.describe().round(3).to_string())
    return dict(train=train, test=test, sample=sample, id_col=id_col, label_col=label_col,
                sample_csv=sample_csv, train_csv=train_csv, test_csv=test_csv)
