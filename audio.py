"""Audio inspection and preprocessing: load -> mono 16 kHz -> trim edge silence -> loudness normalise -> cache."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .utils import progress


def scan_audio_info(meta: pd.DataFrame) -> pd.DataFrame:
    """Cheap header read (no decoding) for every file: sample rate, channels, duration, corrupt flag."""
    import soundfile as sf
    rows = []
    for r in progress(list(meta.itertuples()), desc="audio info"):
        row = dict(split=r.split, stem=r.stem, path=r.path, ok=False, error="", samplerate=np.nan,
                   channels=np.nan, frames=np.nan, duration=np.nan, subtype="")
        if not isinstance(r.path, str) or not os.path.exists(r.path):
            row["error"] = "file not found"
        else:
            try:
                i = sf.info(r.path)
                row.update(ok=True, samplerate=i.samplerate, channels=i.channels, frames=i.frames,
                           duration=i.duration, subtype=i.subtype)
            except Exception as e:  # corrupt / unreadable
                row["error"] = repr(e)[:200]
        rows.append(row)
    return pd.DataFrame(rows)


def preprocess_waveform(y: np.ndarray, sr: int, top_db: float, target_rms: float) -> np.ndarray:
    """Trim leading/trailing silence (internal pauses are kept: they carry fluency information) and set loudness.

    Loudness normalisation = scale so the RMS equals ``target_rms``, then limit the peak to 0.99 to avoid clipping.
    """
    import librosa
    trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    if len(trimmed) >= sr:        # never trim a clip down to <1 s (would mean trimming went wrong)
        y = trimmed
    rms = float(np.sqrt(np.mean(y ** 2))) + 1e-9
    y = y * (target_rms / rms)
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    if peak > 0.99:
        y = y * (0.99 / peak)
    return y.astype(np.float32)


def preprocess_all(meta: pd.DataFrame, cfg) -> pd.DataFrame:
    """Preprocess every file once and cache as 16-bit PCM wav. Adds columns ``cache_path``, ``pre_ok``, ``pre_error``.

    Re-running skips files already cached. Files that cannot be read get ``pre_ok=False`` and an empty transcript later,
    so one corrupt file can never crash the run or leave a hole in the submission.
    """
    import librosa
    import soundfile as sf
    meta = meta.copy()
    out_paths, oks, errs = [], [], []
    for r in progress(list(meta.itertuples()), desc="preprocess"):
        out = cfg.audio_cache_dir / r.split / f"{r.stem}.wav"
        if out.exists():
            out_paths.append(str(out)); oks.append(True); errs.append("")
            continue
        try:
            if not isinstance(r.path, str) or not os.path.exists(r.path):
                raise FileNotFoundError("audio file not found")
            y, sr = librosa.load(r.path, sr=cfg.SAMPLE_RATE, mono=True)   # resample + mono mixdown
            if len(y) == 0:
                raise ValueError("empty audio")
            y = preprocess_waveform(y, sr, cfg.TRIM_TOP_DB, cfg.TARGET_RMS)
            out.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(out), y, sr, subtype="PCM_16")
            out_paths.append(str(out)); oks.append(True); errs.append("")
        except Exception as e:
            out_paths.append(None); oks.append(False); errs.append(repr(e)[:200])
    meta["cache_path"], meta["pre_ok"], meta["pre_error"] = out_paths, oks, errs
    bad = int((~meta["pre_ok"]).sum())
    print(f"preprocessed {len(meta) - bad}/{len(meta)} files OK" + (f"  ({bad} FAILED -> see pre_error)" if bad else ""))
    return meta
