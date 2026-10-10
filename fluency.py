"""Fluency features from Whisper timestamps: speech rate, pauses, and how confident the ASR was.

Pauses are gaps between consecutive WORDS (or between segments if word timestamps are missing) that are at least
PAUSE_THR seconds. Whisper's word times are approximate, so treat pause measures as a noisy proxy, not ground truth.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def fluency_features(rec: dict, pause_thr: float = 0.30, long_pause_thr: float = 1.0) -> dict:
    """Features for one cached record (see transcribe._transcribe_one for the record layout)."""
    segs = rec.get("segments", [])
    words = [w for s in segs for w in s.get("words", [])]               # [word, start, end, prob]
    duration = float(rec.get("duration", 0.0)) or np.nan
    out = dict(audio_duration=duration, n_segments=float(len(segs)))
    if words:
        starts = np.array([w[1] for w in words], float)
        ends = np.array([w[2] for w in words], float)
        probs = np.array([w[3] for w in words], float)
        n_words = len(words)
    elif segs:                                                          # fall back to segment level
        starts = np.array([s["start"] for s in segs], float)
        ends = np.array([s["end"] for s in segs], float)
        probs = np.array([], float)
        n_words = len(" ".join(s["text"] for s in segs).split())
    else:
        return {**out, **{k: np.nan for k in (
            "speech_rate_wps", "articulation_rate_wps", "speech_span", "pause_count", "pause_mean", "pause_max",
            "long_pause_count", "pause_ratio", "pauses_per_min", "mean_run_words", "word_prob_mean",
            "word_prob_low_ratio", "seg_avg_logprob_mean", "seg_avg_logprob_min", "seg_no_speech_mean")}}
    gaps = np.clip(starts[1:] - ends[:-1], 0.0, None)
    pauses = gaps[gaps >= pause_thr]
    span = max(float(ends[-1] - starts[0]), 1e-6)
    pause_total = float(pauses.sum())
    out.update(
        speech_rate_wps=n_words / duration if duration == duration and duration > 0 else np.nan,
        articulation_rate_wps=n_words / max(span - pause_total, 1e-6),          # speed while actually talking
        speech_span=span,
        pause_count=float(len(pauses)),
        pause_mean=float(pauses.mean()) if len(pauses) else 0.0,
        pause_max=float(pauses.max()) if len(pauses) else 0.0,
        long_pause_count=float((gaps >= long_pause_thr).sum()),
        pause_ratio=pause_total / span,
        pauses_per_min=len(pauses) / (span / 60.0),
        mean_run_words=n_words / (len(pauses) + 1.0),                           # words between pauses
        word_prob_mean=float(probs.mean()) if len(probs) else np.nan,           # ASR confidence ~ intelligibility
        word_prob_low_ratio=float((probs < 0.5).mean()) if len(probs) else np.nan,
        seg_avg_logprob_mean=float(np.mean([s["avg_logprob"] for s in segs])),
        seg_avg_logprob_min=float(np.min([s["avg_logprob"] for s in segs])),
        seg_no_speech_mean=float(np.mean([s["no_speech_prob"] for s in segs])),
    )
    return out


def fluency_table(records, pause_thr: float, long_pause_thr: float) -> pd.DataFrame:
    """DataFrame (one row per record, in the given order) with columns prefixed ``flu_``."""
    return pd.DataFrame([fluency_features(r, pause_thr, long_pause_thr) for r in records]).add_prefix("flu_")
