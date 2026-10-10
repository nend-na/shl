"""Local Whisper transcription (faster-whisper) with a resumable on-disk cache.

Whisper runs ONCE per file. Every record (text + segment + word timestamps + confidences) is appended to
``segments.jsonl`` immediately, so an interrupted run resumes where it stopped and later notebooks only read the cache.
"""
from __future__ import annotations

import json
import os

import pandas as pd

from .utils import get_device, progress


def load_whisper(cfg):
    """Load faster-whisper from a hub size name or a local CTranslate2 folder (see CONFIG['WHISPER_MODEL'])."""
    from faster_whisper import WhisperModel
    device = get_device()
    compute = cfg.WHISPER_COMPUTE_TYPE or ("float16" if device == "cuda" else "int8")
    print(f"Loading Whisper '{cfg.WHISPER_MODEL}' on {device} ({compute})")
    return WhisperModel(cfg.WHISPER_MODEL, device=device, compute_type=compute,
                        download_root=cfg.MODEL_CACHE_DIR, local_files_only=bool(cfg.OFFLINE))


def _load_done(path) -> dict:
    """Read the jsonl cache into {(split, stem): record}. A half-written last line is ignored."""
    done = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done[(rec["split"], rec["stem"])] = rec
                except Exception:
                    continue
    return done


def _transcribe_one(model, audio, cfg) -> dict:
    """Run Whisper on one waveform; return text + segments (+ words with probabilities)."""
    segments, _ = model.transcribe(
        audio, language=cfg.WHISPER_LANGUAGE, beam_size=cfg.WHISPER_BEAM,
        word_timestamps=cfg.WHISPER_WORD_TIMESTAMPS, vad_filter=False,      # no VAD: keep real pauses visible
        condition_on_previous_text=False,                                     # avoids repetition loops
        initial_prompt=cfg.WHISPER_INITIAL_PROMPT)
    segs = []
    for s in segments:
        words = [[w.word, round(float(w.start), 3), round(float(w.end), 3), round(float(w.probability), 4)]
                 for w in (s.words or [])]
        segs.append(dict(start=round(float(s.start), 3), end=round(float(s.end), 3), text=s.text.strip(),
                         avg_logprob=round(float(s.avg_logprob), 4), no_speech_prob=round(float(s.no_speech_prob), 4),
                         compression_ratio=round(float(s.compression_ratio), 3), words=words))
    return dict(text=" ".join(s["text"] for s in segs).strip(), segments=segs)


def transcribe_all(meta: pd.DataFrame, cfg, model=None) -> dict:
    """Transcribe every row of ``meta`` not yet in the cache. ``meta`` needs: split, id, stem, cache_path.

    Returns {(split, stem): record}. Also (re)writes ``transcripts.csv`` (one row per file, no timestamps).
    """
    import soundfile as sf
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    done = _load_done(cfg.segments_jsonl)
    todo = [r for r in meta.itertuples() if (r.split, r.stem) not in done]
    print(f"transcripts cached: {len(done)} | to transcribe now: {len(todo)}")
    if todo:
        model = model or load_whisper(cfg)
        with open(cfg.segments_jsonl, "a") as f:
            for r in progress(todo, desc="whisper"):
                rec = dict(split=r.split, id=str(r.id), stem=r.stem, duration=0.0, ok=False, text="", segments=[])
                if isinstance(r.cache_path, str) and os.path.exists(r.cache_path):
                    try:
                        audio, sr = sf.read(r.cache_path, dtype="float32")
                        rec.update(_transcribe_one(model, audio, cfg), duration=round(len(audio) / sr, 3), ok=True)
                    except Exception as e:
                        rec["error"] = repr(e)[:200]
                f.write(json.dumps(rec) + "\n")
                f.flush()
        done = _load_done(cfg.segments_jsonl)
    export_transcripts_csv(done, cfg)
    return done


def export_transcripts_csv(records: dict, cfg) -> pd.DataFrame:
    """Flat table of transcripts (the file the baseline/final notebooks read)."""
    df = pd.DataFrame([dict(split=r["split"], id=r["id"], stem=r["stem"], duration=r.get("duration", 0.0),
                            ok=r.get("ok", False), n_segments=len(r.get("segments", [])), text=r.get("text", ""))
                       for r in records.values()])
    df.to_csv(cfg.transcripts_csv, index=False)
    return df


def load_transcripts(cfg) -> pd.DataFrame:
    """Read transcripts.csv written by ``transcribe_all`` (NaN text -> empty string)."""
    df = pd.read_csv(cfg.transcripts_csv)
    df["text"] = df["text"].fillna("").astype(str)
    return df


def load_segments(cfg) -> dict:
    """Read the full cache incl. segment/word timestamps: {(split, stem): record}."""
    return _load_done(cfg.segments_jsonl)
