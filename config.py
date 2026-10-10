"""Single source of truth for paths, model names and switches.

Every notebook builds ``Config(**CONFIG)`` from the dict in its first code cell. Anything you do not
override keeps the default below. Heavy libraries are never imported here so this file is cheap to load.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class Config:
    # ------------------------------------------------------------------ data locations
    DATA_DIR: str = "/kaggle/input/<competition-folder>"
    TRAIN_CSV: str | None = None          # None -> auto-discover under DATA_DIR
    TEST_CSV: str | None = None
    SAMPLE_SUB_CSV: str | None = None
    TRAIN_AUDIO_DIR: str | None = None
    TEST_AUDIO_DIR: str | None = None
    ID_COL: str | None = None             # file-name column in train.csv / test.csv
    LABEL_COL: str | None = None          # numeric grammar score column in train.csv

    # ------------------------------------------------------------------ outputs
    CACHE_DIR: str = "/kaggle/working/cache"
    OUTPUT_DIR: str = "/kaggle/working"

    # ------------------------------------------------------------------ models (hub name OR local path)
    OFFLINE: bool = False                 # True -> never contact the internet (use local paths below)
    MODEL_CACHE_DIR: str | None = None
    WHISPER_MODEL: str = "small"          # faster-whisper size, or folder of a CTranslate2 Whisper model
    WHISPER_COMPUTE_TYPE: str | None = None   # None -> float16 on GPU, int8 on CPU
    WHISPER_BEAM: int = 5
    WHISPER_LANGUAGE: str = "en"
    WHISPER_WORD_TIMESTAMPS: bool = True  # word-level times -> better pause features (a bit slower)
    WHISPER_INITIAL_PROMPT: str | None = None  # e.g. a disfluent sentence to make Whisper keep "um"/"uh"
    BASELINE_TEXT_ENCODER: str = "sentence-transformers/all-MiniLM-L6-v2"
    TEXT_ENCODER: str = "sentence-transformers/all-mpnet-base-v2"
    HF_ENCODER: str = "roberta-base"      # mean-pooled hidden states (try microsoft/deberta-v3-base)
    LM_NAME: str = "gpt2"                 # perplexity model
    AUDIO_ENCODER: str = "microsoft/wavlm-base-plus"
    LLM_NAME: str = "Qwen/Qwen2.5-1.5B-Instruct"   # optional zero-shot rubric scorer

    # ------------------------------------------------------------------ switches
    RUN_LANGUAGETOOL: bool = True         # still skipped automatically if Java / download unavailable
    RUN_LLM_RUBRIC: bool = False          # optional; keep only if it improves CV
    FORCE_RECOMPUTE: bool = False         # ignore cached feature files

    # ------------------------------------------------------------------ preprocessing
    SAMPLE_RATE: int = 16000
    TRIM_TOP_DB: float = 30.0             # silence threshold (dB below peak) for edge trimming
    TARGET_RMS: float = 0.05              # loudness normalisation target (linear RMS)

    # ------------------------------------------------------------------ fluency
    PAUSE_THR: float = 0.30               # seconds; a gap between words >= this counts as a pause
    LONG_PAUSE_THR: float = 1.00

    # ------------------------------------------------------------------ batch sizes
    TEXT_BATCH: int = 32
    LM_MAX_TOKENS: int = 1023

    # ------------------------------------------------------------------ reproducibility / CV
    SEED: int = 42
    CV_SPLITS: int = 5
    CV_REPEATS: int = 3

    # ------------------------------------------------------------------ derived paths
    @property
    def cache_dir(self) -> Path:
        return Path(self.CACHE_DIR)

    @property
    def audio_cache_dir(self) -> Path:
        return self.cache_dir / "audio16k"

    @property
    def segments_jsonl(self) -> Path:
        return self.cache_dir / "segments.jsonl"

    @property
    def transcripts_csv(self) -> Path:
        return self.cache_dir / "transcripts.csv"

    @property
    def features_dir(self) -> Path:
        return self.cache_dir / "features"

    @property
    def output_dir(self) -> Path:
        return Path(self.OUTPUT_DIR)

    # ------------------------------------------------------------------ helpers
    def apply_env(self) -> "Config":
        """Set environment variables. Call BEFORE importing torch/transformers/huggingface_hub."""
        os.environ["PYTHONHASHSEED"] = str(self.SEED)
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        if self.OFFLINE:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        for d in (self.cache_dir, self.features_dir, self.output_dir):
            d.mkdir(parents=True, exist_ok=True)
        return self

    def seed_everything(self) -> "Config":
        """Fix python / numpy / torch seeds (torch is optional)."""
        random.seed(self.SEED)
        try:
            import numpy as np
            np.random.seed(self.SEED)
        except Exception:
            pass
        try:
            import torch
            torch.manual_seed(self.SEED)
            torch.cuda.manual_seed_all(self.SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass
        return self

    def hf_kwargs(self) -> dict:
        """Keyword args for ``from_pretrained`` so hub names and local paths both work."""
        return dict(cache_dir=self.MODEL_CACHE_DIR, local_files_only=bool(self.OFFLINE))

    def summary(self) -> str:
        return "\n".join(f"  {k:24s} = {v}" for k, v in asdict(self).items())
