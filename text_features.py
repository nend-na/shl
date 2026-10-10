"""Hand-crafted, fully explainable text statistics computed from the Whisper transcript (no models involved)."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

_WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
FILLERS = {"um", "umm", "uh", "uhh", "uhm", "er", "erm", "ah", "hmm", "hm", "mm", "mmm", "eh"}
FILLER_PHRASES = ("you know", "i mean", "kind of", "sort of")
# Subordinating words/connectives: a cheap proxy for "attempts complex structures" (rubric levels 4-5).
SUBORDINATORS = {"because", "although", "though", "while", "whereas", "if", "unless", "since", "which", "whom",
                 "whose", "whether", "however", "therefore", "moreover", "furthermore", "despite", "until",
                 "when", "after", "before", "that"}
_FALSE_START = re.compile(r"--|—|–|\.\.\.|…")


def text_features(text: str) -> dict:
    """Return ~20 numeric features describing length, variety, sentence shape and disfluency of ``text``."""
    text = text or ""
    tokens = [w.lower() for w in _WORD.findall(text)]
    n = len(tokens)
    sents = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if _WORD.search(s)]
    sent_lens = np.array([len(_WORD.findall(s)) for s in sents], float) if sents else np.array([0.0])
    n_sent = len(sents)
    lower = " " + re.sub(r"[^a-z' ]", " ", text.lower()) + " "
    lower = re.sub(r"\s+", " ", lower)
    safe_n = max(n, 1)
    safe_s = max(n_sent, 1)
    uniq = len(set(tokens))
    return dict(
        n_words=float(n),
        unique_ratio=uniq / safe_n,                           # type/token ratio (length dependent)
        root_ttr=uniq / np.sqrt(safe_n),                      # less length dependent lexical variety
        n_sentences=float(n_sent),
        avg_sent_len=float(sent_lens.mean()),
        std_sent_len=float(sent_lens.std()),
        short_sent_ratio=float((sent_lens < 4).mean()) if n_sent else 0.0,   # fragments
        long_sent_ratio=float((sent_lens > 25).mean()) if n_sent else 0.0,   # run-ons
        avg_word_len=float(np.mean([len(t) for t in tokens])) if n else 0.0,
        long_word_ratio=float(np.mean([len(t) >= 7 for t in tokens])) if n else 0.0,
        filler_rate=sum(t in FILLERS for t in tokens) / safe_n,
        filler_phrase_rate=sum(lower.count(" " + p + " ") for p in FILLER_PHRASES) / safe_n,
        repeat_rate=sum(a == b for a, b in zip(tokens, tokens[1:])) / safe_n,            # "the the", "I, I"
        repeat_phrase_rate=sum(tokens[i:i + 2] == tokens[i + 2:i + 4] for i in range(max(n - 3, 0))) / safe_n,
        false_start_rate=len(_FALSE_START.findall(text)) / safe_s,                        # broken-off sentences
        comma_rate=text.count(",") / safe_n,
        subordinator_rate=sum(t in SUBORDINATORS for t in tokens) / safe_s,               # per sentence
        contraction_rate=sum("'" in t for t in tokens) / safe_n,
    )


def text_feature_table(texts) -> pd.DataFrame:
    """DataFrame (one row per text) with columns prefixed ``hand_``."""
    df = pd.DataFrame([text_features(t) for t in texts])
    return df.add_prefix("hand_")
