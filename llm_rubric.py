"""OPTIONAL zero-shot feature: a local open LLM reads the transcript and rates it 1-5 against the task rubric.

The score is the expectation over the model's next-token probabilities for the digits 1..5 (a smooth number, not
a hard label). No labels are used, so there is no leakage; keep it only if it improves cross-validated RMSE.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import get_device, progress

RUBRIC = """Score the speaker's GRAMMAR from 1 to 5 using this rubric.
1: struggles with sentence structure/syntax, limited control of simple grammar, relies on memorized patterns.
2: limited understanding; simple structures with consistent basic mistakes; may leave sentences incomplete.
3: decent grasp of sentence structure but with grammatical errors, or decent grammar but syntax/structure errors.
4: strong understanding, consistent good control; occasional minor errors that do not cause misunderstanding; can self-correct.
5: high accuracy, adept control of complex grammar, seldom noticeable mistakes, handles complex structures, self-corrects when necessary."""


def llm_rubric_features(texts, cfg) -> pd.DataFrame | None:
    """Column ``llm_expected_score`` (1..5, continuous). Returns None (with a message) if the LLM cannot be run."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        device = get_device()
        tok = AutoTokenizer.from_pretrained(cfg.LLM_NAME, **cfg.hf_kwargs())
        dtype = torch.float16 if device == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(cfg.LLM_NAME, torch_dtype=dtype, **cfg.hf_kwargs()).to(device).eval()
        digit_ids = []
        for d in "12345":
            ids = tok.encode(d, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f"digit {d!r} is not a single token for this tokenizer")
            digit_ids.append(ids[0])
    except Exception as e:
        print("[LLM rubric] UNAVAILABLE -> skipped. Reason:", repr(e)[:300])
        return None
    scores = []
    for t in progress(list(texts), desc=f"LLM rubric ({cfg.LLM_NAME})"):
        words = (t or "").split()[:600]
        msgs = [{"role": "system", "content": "You are a strict, consistent English-speaking examiner."},
                {"role": "user", "content": f"{RUBRIC}\n\nTranscript of the speech:\n\"\"\"\n{' '.join(words)}\n\"\"\"\n\n"
                                            "Reply with a single digit from 1 to 5 and nothing else."}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        x = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**x).logits[0, -1].float()
        p = torch.softmax(logits[digit_ids], dim=0).cpu().numpy()
        scores.append(float((p * np.arange(1, 6)).sum()))
    del model
    return pd.DataFrame({"llm_expected_score": scores})
