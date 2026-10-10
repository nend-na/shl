"""Language-model perplexity of the transcript (GPT-2 by default), computed locally.

Idea: grammatical, fluent English is more predictable to a language model than broken English, so lower
per-token loss ~ better grammar. Limitation: Whisper tends to 'repair' speech, and perplexity also reflects topic/vocab.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import get_device, progress


def lm_features(texts, cfg) -> pd.DataFrame:
    """Per-text mean/std/90th-percentile token NLL, perplexity and token count (columns prefixed ``lm_``)."""
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = get_device()
    tok = AutoTokenizer.from_pretrained(cfg.LM_NAME, **cfg.hf_kwargs())
    model = AutoModelForCausalLM.from_pretrained(cfg.LM_NAME, **cfg.hf_kwargs()).to(device).eval()
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    rows = []
    for t in progress(list(texts), desc=f"LM ({cfg.LM_NAME})"):
        ids = tok(t or "", add_special_tokens=False, truncation=True, max_length=cfg.LM_MAX_TOKENS)["input_ids"]
        if len(ids) < 2:
            rows.append(dict(nll_mean=np.nan, nll_std=np.nan, nll_p90=np.nan, ppl=np.nan, n_tokens=float(len(ids))))
            continue
        x = torch.tensor([[bos] + ids], device=device)
        with torch.no_grad():
            logits = model(x).logits[0, :-1].float()
        nll = F.cross_entropy(logits, x[0, 1:], reduction="none").cpu().numpy()      # loss of each real token
        rows.append(dict(nll_mean=float(nll.mean()), nll_std=float(nll.std()), nll_p90=float(np.percentile(nll, 90)),
                         ppl=float(np.exp(min(nll.mean(), 20.0))), n_tokens=float(len(ids))))
    del model
    return pd.DataFrame(rows).add_prefix("lm_")
