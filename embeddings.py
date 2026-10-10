"""Embeddings, all computed locally: sentence encoder, mean-pooled HF encoder, and WavLM audio embeddings."""
from __future__ import annotations

import numpy as np

from .utils import get_device, progress


def embed_sentence_transformer(texts, model_name: str, cfg) -> np.ndarray:
    """One L2-normalised sentence embedding per transcript (e.g. all-MiniLM-L6-v2 / all-mpnet-base-v2)."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, device=get_device(), cache_folder=cfg.MODEL_CACHE_DIR)
    emb = model.encode([t or "" for t in texts], batch_size=cfg.TEXT_BATCH, show_progress_bar=True,
                       convert_to_numpy=True, normalize_embeddings=True)
    del model
    return emb.astype(np.float32)


def embed_hf_meanpool(texts, model_name: str, cfg, max_length: int = 512) -> np.ndarray:
    """Mean of the last hidden states over real tokens (padding masked out) of a RoBERTa/DeBERTa-style encoder."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    device = get_device()
    tok = AutoTokenizer.from_pretrained(model_name, **cfg.hf_kwargs())
    model = AutoModel.from_pretrained(model_name, **cfg.hf_kwargs()).to(device).eval()
    texts = [t or "" for t in texts]
    out = []
    for i in progress(range(0, len(texts), cfg.TEXT_BATCH), desc=f"encode ({model_name})"):
        batch = tok(texts[i:i + cfg.TEXT_BATCH], padding=True, truncation=True, max_length=max_length,
                    return_tensors="pt").to(device)
        with torch.no_grad():
            h = model(**batch).last_hidden_state                      # (B, T, D)
        m = batch["attention_mask"].unsqueeze(-1).float()
        out.append(((h * m).sum(1) / m.sum(1).clamp(min=1)).cpu().numpy())
    del model
    return np.concatenate(out).astype(np.float32)


def embed_wavlm(paths, cfg) -> np.ndarray:
    """Mean + std over time of WavLM's last hidden state for each preprocessed 16 kHz file (2 x 768 dims).

    Files that are missing/unreadable get the average of the other rows so the matrix has no holes.
    """
    import os
    import soundfile as sf
    import torch
    from transformers import AutoFeatureExtractor, AutoModel
    device = get_device()
    fe = AutoFeatureExtractor.from_pretrained(cfg.AUDIO_ENCODER, **cfg.hf_kwargs())
    model = AutoModel.from_pretrained(cfg.AUDIO_ENCODER, **cfg.hf_kwargs()).to(device).eval()
    feats = []
    for p in progress(list(paths), desc=f"audio ({cfg.AUDIO_ENCODER})"):
        if not (isinstance(p, str) and os.path.exists(p)):
            feats.append(None)
            continue
        try:
            wav, sr = sf.read(p, dtype="float32")
            inp = fe(wav, sampling_rate=sr, return_tensors="pt").input_values.to(device)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
                h = model(inp).last_hidden_state[0].float()           # (T, D)
            feats.append(torch.cat([h.mean(0), h.std(0)]).cpu().numpy())
        except Exception as e:
            print("audio embedding failed for", p, repr(e)[:120])
            feats.append(None)
    good = [f for f in feats if f is not None]
    if not good:
        raise RuntimeError("No audio embedding could be computed.")
    fill = np.mean(good, axis=0)
    del model
    return np.stack([f if f is not None else fill for f in feats]).astype(np.float32)
