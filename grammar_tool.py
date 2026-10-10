"""OPTIONAL LanguageTool grammar-error counts. Never blocks the pipeline: returns None if it cannot run.

LanguageTool needs Java and a one-off download of its server. We always start the LOCAL server (never a remote one),
so transcripts stay on this machine. With Kaggle internet OFF the download usually fails -> feature skipped.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import progress


def languagetool_features(texts, enabled: bool = True) -> pd.DataFrame | None:
    """Columns prefixed ``lt_``: error / grammar-error counts and per-100-word rates. None if unavailable."""
    if not enabled:
        print("[LanguageTool] disabled in CONFIG -> skipped.")
        return None
    try:
        import language_tool_python
        tool = language_tool_python.LanguageTool("en-US")      # local Java server, no remote_server argument
    except Exception as e:
        print("[LanguageTool] UNAVAILABLE -> feature group skipped (pipeline continues without it).")
        print("   needs: `pip install language-tool-python`, Java, and a first-run download. Reason:", repr(e)[:300])
        return None
    rows = []
    for t in progress(list(texts), desc="LanguageTool"):
        t = t or ""
        n_words = max(len(t.split()), 1)
        try:
            matches = tool.check(t)
        except Exception:
            rows.append(dict(err_count=np.nan, err_rate100=np.nan, grammar_count=np.nan, grammar_rate100=np.nan))
            continue
        grammar = [m for m in matches
                   if str(getattr(m, "category", "")).upper() == "GRAMMAR"
                   or str(getattr(m, "ruleIssueType", "")).lower() == "grammar"]
        rows.append(dict(err_count=float(len(matches)), err_rate100=100.0 * len(matches) / n_words,
                         grammar_count=float(len(grammar)), grammar_rate100=100.0 * len(grammar) / n_words))
    try:
        tool.close()
    except Exception:
        pass
    return pd.DataFrame(rows).add_prefix("lt_")
