"""Write submission.csv and validate it against a template CSV (columns + row order).

NOTE for this competition: the leaderboard expects one row per ``test.csv`` row (216). The provided
sample_submission.csv has only 204 rows with different file names, so pass ``test.csv`` as the template."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import stem_of


def write_submission(preds, test_meta: pd.DataFrame, sample_csv: str, out_path: str,  # sample_csv = template (test.csv)
                     target_col: str | None = None) -> pd.DataFrame:
    """Fill the sample submission's score column with ``preds`` (aligned with ``test_meta`` rows).

    The id column of the sample file is the column whose values best overlap the test file stems; the other column
    (or ``target_col``) receives the predictions. Rows keep the sample file's order.
    """
    sample = pd.read_csv(sample_csv)
    if len(sample) != len(test_meta):
        raise ValueError(f"template has {len(sample)} rows but there are {len(test_meta)} test files; "
                         "use test.csv as the template (the grader expects one row per test file).")
    stems = set(test_meta["stem"])
    overlap = {c: sample[c].map(stem_of).isin(stems).mean() for c in sample.columns}
    id_col = max(overlap, key=overlap.get)
    if overlap[id_col] < 0.99:
        raise ValueError(f"Could not match sample_submission ids to test files (overlap per column: {overlap}).")
    if target_col is None:
        others = [c for c in sample.columns if c != id_col]
        if len(others) != 1:
            raise ValueError(f"Cannot tell which column is the score: {list(sample.columns)}; pass target_col.")
        target_col = others[0]
    pred_by_stem = dict(zip(test_meta["stem"], np.clip(np.asarray(preds, float), 0.0, 5.0)))
    sub = sample.copy()
    sub[target_col] = sub[id_col].map(lambda v: pred_by_stem.get(stem_of(v), np.nan)).astype(float)
    validate_submission(sub, sample, id_col, target_col)
    sub.to_csv(out_path, index=False)
    print(f"Wrote {out_path}: {sub.shape[0]} rows, columns {list(sub.columns)} -> validation PASSED")
    return sub


def validate_submission(sub: pd.DataFrame, sample: pd.DataFrame, id_col: str, target_col: str) -> None:
    """Raise AssertionError if the file would be rejected: columns, order, row count, NaNs, range."""
    assert list(sub.columns) == list(sample.columns), f"columns differ: {list(sub.columns)} vs {list(sample.columns)}"
    assert len(sub) == len(sample), f"row count {len(sub)} != {len(sample)}"
    assert (sub[id_col].astype(str).values == sample[id_col].astype(str).values).all(), "id order differs from sample"
    assert not sub[target_col].isna().any(), "NaN in predictions (some test file was not predicted)"
    assert np.isfinite(sub[target_col].to_numpy(float)).all(), "non-finite predictions"
    assert sub[target_col].between(0, 5).all(), "predictions outside [0, 5]"
