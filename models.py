"""Models, cross-validation and stacking.

Leakage rules enforced here:
  * every imputer / scaler / ridge-alpha search lives INSIDE a sklearn Pipeline that is re-fit on the training part of
    each fold, so nothing is learned from validation rows;
  * no early stopping, no feature selection;
  * test rows are only ever passed to ``predict``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .metrics import clip_scores, pearson, rmse


# ----------------------------------------------------------------------------- model factories
def make_constant():
    """'Learned nothing' baseline: always predict the training-fold mean."""
    return DummyRegressor(strategy="mean")


def make_ridge():
    """Median-impute -> standardise -> Ridge whose alpha is picked by efficient leave-one-out CV on the TRAIN part only."""
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                     ("m", RidgeCV(alphas=np.logspace(-1, 5, 25)))])


def make_svr():
    """Median-impute -> standardise -> RBF support-vector regression with fixed, conservative hyper-parameters."""
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                     ("m", SVR(kernel="rbf", C=1.0, epsilon=0.1, gamma="scale"))])


def make_lgbm(seed: int = 42):
    """Small gradient-boosted trees (shallow, strongly regularised: only ~770 training rows). Falls back to sklearn."""
    try:
        from lightgbm import LGBMRegressor
        return LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=8, min_child_samples=15,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=2.0,
                             random_state=seed, n_jobs=4, verbose=-1)
    except Exception:
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.03, max_leaf_nodes=8, min_samples_leaf=15,
                                             l2_regularization=2.0, random_state=seed)


# ----------------------------------------------------------------------------- folds
def stratify_labels(y, n_splits: int) -> np.ndarray:
    """Round labels to the nearest integer for stratification; merge any class smaller than n_splits into its neighbour."""
    s = np.clip(np.floor(np.asarray(y, float) + 0.5), 0, 5).astype(int)
    while True:
        vals, counts = np.unique(s, return_counts=True)
        if len(vals) == 1 or counts.min() >= n_splits:
            return s
        small = vals[counts.argmin()]
        others = vals[vals != small]
        s[s == small] = others[np.abs(others - small).argmin()]


# ----------------------------------------------------------------------------- cross-validation
def run_cv(factory: Callable, X, y, strat, seeds, n_splits: int = 5, name: str = "model", verbose: bool = True) -> dict:
    """Repeated stratified K-fold. Prints RMSE and Pearson for EVERY fold and overall.

    Returns dict(name, oof_repeats (R,n), oof (n,) = mean over repeats, folds DataFrame, summary dict).
    'Overall' = metric of the complete out-of-fold vector of one repeat, averaged over repeats.
    """
    X, y = np.asarray(X, float), np.asarray(y, float)
    oof_rep = np.zeros((len(seeds), len(y)))
    rows = []
    for r, seed in enumerate(seeds):
        for k, (tr, va) in enumerate(StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X, strat)):
            model = factory()
            model.fit(X[tr], y[tr])
            p = clip_scores(model.predict(X[va]))
            oof_rep[r, va] = p
            row = dict(repeat=r, fold=k, rmse=rmse(y[va], p), pearson=pearson(y[va], p))
            rows.append(row)
            if verbose:
                print(f"[{name}] repeat {r} fold {k} | RMSE {row['rmse']:.4f} | Pearson {row['pearson']:.4f}")
    folds = pd.DataFrame(rows)
    overall_rmse = float(np.mean([rmse(y, o) for o in oof_rep]))
    overall_pear = float(np.nanmean([pearson(y, o) for o in oof_rep]))
    summary = dict(name=name, rmse_overall=overall_rmse, pearson_overall=overall_pear,
                   rmse_fold_mean=float(folds.rmse.mean()), rmse_fold_std=float(folds.rmse.std()),
                   pearson_fold_mean=float(folds.pearson.mean()), pearson_fold_std=float(folds.pearson.std()))
    if verbose:
        print(f"[{name}] OVERALL (mean of {len(seeds)} repeat(s)) | RMSE {overall_rmse:.4f} | Pearson {overall_pear:.4f}\n")
    return dict(name=name, oof_repeats=oof_rep, oof=oof_rep.mean(0), folds=folds, summary=summary)


@dataclass
class Experiment:
    """One row of the ablation table: which features, which model, and the train/test matrices."""
    name: str
    features: str
    model: str
    factory: Callable
    X_train: np.ndarray
    X_test: np.ndarray


def run_experiments(experiments, y, strat, seeds, n_splits: int = 5, verbose: bool = True):
    """Run CV for each experiment. Returns (results DataFrame sorted by RMSE, {name: cv result})."""
    cv, rows = {}, []
    for e in experiments:
        res = run_cv(e.factory, e.X_train, y, strat, seeds, n_splits, e.name, verbose)
        cv[e.name] = res
        rows.append(dict(res["summary"], features=e.features, model=e.model, n_features=e.X_train.shape[1]))
    table = pd.DataFrame(rows).sort_values("rmse_overall").reset_index(drop=True)
    return table, cv


# ----------------------------------------------------------------------------- stacking
def fit_stack(P, y) -> LinearRegression:
    """Non-negative linear stack (weights >= 0 + intercept) of base-model predictions P (n, k)."""
    return LinearRegression(positive=True).fit(np.asarray(P, float), np.asarray(y, float))


def stack_cv(P, y, strat, seeds, n_splits: int = 5, verbose: bool = True) -> dict:
    """Honest estimate of the stack: weights are fit on part of the OOF matrix and scored on the held-out part."""
    P, y = np.asarray(P, float), np.asarray(y, float)
    oof = np.zeros((len(seeds), len(y)))
    rows = []
    for r, seed in enumerate(seeds):
        for k, (tr, va) in enumerate(StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(P, strat)):
            p = clip_scores(fit_stack(P[tr], y[tr]).predict(P[va]))
            oof[r, va] = p
            rows.append(dict(repeat=r, fold=k, rmse=rmse(y[va], p), pearson=pearson(y[va], p)))
            if verbose:
                print(f"[stack] repeat {r} fold {k} | RMSE {rows[-1]['rmse']:.4f} | Pearson {rows[-1]['pearson']:.4f}")
    folds = pd.DataFrame(rows)
    out = dict(oof=oof.mean(0), folds=folds,
               rmse_overall=float(np.mean([rmse(y, o) for o in oof])),
               pearson_overall=float(np.nanmean([pearson(y, o) for o in oof])))
    if verbose:
        print(f"[stack] OVERALL | RMSE {out['rmse_overall']:.4f} | Pearson {out['pearson_overall']:.4f}\n")
    return out
