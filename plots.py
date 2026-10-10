"""Matplotlib figures used in final.ipynb (matplotlib only, no seaborn). Each function returns the figure."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .metrics import pearson, rmse


def save(fig, out_dir, name: str) -> None:
    """Save a figure as PNG under out_dir/figures."""
    from pathlib import Path
    d = Path(out_dir) / "figures"
    d.mkdir(parents=True, exist_ok=True)
    fig.savefig(d / f"{name}.png", dpi=130, bbox_inches="tight")


def plot_label_distribution(y):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(y, bins=np.arange(-0.25, 5.5, 0.5), edgecolor="k")
    ax.axvline(np.mean(y), color="r", ls="--", label=f"mean {np.mean(y):.2f}")
    ax.set(xlabel="grammar score", ylabel="files", title="Training label distribution")
    ax.legend()
    return fig


def plot_hist(values, title: str, xlabel: str, bins: int = 30):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(pd.Series(values).dropna(), bins=bins, edgecolor="k")
    ax.set(title=title, xlabel=xlabel, ylabel="files")
    return fig


def plot_oof_scatter(y, oof, title: str = "Out-of-fold predictions"):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y, oof, s=10, alpha=0.5)
    ax.plot([0, 5], [0, 5], "r--", label="perfect")
    ax.set(xlim=(0, 5.2), ylim=(0, 5.2), xlabel="true score", ylabel="predicted (OOF)",
           title=f"{title}\nRMSE {rmse(y, oof):.3f} | Pearson {pearson(y, oof):.3f}")
    ax.legend()
    return fig


def plot_residual_hist(y, oof):
    res = np.asarray(oof) - np.asarray(y)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(res, bins=30, edgecolor="k")
    ax.axvline(0, color="r", ls="--")
    ax.set(title=f"Residuals (pred - true): mean {res.mean():.3f}, std {res.std():.3f}", xlabel="residual", ylabel="files")
    return fig


def plot_per_score_error(y, oof):
    """MAE and mean signed error per rounded true score: shows regression to the mean at the extremes."""
    df = pd.DataFrame({"y": np.floor(np.asarray(y) + 0.5).astype(int), "res": np.asarray(oof) - np.asarray(y)})
    g = df.groupby("y").res.agg(mae=lambda r: np.abs(r).mean(), bias="mean", n="size")
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    x = np.arange(len(g))
    ax.bar(x - 0.2, g.mae, 0.4, label="mean abs error")
    ax.bar(x + 0.2, g.bias, 0.4, label="mean signed error (bias)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{i}\n(n={n})" for i, n in zip(g.index, g.n)])
    ax.axhline(0, color="k", lw=0.8)
    ax.set(xlabel="true score (rounded)", title="Error by true score")
    ax.legend()
    return fig


def plot_ablation(results: pd.DataFrame, const_rmse: float):
    """Horizontal bars of overall CV RMSE per experiment; dashed line = constant-mean baseline."""
    r = results.sort_values("rmse_overall", ascending=False)
    fig, ax = plt.subplots(figsize=(8, max(3, 0.28 * len(r))))
    ax.barh(r["name"], r["rmse_overall"])
    ax.axvline(const_rmse, color="r", ls="--", label=f"constant baseline {const_rmse:.3f}")
    for i, (v, p) in enumerate(zip(r["rmse_overall"], r["pearson_overall"])):
        ax.text(v, i, f" r={p:.2f}", va="center", fontsize=7)
    ax.set(xlabel="CV RMSE (lower is better)", title="Ablation: feature group x model")
    ax.legend()
    return fig


def plot_feature_corr(df: pd.DataFrame, y, top: int = 25):
    """Pearson correlation of each engineered feature with the score (train only)."""
    c = df.apply(lambda col: pearson(y, col.fillna(col.median()).to_numpy()) if col.notna().any() else np.nan).dropna()
    c = c.reindex(c.abs().sort_values(ascending=False).index[:top])[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.28 * len(c) + 1))
    ax.barh(c.index, c.values, color=["tab:green" if v > 0 else "tab:red" for v in c.values])
    ax.set(xlabel="Pearson r with grammar score", title=f"Top {len(c)} feature correlations")
    return fig


def plot_importance(names, values, title: str, top: int = 20):
    s = pd.Series(values, index=names)
    s = s.reindex(s.abs().sort_values(ascending=False).index[:top])[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.28 * len(s) + 1))
    ax.barh(s.index, s.values, color=["tab:green" if v >= 0 else "tab:red" for v in s.values])
    ax.set(title=title)
    return fig
