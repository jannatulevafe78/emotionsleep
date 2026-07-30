"""
Statistics, explainability and publication figures.

The statistics module matters more than the models for review. Comparing two
models by a single accuracy number is not evidence. What follows produces
per-fold paired tests, effect sizes and confidence intervals -- the things a
methods reviewer actually checks.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats


# ======================================================================
# 1. Statistics
# ======================================================================
def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    s = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / (s + 1e-12))


def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    gt = sum((x > y) for x in a for y in b)
    lt = sum((x < y) for x in a for y in b)
    return float((gt - lt) / (len(a) * len(b)))


def bootstrap_ci(x, n=10000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    bs = rng.choice(x, (n, len(x)), replace=True).mean(1)
    return float(np.percentile(bs, 100 * alpha / 2)), float(np.percentile(bs, 100 * (1 - alpha / 2)))


def compare_models(results: dict, metric="balanced_f1_macro", out_dir="outputs"):
    """results: {model_name: cv_result_dict}. Paired across folds."""
    names = list(results)
    scores = {n: np.array([f[metric] for f in results[n]["folds"]]) for n in names}
    report = {"metric": metric, "per_model": {}, "pairwise": [], "omnibus": {}}

    for n in names:
        s = scores[n]
        lo, hi = bootstrap_ci(s)
        report["per_model"][n] = {"mean": float(s.mean()), "std": float(s.std()),
                                  "ci95": [lo, hi], "folds": s.tolist()}

    # omnibus across all models (Friedman needs >=3 models, >=2 folds)
    if len(names) >= 3:
        mat = np.stack([scores[n] for n in names])
        try:
            chi2, p = stats.friedmanchisquare(*mat)
            report["omnibus"] = {"test": "friedman", "statistic": float(chi2),
                                 "p_value": float(p)}
        except Exception as e:
            report["omnibus"] = {"error": str(e)}

    # pairwise paired tests with Holm correction
    raw = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = scores[names[i]], scores[names[j]]
            try:
                t_p = stats.ttest_rel(a, b).pvalue
            except Exception:
                t_p = float("nan")
            try:
                w_p = stats.wilcoxon(a, b).pvalue
            except Exception:
                w_p = float("nan")
            raw.append({"a": names[i], "b": names[j],
                        "mean_diff": float(a.mean() - b.mean()),
                        "ttest_p": float(t_p), "wilcoxon_p": float(w_p),
                        "cohens_d": cohens_d(a, b),
                        "cliffs_delta": cliffs_delta(a, b)})
    # Holm-Bonferroni on the t-test p-values
    order = np.argsort([r["ttest_p"] if np.isfinite(r["ttest_p"]) else 1.0 for r in raw])
    m = len(raw)
    prev = 0.0
    for rank, k in enumerate(order):
        p = raw[k]["ttest_p"]
        adj = min(1.0, max(prev, (m - rank) * p)) if np.isfinite(p) else float("nan")
        raw[k]["ttest_p_holm"] = float(adj)
        prev = adj if np.isfinite(adj) else prev
    report["pairwise"] = raw

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / f"stats_{metric}.json").write_text(json.dumps(report, indent=2))
    return report


# ======================================================================
# 2. Figures
# ======================================================================
def plot_confusion(cm, classes, path, normalize=True, title=""):
    cm = np.asarray(cm, float)
    disp = cm / np.clip(cm.sum(1, keepdims=True), 1, None) if normalize else cm
    fig, ax = plt.subplots(figsize=(1.3 * len(classes) + 2, 1.15 * len(classes) + 2))
    im = ax.imshow(disp, cmap="Blues", vmin=0, vmax=disp.max())
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    if title:
        ax.set_title(title)
    for i in range(len(classes)):
        for j in range(len(classes)):
            v = disp[i, j]
            ax.text(j, i, f"{v:.2f}" if normalize else f"{int(v)}",
                    ha="center", va="center",
                    color="white" if v > disp.max() * 0.55 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def plot_model_comparison(report, path, metric="macro F1"):
    names = list(report["per_model"])
    means = [report["per_model"][n]["mean"] for n in names]
    ci = np.array([report["per_model"][n]["ci95"] for n in names])
    err = np.abs(ci.T - np.array(means))
    order = np.argsort(means)[::-1]
    fig, ax = plt.subplots(figsize=(1.1 * len(names) + 3, 4))
    ax.bar(range(len(names)), [means[i] for i in order],
           yerr=err[:, order], capsize=5, color="#4C72B0")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([names[i] for i in order], rotation=30, ha="right")
    ax.set_ylabel(metric); ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(f"{metric} by model (mean, 95% bootstrap CI)")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def plot_tsne(emb, y, classes, path, seed=42, perplexity=30, sample=4000):
    from sklearn.manifold import TSNE
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(emb), min(sample, len(emb)), replace=False)
    z = TSNE(2, perplexity=min(perplexity, len(idx) // 4), init="pca",
             random_state=seed).fit_transform(emb[idx])
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for c in np.unique(y[idx]):
        m = y[idx] == c
        ax.scatter(z[m, 0], z[m, 1], s=6, alpha=0.6,
                   label=classes[c] if c < len(classes) else str(c))
    ax.legend(markerscale=2, fontsize=9); ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.set_title("Learned representation")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def plot_curves(hist, path):
    ep = [h["epoch"] for h in hist]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(ep, [h["train_loss"] for h in hist]); ax[0].set_title("Train loss")
    ax[0].set_xlabel("Epoch"); ax[0].grid(alpha=0.3)
    ax[1].plot(ep, [h["val_f1"] for h in hist], color="#DD8452")
    ax[1].set_title("Validation macro-F1"); ax[1].set_xlabel("Epoch"); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def plot_adjacency(A, path, ch_names=None):
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(A, cmap="viridis")
    if ch_names is not None and len(ch_names) <= 32:
        ax.set_xticks(range(len(ch_names))); ax.set_xticklabels(ch_names, rotation=90, fontsize=7)
        ax.set_yticks(range(len(ch_names))); ax.set_yticklabels(ch_names, fontsize=7)
    ax.set_title("Learned channel adjacency")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


# ======================================================================
# 3. Deep-model explainability
# ======================================================================
def saliency(model, x, target=None, device="cpu"):
    """Vanilla gradient saliency. x: (B, C, T) tensor."""
    import torch
    model.eval()
    x = x.clone().to(device).requires_grad_(True)
    out = model(x)
    if target is None:
        target = out.argmax(1)
    out.gather(1, target.view(-1, 1)).sum().backward()
    return x.grad.detach().abs().cpu().numpy()


def integrated_gradients(model, x, target=None, steps=32, device="cpu"):
    """IG against a zero baseline -- the standard attribution for time series."""
    import torch
    model.eval()
    x = x.to(device)
    base = torch.zeros_like(x)
    if target is None:
        with torch.no_grad():
            target = model(x).argmax(1)
    total = torch.zeros_like(x)
    for a in torch.linspace(1.0 / steps, 1.0, steps):
        xi = (base + a * (x - base)).requires_grad_(True)
        out = model(xi)
        out.gather(1, target.view(-1, 1)).sum().backward()
        total += xi.grad.detach()
    return ((x - base) * total / steps).abs().cpu().numpy()


def plot_attribution(x, attr, path, sfreq=100, ch_names=None, max_ch=4):
    """Overlay attribution intensity on the raw trace."""
    C = min(x.shape[0], max_ch)
    t = np.arange(x.shape[-1]) / sfreq
    fig, axes = plt.subplots(C, 1, figsize=(11, 1.9 * C), sharex=True)
    axes = np.atleast_1d(axes)
    a = attr / (attr.max() + 1e-12)
    for c in range(C):
        axes[c].plot(t, x[c], lw=0.6, color="#333")
        axes[c].scatter(t, x[c], c=a[c], cmap="hot", s=2.5, alpha=0.85)
        axes[c].set_ylabel(ch_names[c] if ch_names else f"ch{c}", fontsize=9)
        axes[c].grid(alpha=0.2)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Gradient attribution over raw EEG", fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def band_attribution(attr, sfreq=100):
    """Project time-domain attribution into frequency bands -- far more
    interpretable in a paper than a raw saliency trace."""
    from scipy import signal as sps
    from .features import BANDS
    f, p = sps.welch(attr, fs=sfreq, nperseg=min(attr.shape[-1], int(4 * sfreq)), axis=-1)
    out = {}
    for name, (lo, hi) in BANDS.items():
        m = (f >= lo) & (f < hi)
        out[name] = float(np.trapezoid(p[..., m], f[m], axis=-1).mean())
    tot = sum(out.values()) + 1e-12
    return {k: v / tot for k, v in out.items()}
