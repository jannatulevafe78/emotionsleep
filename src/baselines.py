"""
Classical baselines on the engineered feature table.

Why bother: on EEG these are frequently competitive with deep nets, and a
reviewer will ask. If your fusion transformer cannot beat LightGBM on
handcrafted band powers, that is a finding you need to know before submitting,
not after. They also give you a clean SHAP feature-importance figure.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .train import compute_metrics


def _make(name, n_classes, seed=42):
    if name == "logreg":
        return Pipeline([("sc", StandardScaler()),
                         ("m", LogisticRegression(max_iter=2000, C=1.0,
                                                  class_weight="balanced"))])
    if name == "rf":
        return RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                      class_weight="balanced_subsample",
                                      n_jobs=-1, random_state=seed)
    if name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8,
                             tree_method="hist", n_jobs=-1,
                             random_state=seed, eval_metric="mlogloss")
    if name == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=600, num_leaves=63,
                              learning_rate=0.05, subsample=0.8,
                              colsample_bytree=0.8, class_weight="balanced",
                              n_jobs=-1, random_state=seed, verbose=-1)
    raise KeyError(name)


def run_baseline(F, y, groups, name, n_classes, cfg, tag="run", verbose=True):
    scheme = cfg["cv"]["scheme"]
    sp = (LeaveOneGroupOut() if scheme == "loso"
          else GroupKFold(n_splits=cfg["cv"]["n_folds"]))
    folds = []
    oof_p = np.zeros((len(y), n_classes), np.float32)
    oof_y = np.zeros(len(y), np.int64)

    for k, (tr, va) in enumerate(sp.split(F, y, groups)):
        clf = _make(name, n_classes)
        clf.fit(F[tr], y[tr])
        prob = clf.predict_proba(F[va])
        # guard against a class missing from a fold's training data
        if prob.shape[1] != n_classes:
            full = np.zeros((len(va), n_classes), np.float32)
            for i, c in enumerate(np.unique(y[tr])):
                full[:, c] = prob[:, i]
            prob = full
        pred = prob.argmax(1)
        oof_p[va], oof_y[va] = prob, pred
        met = compute_metrics(y[va], pred, prob, n_classes)
        met["fold"] = k
        folds.append(met)
        if verbose:
            print(f"  [{name}] fold {k}: acc {met['accuracy']:.4f} "
                  f"macroF1 {met['balanced_f1_macro']:.4f}")

    keys = ["accuracy", "balanced_f1_macro", "precision_macro",
            "recall_macro", "kappa", "roc_auc", "pr_auc"]
    summary = {k: {"mean": float(np.nanmean([f[k] for f in folds])),
                   "std": float(np.nanstd([f[k] for f in folds]))} for k in keys}
    res = {"tag": tag, "model": name, "scheme": scheme, "summary": summary,
           "folds": folds, "oof": compute_metrics(y, oof_y, oof_p, n_classes)}
    out = Path(cfg["paths"]["out"]) / f"cv_{tag}_{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    if verbose:
        print(f"  == {name}: acc {summary['accuracy']['mean']:.4f} "
              f"+/- {summary['accuracy']['std']:.4f}")
    return res, oof_p


def shap_importance(F, y, groups, feat_names, n_classes, out_dir,
                    model="rf", max_display=25, sample=2000, seed=42):
    """SHAP on a model fitted to all data. Interpretation only -- never quote
    performance from this fit."""
    try:
        import shap
    except ImportError:
        print("  shap not installed; skipping (pip install shap)")
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(F), min(sample, len(F)), replace=False)
    clf = _make(model, n_classes)
    clf.fit(F, y)
    expl = shap.TreeExplainer(clf)
    sv = expl.shap_values(F[idx])
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    plt.figure()
    shap.summary_plot(sv, F[idx], feature_names=feat_names,
                      max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "shap_summary.png", dpi=200)
    plt.close()

    arr = np.abs(np.array(sv))
    imp = arr.mean(axis=tuple(range(arr.ndim - 1))) if arr.ndim > 2 else np.abs(sv).mean(0)
    order = np.argsort(imp)[::-1]
    ranking = [{"feature": feat_names[i], "mean_abs_shap": float(imp[i])}
               for i in order[:50]]
    (Path(out_dir) / "shap_ranking.json").write_text(json.dumps(ranking, indent=2))
    print(f"  shap -> {out_dir}/shap_summary.png")
    return ranking
