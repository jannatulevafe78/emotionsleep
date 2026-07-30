"""
The two analyses the proposal asks for.

  reactivity()   per-subject emotional reactivity from EEG/physio epochs
  relate()       does night-before sleep quality predict next-day reactivity?
                 (n=89 between-subjects, sleep measured FIRST -> temporal precedence)
  reverse()      do mood / depression scores associate with sleep quality?
                 (cross-sectional, same timepoint -> correlational only)

Design note, state this in the paper: one night and one emotion session per
person means the forward path has temporal precedence but is between-subjects,
and the reverse path has no precedence at all. Do not call either one causal.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .ecsmp import EMOTIONS
from .features import BANDS


# ======================================================================
# 1. Emotional reactivity per subject
# ======================================================================
def reactivity(X, y, g, sfreq=128, baseline_class=0):
    """Reactivity = deviation of each emotion's features from that subject's
    own neutral baseline. Within-subject differencing removes the huge
    between-person offsets in absolute EEG power.

    Returns a per-subject dataframe with one column per emotion x band.
    """
    from scipy import signal as sps

    rows = []
    for sub in np.unique(g):
        m = g == sub
        Xs, ys = X[m], y[m]
        if baseline_class not in ys:
            continue
        # band power per epoch, averaged over channels
        def bandpow(A):
            f, p = sps.welch(np.asarray(A, float), fs=sfreq,
                             nperseg=min(A.shape[-1], int(2 * sfreq)), axis=-1)
            out = {}
            for name, (lo, hi) in BANDS.items():
                sel = (f >= lo) & (f < hi)
                out[name] = np.trapezoid(p[..., sel], f[sel], axis=-1).mean(axis=-1)
            return out

        base = bandpow(Xs[ys == baseline_class])
        row = {"subject": int(sub)}
        for ci, emo in enumerate(EMOTIONS):
            if ci == baseline_class or (ys == ci).sum() == 0:
                continue
            cur = bandpow(Xs[ys == ci])
            for band in BANDS:
                b0 = np.mean(base[band]) + 1e-12
                row[f"react_{emo}_{band}"] = float((np.mean(cur[band]) - b0) / b0)
            # overall magnitude of response across bands
            row[f"react_{emo}_mag"] = float(np.mean(
                [abs(row[f"react_{emo}_{b}"]) for b in BANDS]))
        # aggregate reactivity indices
        mags = [v for k, v in row.items() if k.endswith("_mag")]
        row["reactivity_overall"] = float(np.mean(mags)) if mags else np.nan
        neg = [row.get(f"react_{e}_mag") for e in ("fear", "sad", "anger", "disgust")]
        neg = [v for v in neg if v is not None]
        row["reactivity_negative"] = float(np.mean(neg)) if neg else np.nan
        pos = row.get("react_happy_mag")
        row["reactivity_positive"] = float(pos) if pos is not None else np.nan
        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"    reactivity: {df.shape[0]} subjects, {df.shape[1]-1} features")
    return df


# ======================================================================
# 2. Forward path: sleep -> next-day reactivity
# ======================================================================
SLEEP_PREDICTORS = ["sdnn", "rmssd", "pnn50", "lf_hf", "lf_nu", "mean_hr",
                    "psqi_duration", "psqi_efficiency", "psqi_quality",
                    "rmssd_instability", "sdnn_seg_std", "hr_seg_std",
                    "night_hours", "sleep_hours_diary", "psqi_global"]
OUTCOMES = ["reactivity_overall", "reactivity_negative", "reactivity_positive"]


def relate(sleep_df, react_df, scales=None, out_dir="outputs",
           covariates=("age", "gender")):
    """OLS per (sleep predictor, reactivity outcome) pair with FDR correction.

    Partial correlations control for available covariates. Reported alongside
    the raw correlation so the reader can see what the controls did.
    """
    import statsmodels.api as sm
    from statsmodels.stats.multitest import multipletests

    if "hrv_implausible" in sleep_df.columns:
        n0 = len(sleep_df)
        sleep_df = sleep_df[~sleep_df["hrv_implausible"].astype(bool)]
        print(f"    excluded {n0 - len(sleep_df)} nights with implausible HRV")
    df = sleep_df.merge(react_df, on="subject", how="inner")
    if scales is not None:
        cov_cols = ["subject"] + [c for c in scales.columns
                                  if any(k in c for k in covariates)
                                  or "psqi" in c]
        df = df.merge(scales[list(dict.fromkeys(cov_cols))], on="subject", how="left")
    print(f"    merged: {df.shape[0]} subjects")
    if len(df) < 20:
        return {"error": f"only {len(df)} subjects after merge"}

    covs = [c for c in df.columns if any(k in c for k in covariates)]
    covs = [c for c in covs if pd.api.types.is_numeric_dtype(df[c])
            and df[c].notna().sum() > len(df) * 0.7][:3]

    tests = []
    for xv in [c for c in SLEEP_PREDICTORS if c in df.columns]:
        for yv in [c for c in OUTCOMES if c in df.columns]:
            d = df[[xv, yv] + covs].replace([np.inf, -np.inf], np.nan).dropna()
            if len(d) < 20:
                continue
            r, p = stats.pearsonr(d[xv], d[yv])
            rho, prho = stats.spearmanr(d[xv], d[yv])
            rec = {"sleep_predictor": xv, "outcome": yv, "n": int(len(d)),
                   "pearson_r": float(r), "pearson_p": float(p),
                   "spearman_rho": float(rho), "spearman_p": float(prho)}
            if covs:
                Xd = sm.add_constant(d[[xv] + covs].astype(float))
                try:
                    m = sm.OLS(d[yv].astype(float), Xd).fit()
                    rec.update(adj_beta=float(m.params[xv]),
                               adj_p=float(m.pvalues[xv]),
                               adj_r2=float(m.rsquared),
                               covariates=covs)
                except Exception:
                    pass
            tests.append(rec)

    if not tests:
        return {"error": "no testable pairs -- check column names"}

    # ---- split into pre-specified primary family and exploratory rest ----
    PRIMARY_X = ("pnn50", "rmssd", "mean_hr", "psqi_global")
    PRIMARY_Y = ("reactivity_overall", "reactivity_negative")
    EXPECTED = {"pnn50": -1, "rmssd": -1, "mean_hr": +1, "psqi_global": +1}
    for t in tests:
        t["family"] = ("primary"
                       if (t["sleep_predictor"] in PRIMARY_X
                           and t["outcome"] in PRIMARY_Y) else "exploratory")
        exp = EXPECTED.get(t["sleep_predictor"])
        if exp is not None:
            t["direction_as_predicted"] = bool(
                (t["pearson_r"] < 0) if exp < 0 else (t["pearson_r"] > 0))

    prim = [t for t in tests if t.get("family") == "primary"]
    if prim:
        pv = [t.get("adj_p", t["pearson_p"]) for t in prim]
        rj, q, _, _ = multipletests(pv, alpha=0.05, method="fdr_bh")
        for t, qq, rr in zip(prim, q, rj):
            t["q_primary"] = float(qq)
            t["significant_primary"] = bool(rr)

    pvals = [t.get("adj_p", t["pearson_p"]) for t in tests]
    rej, q, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
    for t, qq, rr in zip(tests, q, rej):
        t["q_fdr"] = float(qq)
        t["significant_fdr"] = bool(rr)

    sig = [t for t in tests if t["significant_fdr"]]
    sig_p = [t for t in tests if t.get("significant_primary")]
    n_prim = len([t for t in tests if t.get("family") == "primary"])
    ok_dir = [t for t in tests if t.get("family") == "primary"
              and t.get("direction_as_predicted")]
    res = {"n_subjects": int(len(df)),
           "primary_family_size": n_prim,
           "primary_significant": len(sig_p),
           "primary_directions_as_predicted": f"{len(ok_dir)}/{n_prim}",
           "n_tests": len(tests),
           "n_significant_fdr": len(sig), "covariates_used": covs,
           "tests": sorted(tests, key=lambda t: t["q_fdr"]),
           "direction": "sleep (night N) -> reactivity (day N+1); "
                        "temporal precedence, between-subjects"}
    res["interpretation"] = (
        f"PRIMARY (pre-specified, {n_prim} tests): {len(sig_p)} significant "
        f"after FDR; {len(ok_dir)}/{n_prim} in the predicted direction. "
        f"EXPLORATORY: {len(sig)} of {len(tests)} survive correction across all "
        f"tests. "
        + ("Forward path supported." if sig else
           "No association survives correction. With n~"
           f"{len(df)} this is an informative null, not a failure -- report it. "
           "Published effects here are small (r 0.1-0.3), so n=89 has limited "
           "power to detect them."))

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(out_dir, "relation_sleep_to_emotion.json").write_text(
        json.dumps(res, indent=2, default=str))
    df.to_csv(Path(out_dir) / "subject_level_merged.csv", index=False)
    print(f"\n    PRIMARY family ({n_prim} pre-specified tests):")
    for t in sorted([x for x in tests if x.get("family") == "primary"],
                    key=lambda x: x.get("q_primary", 1)):
        mark = "  *SIG*" if t.get("significant_primary") else ""
        dirn = "as predicted" if t.get("direction_as_predicted") else "OPPOSITE"
        print(f"      {t['sleep_predictor']:14s} -> "
              f"{t['outcome'].replace('reactivity_',''):9s} "
              f"r={t['pearson_r']:+.3f}  p={t['pearson_p']:.4f}  "
              f"q={t.get('q_primary', float('nan')):.4f}  n={t['n']}  "
              f"{dirn}{mark}")
    print(f"\n    directions matching prediction: {len(ok_dir)}/{n_prim}")
    print(f"    all {len(tests)} tests, FDR across everything: "
          f"{len(sig)} significant")
    print("\n    top associations overall:")
    for t in res["tests"][:5]:
        print(f"      {t['sleep_predictor']:20s} -> {t['outcome']:22s} "
              f"r={t['pearson_r']:+.3f} q={t['q_fdr']:.3f}"
              f"{'  *' if t['significant_fdr'] else ''}")
    return res


# ======================================================================
# 3. Reverse path: mood / depression <-> sleep quality
# ======================================================================
def reverse(sleep_df, scales, out_dir="outputs"):
    """Cross-sectional only. No temporal precedence -- say so explicitly."""
    from statsmodels.stats.multitest import multipletests

    if scales is None:
        return {"error": "no scale table available"}
    mood_cols = [c for c in scales.columns
                 if any(k in c for k in ("sds", "bdi", "phq", "depress", "sas",
                                         "gad", "anxiet", "panas", "mood"))
                 and pd.api.types.is_numeric_dtype(scales[c])]
    psqi_cols = [c for c in scales.columns if "psqi" in c or "sleep_quality" in c]
    if not mood_cols:
        return {"error": f"no mood/depression columns found in {list(scales.columns)[:15]}"}

    have = [c for c in mood_cols + psqi_cols if c in sleep_df.columns]
    need = [c for c in mood_cols + psqi_cols if c not in sleep_df.columns]
    df = (sleep_df.merge(scales[["subject"] + need], on="subject", how="inner")
          if need else sleep_df.copy())
    mood_cols = [c for c in mood_cols if c in df.columns]
    psqi_cols = [c for c in psqi_cols if c in df.columns]
    targets = [c for c in ("sdnn", "rmssd", "lf_hf", "mean_hr") if c in df.columns]
    targets += psqi_cols

    tests = []
    for mc in mood_cols:
        for tc in targets:
            d = df[[mc, tc]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(d) < 20:
                continue
            r, p = stats.pearsonr(d[mc], d[tc])
            tests.append({"mood_measure": mc, "sleep_measure": tc,
                          "n": int(len(d)), "pearson_r": float(r),
                          "pearson_p": float(p)})
    if not tests:
        return {"error": "no testable pairs"}
    rej, q, _, _ = multipletests([t["pearson_p"] for t in tests],
                                 alpha=0.05, method="fdr_bh")
    for t, qq, rr in zip(tests, q, rej):
        t["q_fdr"] = float(qq); t["significant_fdr"] = bool(rr)

    sig = [t for t in tests if t["significant_fdr"]]
    sig_p = [t for t in tests if t.get("significant_primary")]
    n_prim = len([t for t in tests if t.get("family") == "primary"])
    ok_dir = [t for t in tests if t.get("family") == "primary"
              and t.get("direction_as_predicted")]
    res = {"n_subjects": int(len(df)),
           "primary_family_size": n_prim,
           "primary_significant": len(sig_p),
           "primary_directions_as_predicted": f"{len(ok_dir)}/{n_prim}",
           "n_tests": len(tests),
           "n_significant_fdr": len(sig),
           "tests": sorted(tests, key=lambda t: t["q_fdr"]),
           "design_caveat": "CROSS-SECTIONAL. Mood and sleep measured at the "
                            "same timepoint. Direction is not identified; this "
                            "cannot support 'emotion causes poor sleep'.",
           "interpretation": (f"{len(sig)}/{len(tests)} associations survive FDR. "
                              "Interpret as association only.")}
    Path(out_dir, "relation_mood_and_sleep.json").write_text(
        json.dumps(res, indent=2, default=str))
    print(f"    reverse (cross-sectional): {len(sig)}/{len(tests)} significant")
    return res


# ======================================================================
# 4. Figures
# ======================================================================
def plot_relation(res, out_path, top=8):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ts = res.get("tests", [])[:top]
    if not ts:
        return
    labs = [f"{t['sleep_predictor'][:14]} → {t['outcome'].replace('reactivity_','')}"
            for t in ts]
    rs = [t["pearson_r"] for t in ts]
    cols = ["#2C7BB6" if t["significant_fdr"] else "#BBBBBB" for t in ts]
    fig, ax = plt.subplots(figsize=(7.5, 0.5 * len(ts) + 2))
    ax.barh(labs, rs, color=cols)
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("Pearson r  (blue = significant after FDR)")
    ax.set_title("Night-before sleep HRV vs next-day emotional reactivity")
    ax.invert_yaxis()
    for i, t in enumerate(ts):
        ax.text(t["pearson_r"], i, f"  q={t['q_fdr']:.3f}", va="center", fontsize=8)
    fig.tight_layout(); fig.savefig(out_path, dpi=200); plt.close(fig)


def plot_scatter(df, xv, yv, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = df[[xv, yv]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < 10:
        return
    r, p = stats.pearsonr(d[xv], d[yv])
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(d[xv], d[yv], s=28, alpha=0.75, color="#2C7BB6")
    z = np.polyfit(d[xv], d[yv], 1)
    xs = np.linspace(d[xv].min(), d[xv].max(), 50)
    ax.plot(xs, np.polyval(z, xs), "r--", lw=1.5)
    ax.set_xlabel(xv); ax.set_ylabel(yv)
    ax.set_title(f"r = {r:.3f}, p = {p:.4g}, n = {len(d)}")
    fig.tight_layout(); fig.savefig(out_path, dpi=200); plt.close(fig)
