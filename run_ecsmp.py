#!/usr/bin/env python
"""
ECSMP pipeline.

    python run_ecsmp.py inspect --root <ECSMP folder>    <-- ALWAYS RUN FIRST
    python run_ecsmp.py prep    --root <ECSMP folder>
    python run_ecsmp.py relate                           <-- the research question
    python run_ecsmp.py classify                         <-- 6-class emotion recognition
    python run_ecsmp.py report
    python run_ecsmp.py all     --root <ECSMP folder>

`inspect` writes outputs/ecsmp_inventory.json describing every file it found and
how it classified it. Read that before running anything else -- if the emotion or
sleep counts look wrong, the fix is a one-line pattern edit in src/ecsmp.py.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent))


def load_cfg(p="config.yaml"):
    cfg = yaml.safe_load(open(p))
    for v in cfg["paths"].values():
        Path(v).mkdir(parents=True, exist_ok=True)
    return cfg


def _hdr(t):
    print(f"\n{'='*66}\n{t}\n{'='*66}")


# ----------------------------------------------------------------------
def cmd_inspect(a, cfg):
    from src import ecsmp
    _hdr("INSPECT")
    inv = ecsmp.inspect_dataset(a.root, save="outputs/ecsmp_inventory.json")
    print("\nSanity check:")
    print(f"  EEG subjects        : {len(inv['subjects_eeg'])}   (expect ~86)")
    print(f"  sleep subjects      : {len(inv['subjects_sleep'])}  (expect ~65)")
    print(f"  BOTH (analysis n)   : {len(inv['subjects_both'])}")
    if not inv['subjects_both']:
        print("  ! no overlap -- subject id parsing is wrong")
    return inv


def cmd_prep(a, cfg):
    from src import ecsmp
    _hdr("PREP")
    inv = (json.loads(Path("outputs/ecsmp_inventory.json").read_text())
           if Path("outputs/ecsmp_inventory.json").exists()
           else ecsmp.inspect_dataset(a.root, verbose=False))
    out = {}

    print("\n[1/3] emotion epochs")
    try:
        X, y, g = ecsmp.build_emotion(inv, cfg["paths"]["proc"],
                                      sfreq=cfg["ecsmp"]["sfreq"],
                                      epoch_sec=cfg["ecsmp"]["epoch_sec"],
                                      drop_onset_sec=cfg["ecsmp"]["drop_first_sec"])
        out["emotion"] = {"shape": list(X.shape),
                          "subjects": int(len(np.unique(g))),
                          "class_counts": np.bincount(y).tolist()}
    except Exception as e:
        print(f"  ! {e}")
        out["emotion"] = {"error": str(e)}

    print("\n[2/3] sleep HRV")
    try:
        df = ecsmp.build_sleep_hrv(inv, cfg["paths"]["proc"])
        out["sleep"] = {"n_subjects": int(len(df))}
    except Exception as e:
        print(f"  ! {e}")
        out["sleep"] = {"error": str(e)}

    print("\n[3/3] questionnaire tables")
    try:
        meta = ecsmp.load_metadata(inv["root"])
        sc = ecsmp.build_subject_scores(meta, cfg["paths"]["proc"])
        out["scales"] = {"shape": list(sc.shape) if sc is not None else None,
                         "columns": [c for c in sc.columns] if sc is not None else []}
    except Exception as e:
        print(f"  ! {e}")
        out["scales"] = {"error": str(e)}
    return out


def cmd_relate(a, cfg):
    from src import data, study
    _hdr("RELATE  (sleep -> next-day emotional reactivity)")
    proc, out = Path(cfg["paths"]["proc"]), cfg["paths"]["out"]

    sleep = pd.read_csv(proc / "sleep_hrv.csv")
    scales = (pd.read_csv(proc / "subject_scores.csv")
              if (proc / "subject_scores.csv").exists() else None)
    if scales is not None:
        sleep = sleep.merge(scales, on="subject", how="left")

    rfile = proc / "reactivity.csv"
    if rfile.exists() and not a.refresh:
        react = pd.read_csv(rfile)
        print(f"    cached reactivity {react.shape}")
    else:
        X, y, g, meta = data.load(proc, "ecsmp_emotion")
        react = study.reactivity(X, y, g, sfreq=cfg["ecsmp"]["sfreq"])
        react.to_csv(rfile, index=False)

    res = study.relate(sleep, react, scales, out_dir=out)
    if "tests" in res:
        study.plot_relation(res, Path(out) / "fig_relation.png")
        merged = pd.read_csv(Path(out) / "subject_level_merged.csv")
        top = res["tests"][0]
        study.plot_scatter(merged, top["sleep_predictor"], top["outcome"],
                           Path(out) / "fig_scatter_top.png")
    rev = study.reverse(sleep, scales, out_dir=out)
    return {"forward": res, "reverse": rev}


def cmd_classify(a, cfg):
    from src import analysis, baselines, data, ecsmp, features, train
    _hdr("CLASSIFY  (6-class emotion recognition)")
    proc, out = cfg["paths"]["proc"], cfg["paths"]["out"]
    X, y, g, meta = data.load(proc, "ecsmp_emotion")
    n_cls = int(y.max()) + 1
    print(f"    X{X.shape}  {n_cls} classes  {len(np.unique(g))} subjects")

    res = {}
    for m in [s.strip() for s in a.models.split(",")]:
        try:
            r, _, _ = train.run_cv(X, y, g, cfg, m, n_cls,
                                   device=a.device, tag="ecsmp_emotion")
            res[m] = r
            analysis.plot_confusion(r["oof"]["confusion"], ecsmp.EMOTIONS,
                                    f"{out}/cm_emotion_{m}.png", title=m)
        except Exception as e:
            print(f"  ! {m}: {e}")

    if a.baselines:
        cache = Path(proc) / "ecsmp_feats.npz"
        if cache.exists():
            z = np.load(cache, allow_pickle=True)
            F, names = z["F"], list(z["names"])
        else:
            print("    extracting features ...")
            F, names = features.extract(X, cfg["ecsmp"]["sfreq"])
            np.savez_compressed(cache, F=F, names=np.array(names, object))
        for m in ("rf", "lightgbm"):
            try:
                r, _ = baselines.run_baseline(F, y, g, m, n_cls, cfg, tag="ecsmp")
                res[m] = r
            except Exception as e:
                print(f"  skip {m}: {e}")

    if len(res) > 1:
        rep = analysis.compare_models(res, out_dir=out)
        analysis.plot_model_comparison(rep, f"{out}/fig_compare_emotion.png")
    return {k: v["summary"] for k, v in res.items()}


def cmd_report(a, cfg, collected=None):
    _hdr("REPORT")
    out = Path(cfg["paths"]["out"])
    collected = collected or {}
    L = [f"# ECSMP Results\n\nGenerated {datetime.now():%Y-%m-%d %H:%M}\n"]

    L.append("## Design and what it can support\n")
    L.append("- 89 participants, one night of sleep ECG, one emotion-induction session.")
    L.append("- Sleep was recorded the night **before** emotion induction, so the "
             "sleep → emotion path has temporal precedence. It is nevertheless a "
             "**between-subjects** comparison, not a within-person lagged design.")
    L.append("- The reverse path (mood → sleep) uses questionnaires collected at the "
             "same timepoint and is **cross-sectional only**. Direction is not "
             "identified; do not describe it as causal.\n")

    fwd = (collected.get("relate") or {}).get("forward", {})
    L.append("## 1. Does night-before sleep predict next-day emotional reactivity?\n")
    if "error" in fwd:
        L.append(f"Not run: {fwd['error']}\n")
    elif fwd:
        L.append(f"n = {fwd.get('n_subjects','?')} subjects, "
                 f"{fwd.get('n_tests',0)} tests, "
                 f"**{fwd.get('n_significant_fdr',0)} significant after FDR**.\n")
        L.append("| Sleep predictor | Outcome | r | p | q (FDR) | n | Sig |")
        L.append("|---|---|---|---|---|---|---|")
        for t in fwd.get("tests", [])[:12]:
            L.append(f"| {t['sleep_predictor']} | {t['outcome']} | "
                     f"{t['pearson_r']:+.3f} | {t['pearson_p']:.4g} | "
                     f"{t['q_fdr']:.3f} | {t['n']} | "
                     f"{'**yes**' if t['significant_fdr'] else 'no'} |")
        L.append(f"\n**Interpretation.** {fwd.get('interpretation','')}\n")
        L.append("\n![relation](fig_relation.png)\n\n![scatter](fig_scatter_top.png)\n")

    rev = (collected.get("relate") or {}).get("reverse", {})
    L.append("\n## 2. Mood / depression and sleep quality (cross-sectional)\n")
    if "error" in rev:
        L.append(f"Not run: {rev['error']}\n")
    elif rev:
        L.append(f"{rev.get('n_significant_fdr',0)}/{rev.get('n_tests',0)} "
                 "associations survive FDR.\n")
        L.append("| Mood measure | Sleep measure | r | q (FDR) | n |")
        L.append("|---|---|---|---|---|")
        for t in rev.get("tests", [])[:10]:
            L.append(f"| {t['mood_measure']} | {t['sleep_measure']} | "
                     f"{t['pearson_r']:+.3f} | {t['q_fdr']:.3f} | {t['n']} |")
        L.append(f"\n{rev.get('design_caveat','')}\n")

    cls = collected.get("classify") or {}
    L.append("\n## 3. Six-class emotion recognition (subject-grouped CV)\n")
    if cls:
        L.append("| Model | Accuracy | Macro F1 | κ | AUC |")
        L.append("|---|---|---|---|---|")
        for m, s in cls.items():
            L.append(f"| {m} | {s['accuracy']['mean']:.4f} ± {s['accuracy']['std']:.4f} "
                     f"| {s['balanced_f1_macro']['mean']:.4f} "
                     f"| {s['kappa']['mean']:.4f} | {s['roc_auc']['mean']:.4f} |")
        L.append("\nChance is 16.7% for six classes. Published subject-independent "
                 "physiological emotion recognition typically lands at 30–55%; "
                 "anything above ~70% warrants a leakage audit.\n")

    prep = collected.get("prep") or {}
    if prep:
        L.append("\n## 4. Data prepared\n")
        L.append("```json\n" + json.dumps(prep, indent=2) + "\n```\n")

    L.append("\n## 5. Limitations to state in the paper\n")
    L.append("1. One night per participant; no within-person repeated measures.")
    L.append("2. Sleep quality derived from single-lead ECG HRV, not full PSG staging.")
    L.append("3. Reverse path is cross-sectional and cannot establish direction.")
    L.append("4. n=89 gives limited power for the small effects (r ≈ 0.1–0.3) "
             "reported in this literature; a null result is informative, not a failure.")
    L.append("5. R-peak detection is a simple energy-threshold method; beat-level "
             "accuracy was not validated against manual annotation.\n")

    (out / "REPORT.md").write_text("\n".join(L), encoding="utf-8")
    print(f"  -> {out}/REPORT.md")
    return {}


# ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["inspect", "prep", "relate", "classify",
                                   "report", "all"])
    p.add_argument("--root", default="data/raw/ecsmp")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--device", default="cuda")
    p.add_argument("--models", default="cnn1d,fusion")
    p.add_argument("--baselines", action="store_true")
    p.add_argument("--refresh", action="store_true")
    a = p.parse_args()

    cfg = load_cfg(a.config)
    from src import train as T
    T.set_seed(cfg["seed"])

    collected = {}
    order = (["inspect", "prep", "relate", "classify", "report"]
             if a.cmd == "all" else [a.cmd])
    fns = {"inspect": cmd_inspect, "prep": cmd_prep, "relate": cmd_relate,
           "classify": cmd_classify}
    for s in order:
        if s == "report":
            cmd_report(a, cfg, collected)
            continue
        try:
            collected[s] = fns[s](a, cfg)
        except Exception as e:
            print(f"\n  ! stage '{s}' failed: {e}")
            traceback.print_exc()
            collected[s] = {"error": str(e)}
            if s in ("inspect", "prep") and a.cmd == "all":
                print("  stopping -- later stages depend on this")
                break

    Path(cfg["paths"]["out"], "ecsmp_summary.json").write_text(
        json.dumps(collected, indent=2, default=str), encoding="utf-8")
    print("\nDone.")


if __name__ == "__main__":
    main()
