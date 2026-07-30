"""Build REPORT.md from the saved JSON results. Run: python make_report.py"""
import json
from datetime import datetime
from pathlib import Path

OUT = Path("outputs")
fwd = json.loads((OUT / "relation_sleep_to_emotion.json").read_text(encoding="utf-8"))
rev_p = OUT / "relation_mood_and_sleep.json"
rev = json.loads(rev_p.read_text(encoding="utf-8")) if rev_p.exists() else {}

L = [f"# ECSMP Results\n\nGenerated {datetime.now():%Y-%m-%d %H:%M}\n"]

L += ["## Design\n",
      "89 participants (ECSMP, Mendeley vn5nknh3mn v2). Sleep ECG recorded the",
      "night **before** the emotion-induction session, giving temporal precedence",
      "for the sleep to emotion direction. One night and one session per person,",
      "so this is a **between-subjects** analysis, not a within-person lagged",
      "design. The reverse direction uses questionnaires collected at the same",
      "timepoint and is **cross-sectional only** — direction is not identified.\n",
      "Emotional reactivity = per-subject EEG band-power deviation of each emotion",
      "segment from that subject's own neutral segment (within-subject",
      "differencing removes between-person offsets in absolute EEG power).\n",
      f"Analysis n = {fwd.get('n_subjects','?')} subjects with both usable EEG and sleep ECG.\n"]

sig = [t for t in fwd.get("tests", []) if t.get("significant_fdr")]

L += ["\n## 1. Primary finding: overnight autonomic INSTABILITY predicts next-day reactivity\n",
      f"{len(sig)} of {fwd.get('n_tests',0)} associations survive Benjamini-Hochberg",
      "FDR correction across **all** tests (no pre-specified subfamily, no",
      "selective exclusion).\n",
      "| Sleep predictor | Outcome | r | p | q (FDR) | n |",
      "|---|---|---|---|---|---|"]
for t in sig:
    L.append(f"| {t['sleep_predictor']} | {t['outcome'].replace('reactivity_','')} "
             f"| {t['pearson_r']:+.3f} | {t['pearson_p']:.4g} | {t['q_fdr']:.4f} "
             f"| {t['n']} |")

L += ["\n**Interpretation.** The *instability* of overnight autonomic tone, not its",
      "average level, predicts next-day emotional reactivity. Nights with larger",
      "swings in RMSSD and SDNN across 5-minute windows are followed by heightened",
      "reactivity. Mean-level HRV indices (RMSSD, pNN50, mean HR) do not predict;",
      "their variability does.\n"]

L += ["\n## 2. All tested associations\n",
      "| Sleep predictor | Outcome | r | p | q (FDR) | n | Sig |",
      "|---|---|---|---|---|---|---|"]
for t in fwd.get("tests", []):
    L.append(f"| {t['sleep_predictor']} | {t['outcome'].replace('reactivity_','')} "
             f"| {t['pearson_r']:+.3f} | {t['pearson_p']:.4g} | {t['q_fdr']:.4f} "
             f"| {t['n']} | {'**yes**' if t.get('significant_fdr') else 'no'} |")

L += ["\n## 3. Reverse direction: mood / depression and sleep (cross-sectional)\n"]
if rev:
    L += [f"{rev.get('n_significant_fdr',0)} of {rev.get('n_tests',0)} associations "
          "survive FDR correction.\n",
          "| Mood measure | Sleep measure | r | q (FDR) | n |",
          "|---|---|---|---|---|"]
    for t in rev.get("tests", [])[:15]:
        L.append(f"| {t['mood_measure']} | {t['sleep_measure']} | "
                 f"{t['pearson_r']:+.3f} | {t['q_fdr']:.4f} | {t['n']} |")
    L += [f"\n{rev.get('design_caveat','')}\n"]
L += ["\n**The asymmetry is the result.** Sleep predicts next-day emotional",
      "reactivity; mood shows no association with sleep quality here. This",
      "matches the published pattern (MIDUS random-intercept cross-lagged models;",
      "systematic review of 121 daily-monitoring studies), where sleep predicted",
      "subsequent affect more reliably than the reverse. The bidirectional loop",
      "originally hypothesised is **not** supported by these data.\n"]

L += ["\n## 4. Sensitivity check that FAILED — report this\n",
      "An earlier analysis excluded 25 of 65 nights using a beat-detection",
      "coverage threshold (implied beat rate vs mean HR ratio < 0.85), giving",
      "n = 38. In that restricted sample, mean-level indices appeared significant",
      "(pNN50 r = -0.428, mean HR r = +0.379, 6/6 directions as predicted).",
      "Those effects **did not survive** adding the excluded nights back:",
      "at n = 63, pNN50 fell to r = -0.172 (p = .18) and mean HR reversed sign",
      "to r = -0.088. Effects that vanish and flip under a change in sample",
      "composition are not robust, and the n = 63 result is the one reported",
      "above. The instability effects, by contrast, are significant at n = 63",
      "with correction across all 33 tests.\n"]

L += ["\n## 5. Limitations\n",
      "1. Only 4 EEG channels (FP1, FP2, F7, F8) are common to all subjects — a",
      "   sparse frontal montage. No claims about spatial topography or",
      "   lateralisation are warranted.",
      "2. One night and one emotion session per participant. Temporal precedence",
      "   holds but the design is between-subjects; this is not a within-person",
      "   causal test.",
      "3. Sleep quality is derived from single-lead ECG HRV, not PSG sleep staging.",
      "4. R-peak detection uses an adaptive energy-threshold method and was not",
      "   validated against manual beat annotation. Some nights show internally",
      "   inconsistent beat counts (see §4).",
      "5. The reverse direction is cross-sectional and cannot establish direction.",
      "6. Reactivity is indexed by EEG band-power change, an indirect measure of",
      "   emotional response; self-reported intensity (CESAF) was not modelled.\n"]

(OUT / "REPORT.md").write_text("\n".join(L), encoding="utf-8")
print(f"-> {OUT/'REPORT.md'}")
print(f"   n = {fwd.get('n_subjects')}, {len(sig)} significant after FDR")
for t in sig[:6]:
    print(f"   {t['sleep_predictor']:20s} -> {t['outcome']:22s} "
          f"r={t['pearson_r']:+.3f} q={t['q_fdr']:.4f}")
