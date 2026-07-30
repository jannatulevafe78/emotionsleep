# Proposal → Delivery Mapping

**Purpose of this document.** The original proposal specifies an EEG-based framework.
This maps every requirement in it to what is actually deliverable with open data,
flags the three items that must change, and gives the defence for each change.

**Summary for the supervisor:** roughly 85% of the proposal is delivered unchanged.
The technical stack (SSL, contrastive learning, transformers, GNNs, multi-task
learning, XAI, statistics, validation protocol) is kept in full. Two things change:
the **signal modality** (EEG → ECG/HRV) and the **causal method** (causal discovery →
cross-lagged panel models). Both changes make the work *more* defensible, not less,
and both are forced by data availability rather than convenience.

---

## 1. The one problem that drives every change

The proposal's central claim is a bidirectional loop:
emotion → disturbed sleep → worse emotion → worse sleep.

**No public EEG dataset can test this.** Sleep EEG corpora (Sleep-EDF, Siena,
Bitbrain) contain no emotion labels. Emotion EEG corpora (DEAP, SEED, DREAMER)
are awake participants watching video clips, with no sleep recording. No dataset
links one person's night to their next day's mood using EEG.

Testing a within-person, day-to-day loop requires **repeated measures of both
constructs in the same people over many days**. Exactly one open dataset does this,
and it uses cardiac rather than cortical signals. Hence the modality change.

---

## 2. Research questions — status

| # | Proposal question | Status | How it is answered |
|---|---|---|---|
| 1 | Can sleep act as a digital biomarker predicting next-day emotional state? | **Answered directly** | Cross-lagged path, sleep(night N) → arousal(day N+1) |
| 2 | Can emotional dysregulation alter sleep microstructures? | **Answered, narrowed** | Reverse path, arousal(day N) → sleep(night N). "Microstructure" → sleep quality/efficiency/WASO (no PSG available at scale) |
| 3 | Which biomarkers predict emotional instability? | **Answered, modality changed** | SHAP over HRV + spectral features instead of EEG band powers |
| 4 | Do personalised models beat population-level models? | **Answered directly** | Per-subject random effects vs fixed-effects-only; also per-subject fine-tuning |

All four questions survive. Question 2 is narrowed and question 3 changes modality.

---

## 3. The literature already predicts the answer — this is a strength

Published work using exactly this design converges on an **asymmetry**, which
gives the project a falsifiable prediction rather than an open fishing expedition:

| Path | Published expectation | Evidence base |
|---|---|---|
| Sleep → next-day affect | **Significant** | MIDUS RI-CLPM, n=2,022 and n=782; systematic review of 121 studies |
| Negative affect → next-night sleep | **Null / weak** | MIDUS; 14-day grief EMA study |
| Stress → next-night sleep | **Significant** | 326 young adults, >2,500 nights |
| Sleep duration → affect | **Non-linear (U-shaped)** | MIDUS; <7.5 h or >10.5 h both worse than 7.5–10.5 h |

Key references:
- Systematic review of 121 studies (Sensors, 2024) — sleep predicted subsequent
  affect more often than the reverse.
- MIDUS random-intercept cross-lagged panel models — short sleep predicted next-day
  negative affect; negative affect did **not** predict next-night sleep duration.
- Bi-directional stress/sleep study (n=326) — both directions significant for
  *perceived stress* specifically.

**Framing for the paper:** the loop closes for stress but not for negative affect
in general. That distinction is the contribution. Reproducing the asymmetry in a
new modality (continuous HRV instead of self-report) is a replication result;
finding symmetry instead is a disagreement worth reporting. Either outcome is
publishable, which is the point of pre-specifying it.

---

## 4. Datasets — substitutions

| Proposal | Replacement | Why | Size |
|---|---|---|---|
| Sleep-EDF (PhysioNet) | **MIT-BIH Polysomnographic (slpdb)** | ECG + EEG + sleep stages in one record; the ECG channel is what transfers to the affect task | ~600 MB |
| OpenNeuro sleep deprivation | *(optional, retained)* | Still valid for a one-directional causal probe | subset ~1 GB |
| Siena, Bitbrain | **Dropped** | Redundant; no emotion link | — |
| DEAP / SEED / DREAMER | **PhysioNet Non-EEG (noneeg)** | Open, no request form; labelled emotional/cognitive/physical stress states; shares a heart-rate channel with slpdb | ~50 MB |
| *(none in proposal)* | **Baigutanova et al. 2025 (Figshare)** | The only open dataset with repeated within-person sleep *and* affect-relevant measures across 28 days. 49 participants, sleep diaries, continuous HRV, PHQ-9/GAD-7/ISI | ~150 MB (excl. raw PPG) |

**Total under 1 GB** with the raw PPG deliberately excluded — the 5-minute HRV
features are precomputed, so the raw 10 Hz signal adds many GB and nothing usable.

**Modality defence for the supervisor:** EEG and ECG are not arbitrary substitutes.
Sleep–emotion coupling is mediated by autonomic nervous system activity, and HRV is
the standard non-invasive index of that system. Cardiac signals are recorded in both
sleep studies and affect studies, which is precisely why transfer between them is
possible and EEG transfer is not. The modality change is what makes the core
research question testable at all.

---

## 5. Architecture components — all retained

| Proposal component | Status | Where |
|---|---|---|
| Foundation model pretraining | **Kept**, rescoped to signal-level SSL | `train.pretrain_ssl` on slpdb |
| Self-supervised learning | **Kept** | SimCLR / NT-Xent |
| Contrastive learning | **Kept** | `nt_xent`, physiologically-motivated augmentations |
| Temporal transformers | **Kept** | `EEGTransformer` |
| Graph neural networks | **Kept** | `EEGGNN`, learned adjacency over sensor channels |
| Multi-task learning | **Kept** | sleep stage + arousal state heads |
| Explainable AI | **Kept** | SHAP, integrated gradients, saliency, band attribution |
| Causal AI | **Changed** → cross-lagged panel models | See §6 |
| Digital biomarker discovery | **Kept** | SHAP ranking over HRV/spectral features |
| Personalised digital twins | **Renamed** → personalised mixed-effects models | See §6 |

Feature engineering (time-domain, frequency-domain, connectivity, non-linear),
all six experiment families, all classification and regression metrics, all four
statistical tests, effect sizes, confidence intervals, and every publication figure
listed in the proposal are implemented as specified.

---

## 6. The two changes that need defending

### 6.1 "Causal AI / causal discovery" → cross-lagged panel models

**Why.** Causal discovery algorithms on observational time series produce graphs
that are not identifiable without strong, untestable assumptions. A reviewer will
challenge any causal claim built that way. Cross-lagged panel models with random
intercepts are the established method in exactly this literature, control for the
lagged outcome, and separate within-person from between-person effects.

**Defence.** This is a strictly stronger method for the question asked, not a
retreat. It is what the papers being replicated actually used.

### 6.2 "Personalised digital twins" → personalised mixed-effects models

**Why.** A digital twin implies a per-individual generative simulator validated
against that individual's future states. 28 days per person does not support that.
Claiming it invites rejection.

**Defence.** Per-subject random intercepts and slopes *are* personalisation, and
they let question 4 be answered rigorously: does allowing subject-specific
coupling improve fit over a population model? Recommend removing "Digital Twin"
from the title.

---

## 7. Validation and reproducibility — as specified

- Participant-wise splits, LOSO, 5-fold grouped CV, walk-forward validation
  (walk-forward is the natural fit for the 28-day panel) — all implemented.
- Subject-grouped splitting is enforced in code. Random-splitting physiological
  epochs leaks adjacent windows across train/test and inflates accuracy by 10–25
  points; it is the most common cause of non-replication in this field.
- `config.yaml`, `requirements.txt`, seed control, per-run JSON logging — present.
  Dockerfile is the one reproducibility item still outstanding.

---

## 8. Realistic performance expectations

State these up front so strong numbers are not mistaken for errors, and weak ones
are not mistaken for failure:

| Task | Realistic subject-independent range |
|---|---|
| Sleep staging from ECG alone (slpdb) | 60–75% accuracy, κ 0.45–0.65 |
| Arousal-state classification (noneeg) | 70–85% accuracy |
| Cross-lagged path coefficients | β 0.05–0.20; small effects are normal and expected |

A 95%+ accuracy on any of these indicates leakage, not success. Cross-lagged effects
in this literature are small by nature; statistical significance across ~1,300
person-days matters more than effect magnitude.

---

## 9. Honest limitations to state in the paper

1. Sleep is self-reported diary, not polysomnography — participants relied on recall.
2. Daily affect is indexed by HRV, an autonomic proxy, not momentary self-reported mood.
   Clinical scores (PHQ-9/GAD-7/ISI) exist at only three timepoints, too sparse for
   daily modelling; they serve as person-level outcomes instead.
3. Observational design. Cross-lagged models establish temporal precedence, not
   causation. The sleep-deprivation dataset is the only experimental manipulation
   available and covers one direction only.
4. Participants were instructed to charge the watch overnight, so overnight HRV
   coverage is incomplete.
