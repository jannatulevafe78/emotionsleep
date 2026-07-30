# Overnight Autonomic Instability Predicts Next-Day Emotional Reactivity

**Methodology, data provenance, and results**

---

## 1. Research question and how it was answered

### Original question

Is there a bidirectional loop between sleep and emotion? Specifically: does poor
sleep worsen emotional state, does poor emotional state worsen sleep, and do the
two reinforce each other?

### What the data can and cannot test

The ECSMP dataset provides one night of sleep ECG recorded **before** a
single-session emotion-induction experiment, in the same participants. This
gives:

| Direction | Testable | Design | Evidential strength |
|---|---|---|---|
| Sleep (night N) → emotional reactivity (day N+1) | **Yes** | Between-subjects, n = 63 | Sleep measured first: temporal precedence |
| Mood → sleep quality | Partially | Cross-sectional, same timepoint | Association only; direction not identified |
| Reinforcing loop over time | **No** | Requires repeated days per person | Not testable with one night per subject |

The loop hypothesis as originally stated is **not testable** with this dataset,
and the analysis was therefore restricted to the two directions above.

### Answer obtained

**The relationship is asymmetric, and the predictive quantity is the
*instability* of overnight autonomic tone rather than its average level.**

1. Overnight HRV instability predicts next-day emotional reactivity
   (r ≈ 0.46–0.49, q = 0.0003, n = 63).
2. Mean-level overnight HRV (RMSSD, pNN50, mean HR) does **not** predict
   reactivity.
3. Mood and depression scores show **no** association with sleep quality
   (0 of 24 tests significant).

This asymmetry is consistent with the published literature, in which sleep
predicts subsequent affect more reliably than the reverse. The specific
finding — that variability rather than level carries the signal — is the novel
contribution.

---

## 2. Dataset

### Primary dataset: ECSMP

**ECSMP: A dataset on emotion, cognition, sleep, and multi-modal physiological
signals**
Gao et al., Southeast University, 2021.

- Repository: https://data.mendeley.com/datasets/vn5nknh3mn/2
- DOI: 10.17632/vn5nknh3mn.2
- Data descriptor: https://www.sciencedirect.com/science/article/pii/S2352340921009355
- PubMed: https://pubmed.ncbi.nlm.nih.gov/34926739/
- Size: ~3.9 GB

Contents used:

| Component | Description | Used for |
|---|---|---|
| `EEG_downsample/` | 86 EEGLAB `.mat` files, 250 Hz, 10 channels, ~77 min | Emotional reactivity |
| `ECG_sleep/` | 88 custom `.bin` files, night before the session, 512 Hz | Sleep HRV |
| `scale.xlsx` | Sheets: ERQ, SDS, POMS, PSQI, CESAF | Questionnaire scores |
| `experiment subjects.xlsx` | Sheets: experiment, sleep | Demographics, video order, sleep times |
| `event.txt` | Marker code definitions | Segmentation |
| `readbindata.m` | MATLAB reader for the `.bin` format | Porting the ECG reader |

Not used: `Cantab/` (cognitive battery), `E4/` (Empatica wristband),
`ECG_experiment/`.

### Reference literature for the pre-specified direction

- Systematic review of 121 daily-monitoring studies (Sensors, 2024): sleep
  predicted subsequent daytime affect more often than the reverse.
- MIDUS random-intercept cross-lagged panel models (n = 2,022 and n = 782):
  shorter sleep predicted higher next-day negative affect; negative affect did
  not predict next-night sleep duration.
- Bidirectional stress/sleep study (n = 326, >2,500 nights): both directions
  significant when the affect measure was perceived stress specifically.

---

## 3. Methodology

### 3.1 EEG preprocessing and emotion segmentation

**Channel selection.** Recordings contain 10 channels: FP1, FPZ, FP2, AF3, AF4,
F7, F8, M2, VEO, Trigger. Three were excluded:

- **Trigger** — the event-marker channel. It encodes the emotion condition
  directly; retaining it would leak the class label into the input.
- **VEO** — vertical electro-oculogram (ocular artifact).
- **M2** — mastoid reference.

Channels were then matched **by name** across subjects, not by position. Four
electrodes were common to all 86 subjects: **FP1, FP2, F7, F8**. Position-based
truncation was explicitly avoided because channel order is not guaranteed
identical across files.

**Segmentation.** Marker codes from `event.txt`:

```
101 neutral   102 fear   103 sad   104 happy   105 anger   106 disgust
11  video start (subject 001 only)
12  video end   (subject 001 only)
20  end of watching each video
21  end of the rest period following each video
```

Segmentation rule: onset = the emotion-code latency; offset = the first
subsequent event of type **12 or 20**, whichever occurs first, terminating at
the next emotion code. Segments outside 60–600 s were rejected.

This rule was validated two ways. First, replayed against the marker lists of
six independently inspected files (subjects 001, 002, 003, 012, 043, 089) it
recovered clip durations of 220–312 s in all cases. Second, for subject 001 the
recovered emotion order (neutral, disgust, fear, sad, happy, anger) matches that
subject's `Order of videos` column in `experiment subjects.xlsx` exactly — an
independent cross-check, since video order was randomised per participant.

Note that subject 001 uses the 11/12 marker convention while all other subjects
use 20/21. A rule requiring 11/12 matches only subject 001; this was a real bug
during development and is recorded here because it is a trap for anyone
reprocessing this dataset.

**Epoching.** First 10 s of each clip discarded (onset artifact and
undeveloped response). Signals decimated then resampled to 128 Hz, segmented
into 4 s epochs, bandpass filtered 0.5–45 Hz, 50 Hz notch, robust z-scored
(median/IQR) with clipping at ±20σ. Final array: **34,408 epochs × 4 channels ×
512 samples**, 86 subjects, ~400 epochs per subject, 6 conditions.

### 3.2 Emotional reactivity

Reactivity is computed **within subject**, as the relative deviation of each
emotion condition's band power from that subject's own neutral condition:

```
reactivity[emotion, band] = (power[emotion, band] − power[neutral, band])
                            / power[neutral, band]
```

Bands: delta (0.5–4), theta (4–8), alpha (8–13), sigma (12–16), beta (13–30),
gamma (30–45 Hz), via Welch PSD, averaged across the four channels.

Within-subject differencing is essential: absolute EEG power varies by an order
of magnitude between individuals due to skull thickness, electrode impedance,
and montage placement. Using neutral as each subject's own baseline removes
that nuisance variance.

Aggregate indices:

- `reactivity_overall` — mean absolute deviation across all emotions and bands
- `reactivity_negative` — mean across fear, sad, anger, disgust
- `reactivity_positive` — happy only

### 3.3 Sleep HRV

**Binary format.** The `.bin` files use an undocumented format; the reader was
ported from the supplied `readbindata.m`: 528-byte header, 208-byte tail,
little-endian uint16 samples. Sampling rate is not stored and is inferred from
the wall-clock duration encoded in the filename, snapped to 128/256/512 Hz.

Validation: for `20180413004758-2018041308084397.bin` the reader yields 512 Hz
and a 7.38 h recording; the `sleep` sheet independently records 00:47 → 08:08 =
7.35 h. Agreement to 0.4%.

**R-peak detection.** Signal decimated to 128 Hz (QRS energy is below ~40 Hz, so
beat timing is unaffected), bandpass 5–20 Hz, squared derivative, 50 ms
moving-average energy envelope. Threshold is computed **per 10-second window**
as max(P90 × 0.45, median × 2.5) — a single global threshold cannot track
amplitude drift across an 8-hour night. Refractory period 280 ms. RR intervals
restricted to 300–2000 ms with an ectopic filter at median ± 35%.

**HRV indices.** Time domain: mean RR, mean HR, SDNN, RMSSD, SDSD, pNN20, pNN50,
CVNN. Frequency domain from the 4 Hz interpolated tachogram: VLF (0.0033–0.04),
LF (0.04–0.15), HF (0.15–0.4 Hz), LF/HF, LF n.u., total power.

**Instability indices — the key measures.** The night is divided into
5-minute windows and HRV recomputed per window by slicing the already-detected
RR series. Then:

- `rmssd_instability` = SD of RMSSD across windows
- `sdnn_seg_std` = SD of SDNN across windows
- `hr_seg_std` = SD of mean HR across windows

These quantify how much autonomic tone *fluctuates* over the night, as distinct
from its average.

Result: 65 subjects with usable sleep HRV; 63 with both usable EEG and sleep ECG.

### 3.4 Questionnaire scores

Parsed from `scale.xlsx`, using the pre-computed subscale scores supplied by the
dataset authors: PSQI global plus components A–G; SDS index; POMS tension,
anger, fatigue, depression, vigor, confusion, TMD; ERQ suppression and
reappraisal. Demographics and diary sleep duration from
`experiment subjects.xlsx`.

### 3.5 Statistical analysis

- Pearson and Spearman correlations between each sleep predictor and each
  reactivity outcome (33 tests).
- Benjamini–Hochberg FDR correction applied **across all 33 tests**. No
  pre-specified subfamily was used for the reported result.
- Where covariates (age, sex) were available with >70% completeness, partial
  associations were estimated by OLS.
- Reverse direction: mood/depression scores against sleep measures, 24 tests,
  FDR corrected.

---

## 4. Results

### 4.1 Primary finding

n = 63. Six associations survive FDR correction across all 33 tests.

| Sleep predictor | Outcome | r | q (FDR) |
|---|---|---|---|
| `rmssd_instability` | positive reactivity | **+0.487** | 0.0003 |
| `rmssd_instability` | overall reactivity | **+0.475** | 0.0003 |
| `rmssd_instability` | negative reactivity | **+0.472** | 0.0003 |
| `sdnn_seg_std` | positive reactivity | **+0.464** | 0.0003 |
| `sdnn_seg_std` | overall reactivity | **+0.460** | 0.0003 |
| `sdnn_seg_std` | negative reactivity | **+0.459** | 0.0003 |

Two distinct instability indices converge on r ≈ 0.46–0.49. Because
`rmssd_instability` and `sdnn_seg_std` are related but non-identical measures,
their agreement is corroborating rather than redundant.

### 4.2 Mean-level HRV does not predict

| Predictor | Outcome | r | p |
|---|---|---|---|
| pNN50 | overall reactivity | −0.172 | 0.178 |
| RMSSD | overall reactivity | +0.125 | 0.330 |
| mean HR | overall reactivity | −0.088 | 0.491 |

The dissociation is the substantive point: it is not autonomic *level* during
sleep that predicts next-day emotional response, but the *stability* of
autonomic regulation.

### 4.3 Effects are non-specific to valence

Overall (+0.475), negative (+0.472), and positive (+0.464 to +0.487) reactivity
are predicted almost equally. Overnight autonomic instability is followed by
globally heightened emotional reactivity, not selectively amplified negative
affect. This should be stated explicitly: the uniformity across valence is a
finding, not an oversight.

### 4.4 Reverse direction: null

0 of 24 mood-to-sleep associations survive FDR correction. Because these
measures are contemporaneous, no directional claim is possible either way; the
null is reported as an absence of association, not as evidence against a causal
effect.

### 4.5 Sensitivity analysis that failed — must be reported

An earlier version of the analysis applied a beat-detection quality filter,
excluding nights whose implied beat rate diverged from mean HR by more than 15%.
This removed 25 of 65 nights, leaving n = 38. In that restricted sample,
mean-level indices appeared significant (pNN50 r = −0.428, p = .007; mean HR
r = +0.379, p = .019) with all six pre-specified directions matching prediction.

Those effects did **not survive** restoring the excluded nights: at n = 63 pNN50
fell to r = −0.172 (p = .18) and mean HR reversed sign to r = −0.088. Effects
that vanish and change sign under a change in sample composition are not robust,
and the n = 63 analysis is the one reported.

The instability effects, by contrast, are significant at n = 63 with correction
across all tests and required no exclusion rule.

Reporting this transparently is necessary. It also constitutes a methodological
observation worth stating: in small-n HRV studies, quality-control thresholds
are themselves a source of researcher degrees of freedom capable of generating
apparently clean, theory-consistent results.

---

## 5. How this addresses the research question

| Original sub-question | Answer |
|---|---|
| Can sleep act as a biomarker predicting next-day emotional state? | **Yes**, but the informative feature is overnight autonomic *instability*, not mean sleep quality (r ≈ 0.47, q = 0.0003) |
| Can emotional state alter sleep? | **Not supported here.** 0/24 associations; design cannot identify direction regardless |
| Which biomarkers predict emotional instability? | `rmssd_instability`, `sdnn_seg_std` — segment-wise variability of overnight HRV |
| Is the relationship bidirectional? | **No evidence for a loop.** The relationship is asymmetric, consistent with published findings |

The originally hypothesised reinforcing loop is not supported. Stating this
directly is what makes the positive finding credible.

---

## 6. Limitations

1. **Four EEG channels only** (FP1, FP2, F7, F8) are common to all subjects.
   Sparse frontal coverage; no claims about topography, lateralisation, or
   posterior activity are warranted.
2. **One night, one session per participant.** Temporal precedence holds, but
   the analysis is between-subjects. No within-person or lagged inference is
   possible, and the loop hypothesis cannot be tested at all.
3. **No PSG.** Sleep quality is derived from single-lead ECG HRV; no sleep
   staging, no arousal indices, no apnea screening.
4. **R-peak detection was not validated against manual annotation.** Several
   nights show internally inconsistent beat counts (see §4.5).
5. **Reactivity is indexed by EEG band-power change**, an indirect measure. The
   CESAF self-reported emotion ratings included in the dataset were not
   modelled and would strengthen convergent validity.
6. **Cross-sectional reverse path.** Contemporaneous measurement cannot
   establish direction.
7. **n = 63** limits power for effects smaller than about r = 0.30.
8. **Student sample**, healthy young adults from a single institution. Results
   may not generalise to clinical or older populations.
9. Emotion classification from EEG (6-class) was not completed and is not
   reported.

---

## 7. Reproducibility

- Dataset: openly available at the DOI above; no registration required.
- Fixed random seed (42) throughout.
- Deterministic preprocessing; all parameters in `config.yaml`.
- All results written as JSON: `relation_sleep_to_emotion.json`,
  `relation_mood_and_sleep.json`, `subject_scores.csv`, `sleep_hrv.csv`,
  `reactivity.csv`.
- Subject-grouped splitting is enforced anywhere model training occurs;
  epoch-level random splitting would leak adjacent windows from the same
  recording across train and test.

### Pipeline

```
run_ecsmp.py inspect   → file inventory and structure validation
prep_fast.py           → EEG epochs, sleep HRV, questionnaire scores
run_ecsmp.py relate    → correlation analysis, FDR correction
make_report.py         → results report
```

---

## 8. Suggested framing for submission

**Title direction:** emphasise instability, not the loop. For example:
"Overnight heart-rate-variability instability predicts next-day emotional
reactivity: a between-subjects analysis of the ECSMP dataset."

Avoid "digital twin" — one session per participant does not support a
per-individual validated simulator.

**Positioning:** a replication of the published sleep-to-affect asymmetry using
objective overnight cardiac autonomic measures, with the novel observation that
variability rather than level carries the predictive signal.

**Candidate venues:** *Sensors*, *Frontiers in Psychiatry*, *IEEE Access*,
*Journal of Sleep Research*, *Behavioral Sleep Medicine*.
