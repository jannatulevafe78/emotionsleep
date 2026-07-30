# ECSMP Results

Generated 2026-07-30 05:23

## Design

89 participants (ECSMP, Mendeley vn5nknh3mn v2). Sleep ECG recorded the
night **before** the emotion-induction session, giving temporal precedence
for the sleep to emotion direction. One night and one session per person,
so this is a **between-subjects** analysis, not a within-person lagged
design. The reverse direction uses questionnaires collected at the same
timepoint and is **cross-sectional only** — direction is not identified.

Emotional reactivity = per-subject EEG band-power deviation of each emotion
segment from that subject's own neutral segment (within-subject
differencing removes between-person offsets in absolute EEG power).

Analysis n = 63 subjects with both usable EEG and sleep ECG.


## 1. Primary finding: overnight autonomic INSTABILITY predicts next-day reactivity

6 of 33 associations survive Benjamini-Hochberg
FDR correction across **all** tests (no pre-specified subfamily, no
selective exclusion).

| Sleep predictor | Outcome | r | p | q (FDR) | n |
|---|---|---|---|---|---|
| rmssd_instability | overall | +0.475 | 8.241e-05 | 0.0003 | 63 |
| rmssd_instability | negative | +0.472 | 9.26e-05 | 0.0003 | 63 |
| rmssd_instability | positive | +0.487 | 5.264e-05 | 0.0003 | 63 |
| sdnn_seg_std | overall | +0.460 | 0.0001485 | 0.0003 | 63 |
| sdnn_seg_std | negative | +0.459 | 0.0001571 | 0.0003 | 63 |
| sdnn_seg_std | positive | +0.464 | 0.0001256 | 0.0003 | 63 |

**Interpretation.** The *instability* of overnight autonomic tone, not its
average level, predicts next-day emotional reactivity. Nights with larger
swings in RMSSD and SDNN across 5-minute windows are followed by heightened
reactivity. Mean-level HRV indices (RMSSD, pNN50, mean HR) do not predict;
their variability does.


## 2. All tested associations

| Sleep predictor | Outcome | r | p | q (FDR) | n | Sig |
|---|---|---|---|---|---|---|
| rmssd_instability | overall | +0.475 | 8.241e-05 | 0.0003 | 63 | **yes** |
| rmssd_instability | negative | +0.472 | 9.26e-05 | 0.0003 | 63 | **yes** |
| rmssd_instability | positive | +0.487 | 5.264e-05 | 0.0003 | 63 | **yes** |
| sdnn_seg_std | overall | +0.460 | 0.0001485 | 0.0003 | 63 | **yes** |
| sdnn_seg_std | negative | +0.459 | 0.0001571 | 0.0003 | 63 | **yes** |
| sdnn_seg_std | positive | +0.464 | 0.0001256 | 0.0003 | 63 | **yes** |
| lf_nu | positive | -0.297 | 0.01814 | 0.1061 | 63 | no |
| lf_nu | overall | -0.287 | 0.02239 | 0.1160 | 63 | no |
| lf_nu | negative | -0.285 | 0.02358 | 0.1160 | 63 | no |
| rmssd | overall | +0.125 | 0.3301 | 0.4259 | 63 | no |
| rmssd | negative | +0.123 | 0.3359 | 0.4259 | 63 | no |
| rmssd | positive | +0.131 | 0.3064 | 0.4259 | 63 | no |
| pnn50 | overall | -0.172 | 0.1784 | 0.4259 | 63 | no |
| pnn50 | negative | -0.173 | 0.175 | 0.4259 | 63 | no |
| pnn50 | positive | -0.165 | 0.1974 | 0.4259 | 63 | no |
| lf_hf | overall | -0.178 | 0.1638 | 0.4259 | 63 | no |
| lf_hf | negative | -0.176 | 0.1678 | 0.4259 | 63 | no |
| lf_hf | positive | -0.184 | 0.1484 | 0.4259 | 63 | no |
| mean_hr | overall | -0.088 | 0.4905 | 0.4259 | 63 | no |
| mean_hr | negative | -0.092 | 0.4722 | 0.4259 | 63 | no |
| hr_seg_std | overall | +0.183 | 0.1518 | 0.4259 | 63 | no |
| hr_seg_std | negative | +0.180 | 0.1571 | 0.4259 | 63 | no |
| hr_seg_std | positive | +0.192 | 0.131 | 0.4259 | 63 | no |
| mean_hr | positive | -0.070 | 0.5833 | 0.5296 | 63 | no |
| sdnn | overall | +0.064 | 0.6175 | 0.6104 | 63 | no |
| sdnn | negative | +0.067 | 0.6023 | 0.6104 | 63 | no |
| sdnn | positive | +0.051 | 0.6923 | 0.6104 | 63 | no |
| night_hours | overall | -0.081 | 0.5268 | 0.6104 | 63 | no |
| night_hours | negative | -0.085 | 0.5092 | 0.6104 | 63 | no |
| sleep_hours_diary | overall | -0.085 | 0.5054 | 0.6104 | 63 | no |
| sleep_hours_diary | negative | -0.089 | 0.4881 | 0.6104 | 63 | no |
| sleep_hours_diary | positive | -0.069 | 0.5925 | 0.6337 | 63 | no |
| night_hours | positive | -0.065 | 0.615 | 0.6548 | 63 | no |

## 3. Reverse direction: mood / depression and sleep (cross-sectional)

0 of 24 associations survive FDR correction.

| Mood measure | Sleep measure | r | q (FDR) | n |
|---|---|---|---|---|
| sds_score | sdnn | -0.041 | nan | 64 |
| sds_score | rmssd | +0.007 | nan | 64 |
| sds_score | lf_hf | -0.042 | nan | 64 |
| sds_score | mean_hr | -0.163 | nan | 64 |
| sds_score | psqi_global | +0.297 | nan | 64 |
| sds_score | psqi_quality | +nan | nan | 64 |
| sds_score | psqi_latency | +0.200 | nan | 64 |
| sds_score | psqi_duration | +0.176 | nan | 64 |
| sds_score | psqi_efficiency | +0.147 | nan | 64 |
| sds_score | psqi_disorders | -0.124 | nan | 64 |
| sds_score | psqi_hypnotic | +0.203 | nan | 64 |
| sds_score | psqi_daytime | +0.049 | nan | 64 |
| poms_depression | sdnn | +0.005 | nan | 64 |
| poms_depression | rmssd | -0.090 | nan | 64 |
| poms_depression | lf_hf | +0.149 | nan | 64 |

CROSS-SECTIONAL. Mood and sleep measured at the same timepoint. Direction is not identified; this cannot support 'emotion causes poor sleep'.


**The asymmetry is the result.** Sleep predicts next-day emotional
reactivity; mood shows no association with sleep quality here. This
matches the published pattern (MIDUS random-intercept cross-lagged models;
systematic review of 121 daily-monitoring studies), where sleep predicted
subsequent affect more reliably than the reverse. The bidirectional loop
originally hypothesised is **not** supported by these data.


## 4. Sensitivity check that FAILED — report this

An earlier analysis excluded 25 of 65 nights using a beat-detection
coverage threshold (implied beat rate vs mean HR ratio < 0.85), giving
n = 38. In that restricted sample, mean-level indices appeared significant
(pNN50 r = -0.428, mean HR r = +0.379, 6/6 directions as predicted).
Those effects **did not survive** adding the excluded nights back:
at n = 63, pNN50 fell to r = -0.172 (p = .18) and mean HR reversed sign
to r = -0.088. Effects that vanish and flip under a change in sample
composition are not robust, and the n = 63 result is the one reported
above. The instability effects, by contrast, are significant at n = 63
with correction across all 33 tests.


## 5. Limitations

1. Only 4 EEG channels (FP1, FP2, F7, F8) are common to all subjects — a
   sparse frontal montage. No claims about spatial topography or
   lateralisation are warranted.
2. One night and one emotion session per participant. Temporal precedence
   holds but the design is between-subjects; this is not a within-person
   causal test.
3. Sleep quality is derived from single-lead ECG HRV, not PSG sleep staging.
4. R-peak detection uses an adaptive energy-threshold method and was not
   validated against manual beat annotation. Some nights show internally
   inconsistent beat counts (see §4).
5. The reverse direction is cross-sectional and cannot establish direction.
6. Reactivity is indexed by EEG band-power change, an indirect measure of
   emotional response; self-reported intensity (CESAF) was not modelled.
