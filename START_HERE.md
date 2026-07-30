# ECSMP pipeline

Written against the verified structure of Mendeley `vn5nknh3mn` v2.

## Commands (PowerShell, from this folder)

```powershell
$R = "C:\Users\User\Downloads\vn5nknh3mn-2\vn5nknh3mn-2"
$P = ".\.venv\Scripts\python.exe"

& $P run_ecsmp.py inspect --root $R      # seconds  -- confirm counts
& $P run_ecsmp.py prep    --root $R      # 20-60 min -- EEG epochs + sleep HRV + scores
& $P run_ecsmp.py relate                 # 1 min    -- THE RESEARCH QUESTION
& $P run_ecsmp.py classify --device cpu  # 1-3 h    -- 6-class emotion recognition
& $P run_ecsmp.py report                 # seconds
```

`--root` accepts either the outer or the doubled inner folder; it locates
`EEG_downsample` itself.

## What was verified before shipping

- **Event segmentation.** Rule: emotion code (101-106) -> next `11` = onset ->
  next `12` = offset. Replayed on subject 001's real marker list it yields
  neutral, disgust, fear, sad, happy, anger, matching that subject's
  "Order of videos" column exactly. Clip lengths 3.7-5.2 min.
- **`.bin` ECG reader.** Port of `readbindata.m`: 528-byte header, 208-byte
  tail, little-endian uint16, rate inferred from filename timestamps. On
  `20180413004758-2018041308084397.bin` it gives 512 Hz and a 7.38 h night;
  the sleep sheet records 00:47->08:08 = 7.35 h. 0.4% agreement.

## What was NOT verified

The EEGLAB `.mat` reading, the scale.xlsx column matching, and the HRV
extraction have not been run against the real files -- only compile-checked.
Expect one or two column-name mismatches on first run. The error messages name
the columns they looked for.

## Analysis design

| Path | Design | Strength |
|---|---|---|
| Night-before sleep HRV -> next-day emotional reactivity | between-subjects, n = subjects with BOTH recordings | sleep measured first: temporal precedence |
| POMS / SDS mood -> sleep quality | cross-sectional, same timepoint | association only, direction not identified |

Reactivity = per-subject band-power deviation of each emotion segment from that
subject's own **neutral** segment. Within-subject differencing removes the large
between-person offsets in absolute EEG power.

Sleep predictors: SDNN, RMSSD, pNN50, LF/HF, LF n.u., mean HR, RMSSD
instability, segment-wise SDNN SD, HR SD, night hours, diary sleep hours,
PSQI global.

## Reading the output

`outputs/REPORT.md`.

- **Chance is 16.7%** on six emotions. Subject-independent physiological emotion
  recognition normally lands 30-55%. Above ~70% means audit for leakage.
- **Null relation results are reportable.** Published effects are r 0.1-0.3.
  The `inspect` step prints how many subjects have BOTH EEG and sleep ECG --
  that number is your real n, and it caps the power available.
- Subjects marked `Unfinished` for sleep ECG are excluded automatically.
