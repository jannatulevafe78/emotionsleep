"""
Data layer.

Three jobs:
  1. synth_dataset()      -- fake but realistic EEG so the whole pipeline can be
                             smoke-tested before a single byte is downloaded.
  2. build_sleepedf()     -- Sleep-EDF Expanded  -> epochs.npy + labels.npy + subjects.npy
  3. build_sleepdep()     -- OpenNeuro ds004902  -> same three arrays

Everything downstream reads the same contract:
    X : float32  (N, C, T)   epochs
    y : int64    (N,)        labels
    g : int64    (N,)        subject id  (used for grouped CV -- never random split)

Arrays are written as .npy and loaded with mmap_mode='r' so 16 GB RAM is enough
even when the epoch tensor is larger than memory.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from scipy import signal as sps


# --------------------------------------------------------------------------
# 0. Signal utilities
# --------------------------------------------------------------------------
def bandpass(x: np.ndarray, sfreq: float, lo: float, hi: float, order: int = 4):
    """Zero-phase Butterworth bandpass along the last axis."""
    nyq = sfreq / 2.0
    lo_n = max(lo / nyq, 1e-6)
    hi_n = min(hi / nyq, 0.99)
    b, a = sps.butter(order, [lo_n, hi_n], btype="band")
    return sps.filtfilt(b, a, x, axis=-1).astype(np.float32)


def notch(x: np.ndarray, sfreq: float, freq: float, q: float = 30.0):
    if freq is None or freq <= 0 or freq >= sfreq / 2:
        return x
    b, a = sps.iirnotch(freq / (sfreq / 2.0), q)
    return sps.filtfilt(b, a, x, axis=-1).astype(np.float32)


def robust_z(x: np.ndarray, clip: float = 20.0):
    """Per-channel robust z-score. Median/IQR, not mean/std -- EEG has spikes."""
    med = np.median(x, axis=-1, keepdims=True)
    iqr = np.subtract(*np.percentile(x, [75, 25], axis=-1, keepdims=True))
    iqr = np.where(iqr < 1e-8, 1.0, iqr)
    z = (x - med) / (iqr / 1.349)
    return np.clip(z, -clip, clip).astype(np.float32)


def preprocess_block(x, sfreq, band=(0.5, 45.0), notch_hz=50.0, clip=20.0):
    x = np.asarray(x, dtype=np.float64)
    x = notch(x, sfreq, notch_hz)
    x = bandpass(x, sfreq, band[0], band[1])
    return robust_z(x, clip)


def resample_to(x, sfreq_in, sfreq_out):
    if abs(sfreq_in - sfreq_out) < 1e-6:
        return x.astype(np.float32)
    n_out = int(round(x.shape[-1] * sfreq_out / sfreq_in))
    return sps.resample(x, n_out, axis=-1).astype(np.float32)


# --------------------------------------------------------------------------
# 1. Synthetic EEG  -- for smoke tests and for debugging model capacity
# --------------------------------------------------------------------------
_BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "sigma": (12.0, 16.0),
    "beta": (16.0, 30.0),
}

# Rough per-stage band-power profiles (W, N1, N2, N3, REM).
# Not physiologically exact -- just separable enough that a working model
# should reach ~0.8+ macro-F1. If your model can't beat this, the bug is
# in the model, not the data.
_STAGE_PROFILE = {
    0: dict(delta=0.4, theta=0.5, alpha=1.6, sigma=0.3, beta=1.2),   # Wake
    1: dict(delta=0.8, theta=1.4, alpha=0.7, sigma=0.3, beta=0.6),   # N1
    2: dict(delta=1.4, theta=1.0, alpha=0.4, sigma=1.5, beta=0.4),   # N2 (spindles)
    3: dict(delta=3.0, theta=0.8, alpha=0.2, sigma=0.3, beta=0.2),   # N3 (SWS)
    4: dict(delta=0.6, theta=1.5, alpha=0.6, sigma=0.2, beta=0.8),   # REM
}


def _synth_epoch(profile, n_ch, n_samp, sfreq, rng, subj_gain):
    t = np.arange(n_samp) / sfreq
    out = np.zeros((n_ch, n_samp), dtype=np.float32)
    for ch in range(n_ch):
        sig = rng.normal(0, 0.3, n_samp)  # pink-ish background
        for band, (lo, hi) in _BANDS.items():
            amp = profile[band] * subj_gain[ch] * rng.uniform(0.8, 1.2)
            n_osc = 3
            for _ in range(n_osc):
                f = rng.uniform(lo, hi)
                phase = rng.uniform(0, 2 * np.pi)
                env = 1.0
                if band == "sigma":  # spindles are burst-like, not continuous
                    env = (rng.random() < 0.35) * np.exp(
                        -((t - rng.uniform(0, t[-1])) ** 2) / (2 * 0.35 ** 2)
                    )
                sig += amp / n_osc * env * np.sin(2 * np.pi * f * t + phase)
        out[ch] = sig
    return out


def synth_dataset(
    n_subjects: int = 12,
    epochs_per_subject: int = 200,
    n_channels: int = 2,
    sfreq: int = 100,
    epoch_sec: int = 30,
    n_classes: int = 5,
    seed: int = 42,
):
    """Generate a synthetic sleep-staging dataset with subject-level variability.

    Subject gain jitter is deliberate: it creates the inter-subject shift that
    makes grouped CV score lower than random CV. If your grouped and random CV
    scores are identical, you have leakage.
    """
    rng = np.random.default_rng(seed)
    n_samp = sfreq * epoch_sec
    X, y, g = [], [], []
    # realistic-ish stage priors (W under-represented because we crop wake)
    priors = np.array([0.18, 0.08, 0.42, 0.14, 0.18])[:n_classes]
    priors = priors / priors.sum()

    for s in range(n_subjects):
        subj_gain = rng.uniform(0.6, 1.5, size=n_channels)
        stages = rng.choice(n_classes, size=epochs_per_subject, p=priors)
        # add temporal smoothing -- real hypnograms are not i.i.d.
        for i in range(1, len(stages)):
            if rng.random() < 0.75:
                stages[i] = stages[i - 1]
        for st in stages:
            X.append(_synth_epoch(_STAGE_PROFILE[int(st)], n_channels,
                                  n_samp, sfreq, rng, subj_gain))
            y.append(int(st))
            g.append(s)

    X = np.stack(X).astype(np.float32)
    X = robust_z(X)
    return X, np.array(y, np.int64), np.array(g, np.int64)


def synth_sleepdep(n_subjects=30, epochs_per_session=40, n_channels=8,
                   sfreq=100, window_sec=20, seed=42):
    """Synthetic version of ds004902: same subject in NS and SD condition.

    SD is simulated as elevated theta/delta and reduced alpha -- the direction
    actually reported in the sleep-deprivation literature.
    """
    rng = np.random.default_rng(seed)
    n_samp = sfreq * window_sec
    ns_prof = dict(delta=0.6, theta=0.7, alpha=1.8, sigma=0.3, beta=1.0)
    sd_prof = dict(delta=1.1, theta=1.5, alpha=0.9, sigma=0.3, beta=0.8)
    X, y, g = [], [], []
    for s in range(n_subjects):
        gain = rng.uniform(0.6, 1.5, size=n_channels)
        # subject-specific vulnerability: not everyone degrades equally
        vuln = rng.uniform(0.4, 1.6)
        for cond, prof in ((0, ns_prof), (1, sd_prof)):
            p = {k: ns_prof[k] + vuln * (prof[k] - ns_prof[k]) for k in prof}
            for _ in range(epochs_per_session):
                X.append(_synth_epoch(p, n_channels, n_samp, sfreq, rng, gain))
                y.append(cond)
                g.append(s)
    X = robust_z(np.stack(X).astype(np.float32))
    return X, np.array(y, np.int64), np.array(g, np.int64)


# --------------------------------------------------------------------------
# 2. Sleep-EDF Expanded
# --------------------------------------------------------------------------
def build_sleepedf(raw_dir, out_dir, cfg):
    """Convert Sleep-EDF PSG/Hypnogram EDF pairs into epoch arrays.

    Requires: pip install mne
    Expects raw_dir to contain the sleep-cassette folder from PhysioNet.
    """
    import mne
    mne.set_log_level("ERROR")

    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sc = raw_dir / "sleep-cassette"
    if not sc.exists():
        sc = raw_dir

    psgs = sorted(sc.glob("*-PSG.edf"))
    if not psgs:
        raise FileNotFoundError(f"No *-PSG.edf under {sc}")

    sfreq = cfg["signal"]["sfreq"]
    epoch_sec = cfg["signal"]["epoch_sec"]
    stage_map = cfg["sleepedf"]["stage_map"]
    want_ch = cfg["sleepedf"]["channels"]
    crop_min = cfg["sleepedf"]["crop_wake_min"]
    n_subj = cfg["sleepedf"]["n_subjects"]

    # subject id = characters 3..5 of "SC4001E0" -> "400" ; two nights per subject
    def subj_of(p):
        return int(p.name[3:5])

    keep = sorted({subj_of(p) for p in psgs})[:n_subj]
    psgs = [p for p in psgs if subj_of(p) in keep]

    Xs, ys, gs = [], [], []
    for p in psgs:
        hyps = list(sc.glob(p.name[:6] + "*-Hypnogram.edf"))
        if not hyps:
            print(f"  ! no hypnogram for {p.name}, skipping")
            continue
        raw = mne.io.read_raw_edf(p, preload=True, stim_channel=None)
        ann = mne.read_annotations(hyps[0])
        raw.set_annotations(ann, emit_warning=False)

        chs = [c for c in want_ch if c in raw.ch_names]
        if not chs:
            print(f"  ! channels {want_ch} absent in {p.name}: {raw.ch_names}")
            continue
        raw.pick(chs)

        # crop the long wake tails -- otherwise W swamps the label distribution
        sleep_ann = [a for a in ann if a["description"] != "Sleep stage W"
                     and a["description"] in stage_map]
        if sleep_ann:
            t0 = max(sleep_ann[0]["onset"] - crop_min * 60, 0)
            t1 = min(sleep_ann[-1]["onset"] + sleep_ann[-1]["duration"]
                     + crop_min * 60, raw.times[-1])
            raw.crop(t0, t1)

        events, ev_id = mne.events_from_annotations(
            raw, event_id={k: v + 1 for k, v in stage_map.items()}, chunk_duration=epoch_sec
        )
        if len(events) == 0:
            continue
        ep = mne.Epochs(raw, events, event_id=ev_id, tmin=0.0,
                        tmax=epoch_sec - 1.0 / raw.info["sfreq"],
                        baseline=None, preload=True, on_missing="ignore")
        x = ep.get_data()                      # (n_ep, n_ch, n_samp)
        lab = ep.events[:, 2] - 1              # back to 0..4

        x = resample_to(x, raw.info["sfreq"], sfreq)
        x = preprocess_block(x, sfreq, cfg["signal"]["bandpass"],
                             cfg["signal"]["notch"], cfg["signal"]["clip_sigma"])
        Xs.append(x.astype(np.float32))
        ys.append(lab.astype(np.int64))
        gs.append(np.full(len(lab), subj_of(p), np.int64))
        print(f"  {p.name}: {len(lab)} epochs")

    X = np.concatenate(Xs); y = np.concatenate(ys); g = np.concatenate(gs)
    _save(out_dir, "sleepedf", X, y, g,
          meta=dict(classes=["W", "N1", "N2", "N3", "REM"], sfreq=sfreq,
                    epoch_sec=epoch_sec, channels=chs))
    return X, y, g


# --------------------------------------------------------------------------
# 3. OpenNeuro ds004902 -- sleep deprivation, resting state, PANAS/SSS labels
# --------------------------------------------------------------------------
def build_sleepdep(raw_dir, out_dir, cfg):
    """BIDS EEG -> windowed epochs, labelled by session (NS=0, SD=1).

    Requires: pip install mne mne-bids
    ses-1 = normal sleep, ses-2 = sleep deprivation (per dataset docs; the
    dataset counterbalances order, so ALWAYS read participants.tsv to confirm
    rather than trusting the session number blindly).
    """
    import mne
    mne.set_log_level("ERROR")
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sfreq = cfg["sleepdep"]["sfreq"]
    win = cfg["sleepdep"]["window_sec"]
    n_samp = int(sfreq * win)

    files = sorted(raw_dir.rglob("*_eeg.set")) or sorted(raw_dir.rglob("*_eeg.edf"))
    if not files:
        raise FileNotFoundError(f"No BIDS EEG files under {raw_dir}")

    Xs, ys, gs = [], [], []
    for f in files:
        name = f.name
        sub = int("".join(c for c in name.split("_")[0] if c.isdigit()))
        ses_tok = [t for t in name.split("_") if t.startswith("ses-")]
        cond = 1 if (ses_tok and ses_tok[0].endswith("2")) else 0

        raw = (mne.io.read_raw_eeglab(f, preload=True) if f.suffix == ".set"
               else mne.io.read_raw_edf(f, preload=True))
        raw.pick("eeg")
        x = raw.get_data()
        x = resample_to(x, raw.info["sfreq"], sfreq)
        x = preprocess_block(x, sfreq, cfg["signal"]["bandpass"],
                             cfg["signal"]["notch"], cfg["signal"]["clip_sigma"])

        n_win = x.shape[-1] // n_samp
        if n_win == 0:
            continue
        x = x[:, : n_win * n_samp].reshape(x.shape[0], n_win, n_samp)
        x = np.transpose(x, (1, 0, 2))         # (n_win, n_ch, n_samp)
        Xs.append(x.astype(np.float32))
        ys.append(np.full(n_win, cond, np.int64))
        gs.append(np.full(n_win, sub, np.int64))
        print(f"  {name}: sub-{sub} cond={cond} {n_win} windows")

    # channel count must match across files or the stack fails; take the min
    min_ch = min(a.shape[1] for a in Xs)
    Xs = [a[:, :min_ch] for a in Xs]

    X = np.concatenate(Xs); y = np.concatenate(ys); g = np.concatenate(gs)
    _save(out_dir, "sleepdep", X, y, g,
          meta=dict(classes=["normal_sleep", "sleep_deprived"], sfreq=sfreq,
                    window_sec=win, n_channels=int(min_ch)))
    return X, y, g


# --------------------------------------------------------------------------
# 4. Persistence
# --------------------------------------------------------------------------
def _save(out_dir, tag, X, y, g, meta=None):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{tag}_X.npy", X)
    np.save(out_dir / f"{tag}_y.npy", y)
    np.save(out_dir / f"{tag}_g.npy", g)
    meta = meta or {}
    meta.update(shape=list(X.shape), n_subjects=int(len(np.unique(g))),
                class_counts={int(k): int(v) for k, v in
                              zip(*np.unique(y, return_counts=True))})
    (out_dir / f"{tag}_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"saved {tag}: X{X.shape} y{y.shape} subjects={meta['n_subjects']}")


def load(proc_dir, tag, mmap=True):
    p = Path(proc_dir)
    mode = "r" if mmap else None
    X = np.load(p / f"{tag}_X.npy", mmap_mode=mode)
    y = np.load(p / f"{tag}_y.npy")
    g = np.load(p / f"{tag}_g.npy")
    meta = {}
    mp = p / f"{tag}_meta.json"
    if mp.exists():
        meta = json.loads(mp.read_text())
    return X, y, g, meta
