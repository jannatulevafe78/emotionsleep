"""
ECSMP loader -- written against the VERIFIED structure of Mendeley vn5nknh3mn v2.

Layout (confirmed by direct inspection):
  EEG_downsample/YYYYMMDD_SSS.mat   EEGLAB struct, srate=250, 10 ch, ~77 min
  ECG_sleep/SSS/<ts>-<ts>.bin       custom binary, night-BEFORE recording
  ECG_experiment/SSS/<ts>-<ts>.bin  same format, during the session
  Cantab/SSS/*.csv                  cognitive battery
  scale.xlsx                        sheets: Readme ERQ SDS POMS PSQI CESAF
  experiment subjects.xlsx          sheets: Readme experiment sleep

Event codes (event.txt, verified against urevent):
  101 neutral  102 fear  103 sad  104 happy  105 anger  106 disgust
  11 = video start (immediately follows the emotion code)
  12 = video end

Segmentation rule: for each emotion code, the following 11 is onset and the
next 12 is offset. For subject 001 this yields neutral, disgust, fear, sad,
happy, anger -- exactly matching that subject's 'Order of videos' column,
which is the independent cross-check that the rule is correct.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal as sps
from scipy.io import loadmat

from .ecsmp_hrv import hrv_features, rr_intervals

EMOTIONS = ["neutral", "fear", "sad", "happy", "anger", "disgust"]
CODE2EMO = {101: 0, 102: 1, 103: 2, 104: 3, 105: 4, 106: 5}
VIDEO_START, VIDEO_END = 11, 12
BIN_HEADER, BIN_TAIL = 528, 208


# ======================================================================
# EEG
# ======================================================================
def read_eeg(path):
    m = loadmat(str(path), squeeze_me=True, struct_as_record=False)
    E = m["EEG"]
    data = np.asarray(E.data, dtype=np.float64)
    srate = float(E.srate)
    ev = []
    for e in np.atleast_1d(E.urevent):
        try:
            ev.append((int(e.type), float(e.latency)))
        except Exception:
            continue
    try:
        chans = [str(c.labels) for c in np.atleast_1d(E.chanlocs)]
    except Exception:
        chans = []
    return data, srate, ev, chans


def segment_emotions(events, n_samples):
    """Emotion code -> (onset, offset) samples.

    event.txt: 20 = end of watching each video, 21 = end of the following rest.
    Subject 001 is the exception and uses 11 (video start) / 12 (video end).
    Rule: onset is the emotion-code latency; offset is the first subsequent
    event of type 12 or 20, whichever appears first, and we stop at the next
    emotion code so a missing offset cannot swallow the following clip.
    """
    OFFSETS = (12, 20)
    segs = {}
    for i, (t, lat) in enumerate(events):
        if t not in CODE2EMO:
            continue
        emo = CODE2EMO[t]
        onset = int(round(lat))
        offset = None
        for t2, lat2 in events[i + 1:]:
            if t2 in CODE2EMO:
                break
            if t2 == VIDEO_START and abs(lat2 - lat) < 2 * 250:
                onset = int(round(lat2))
                continue
            if t2 in OFFSETS:
                offset = int(round(lat2))
                break
        if offset is None or offset <= onset:
            continue
        offset = min(offset, n_samples)
        dur = (offset - onset) / 250.0
        if not (60 <= dur <= 600):          # clips run ~3.5-5.5 min
            continue
        segs[emo] = (onset, offset)
    return segs


def subject_of(path):
    m = re.search(r"_(\d{3})", Path(path).stem)
    return int(m.group(1)) if m else None


# ======================================================================
# Custom .bin ECG  (port of readbindata.m)
# ======================================================================
def read_bin_ecg(path):
    """528-byte header, 208-byte tail, little-endian uint16 samples.
    Sampling rate is not stored; it is inferred from the wall-clock duration
    encoded in the filename and snapped to 128 / 256 / 512 Hz."""
    path = Path(path)
    raw = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    body = raw[BIN_HEADER: max(len(raw) - BIN_TAIL, BIN_HEADER)]
    n = len(body) // 2
    if n == 0:
        return np.array([]), 256.0
    pair = body[: 2 * n].reshape(n, 2).astype(np.float64)
    sig = pair[:, 0] + 256.0 * pair[:, 1]

    fs = 256.0
    try:
        a, b = path.stem.split("-")[:2]
        def secs(ts):
            return int(ts[8:10]) * 3600 + int(ts[10:12]) * 60 + int(ts[12:14])
        dur = secs(b) - secs(a)
        if dur < 0:
            dur += 24 * 3600
        if dur > 0:
            ratio = int(round((n / dur) / 128.0))
            fs = float({1: 128, 2: 256, 4: 512}.get(ratio, 256))
    except Exception:
        pass
    return sig, fs


# ======================================================================
# Metadata
# ======================================================================
def _norm_id(df):
    idc = next((c for c in df.columns if str(c).strip().lower() == "id"), None)
    if idc is None:
        return None
    df = df.copy()
    df["subject"] = pd.to_numeric(
        df[idc].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    df = df.dropna(subset=["subject"])
    df["subject"] = df["subject"].astype(int)
    return df


def load_metadata(root, verbose=True):
    root = Path(root)
    out = {}
    xl = root / "scale.xlsx"
    if xl.exists():
        for sheet in ("ERQ", "SDS", "POMS", "PSQI", "CESAF"):
            try:
                d = _norm_id(pd.read_excel(xl, sheet_name=sheet))
                if d is not None:
                    out[sheet.lower()] = d
            except Exception as e:
                print(f"    ! scale.xlsx[{sheet}]: {e}")
    es = root / "experiment subjects.xlsx"
    if es.exists():
        for sheet, key in (("experiment", "experiment"), ("sleep", "sleepinfo")):
            try:
                d = _norm_id(pd.read_excel(es, sheet_name=sheet))
                if d is not None:
                    out[key] = d
            except Exception as e:
                print(f"    ! subjects.xlsx[{sheet}]: {e}")
    if verbose:
        for k, v in out.items():
            print(f"    {k}: {v.shape}")
    return out


def _numeric_cols(df, keywords):
    return [c for c in df.columns
            if any(k in str(c).lower() for k in keywords)
            and pd.api.types.is_numeric_dtype(df[c])]


def build_subject_scores(meta, out_dir=None):
    """One tidy row per subject with the pre-computed scale scores."""
    frames = []

    if "psqi" in meta:
        d = meta["psqi"]
        exact = [c for c in d.columns
                 if str(c).strip().lower() == "psqi"
                 and pd.api.types.is_numeric_dtype(d[c])]
        if exact:
            frames.append(pd.DataFrame({"subject": d["subject"].values,
                                        "psqi_global": d[exact[0]].values}))
        comp_map = {"a.": "psqi_quality", "b.": "psqi_latency",
                    "c.": "psqi_duration", "d.": "psqi_efficiency",
                    "e.": "psqi_disorders", "f.": "psqi_hypnotic",
                    "g.": "psqi_daytime"}
        for pref, name in comp_map.items():
            hit = [c for c in d.columns
                   if str(c).strip().lower().startswith(pref)
                   and pd.api.types.is_numeric_dtype(d[c])]
            if hit:
                frames.append(pd.DataFrame({"subject": d["subject"].values,
                                            name: d[hit[0]].values}))

    if "sds" in meta:
        d = meta["sds"]
        c = _numeric_cols(d, ["index", "standard", "total", "raw score", "score"])
        if c:
            frames.append(pd.DataFrame({"subject": d["subject"].values,
                                        "sds_score": d[c[-1]].values}))

    if "poms" in meta:
        d = meta["poms"]
        for name, kw in (("poms_tension", "tension"), ("poms_anger", "anger ("),
                         ("poms_fatigue", "fatigue"), ("poms_depression", "depression"),
                         ("poms_vigor", "vigor"), ("poms_confusion", "confusion"),
                         ("poms_tmd", "tmd")):
            c = _numeric_cols(d, [kw])
            if c:
                frames.append(pd.DataFrame({"subject": d["subject"].values,
                                            name: d[c[0]].values}))

    if "erq" in meta:
        d = meta["erq"]
        for name, kw in (("erq_suppression", "suppression"),
                         ("erq_reappraisal", "reappraisal")):
            c = _numeric_cols(d, [kw])
            if c:
                frames.append(pd.DataFrame({"subject": d["subject"].values,
                                            name: d[c[0]].values}))

    if "experiment" in meta:
        d = meta["experiment"]
        rec = {"subject": d["subject"].values}
        for c in d.columns:
            cl = str(c).lower()
            if cl == "sex":
                rec["gender"] = d[c].astype(str).str.strip().str.lower().map(
                    {"female": 0, "male": 1}).values
            elif "age" in cl and pd.api.types.is_numeric_dtype(d[c]):
                rec["age"] = d[c].values
        if len(rec) > 1:
            frames.append(pd.DataFrame(rec))

    if "sleepinfo" in meta:
        d = meta["sleepinfo"]
        rec = {"subject": d["subject"].values}
        fa = next((c for c in d.columns if "fall asleep" in str(c).lower()), None)
        wk = next((c for c in d.columns if "wake" in str(c).lower()), None)
        comp = next((c for c in d.columns if "complete" in str(c).lower()), None)
        if fa and wk:
            t0 = pd.to_datetime(d[fa], errors="coerce")
            t1 = pd.to_datetime(d[wk], errors="coerce")
            dur = (t1 - t0).dt.total_seconds() / 3600.0
            dur = dur.where((dur > 0) & (dur < 16))
            rec["sleep_hours_diary"] = dur.values
        if comp:
            rec["sleep_ecg_finished"] = (
                d[comp].astype(str).str.strip().str.lower() == "finished").values
        frames.append(pd.DataFrame(rec))

    if not frames:
        return None
    df = frames[0]
    for f in frames[1:]:
        df = df.merge(f, on="subject", how="outer")
    df = df.loc[:, ~df.columns.duplicated()]
    print(f"    subject scores {df.shape}: "
          f"{[c for c in df.columns if c != 'subject']}")
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        df.to_csv(Path(out_dir) / "subject_scores.csv", index=False)
    return df


# ======================================================================
# Inventory
# ======================================================================
def _resolve_root(root):
    root = Path(root)
    if (root / "EEG_downsample").exists():
        return root
    hits = list(root.rglob("EEG_downsample"))
    if hits:
        return hits[0].parent
    raise FileNotFoundError(f"EEG_downsample not found under {root}")


def inspect_dataset(root, save="outputs/ecsmp_inventory.json", verbose=True):
    root = _resolve_root(root)
    eeg = sorted((root / "EEG_downsample").glob("*.mat"))

    def bins(sub):
        d = root / sub
        return sorted(d.rglob("*.bin")) if d.exists() else []

    def sid(p):
        return int(p.parent.name) if p.parent.name.isdigit() else subject_of(p)

    inv = {
        "root": str(root),
        "eeg": [{"path": str(p), "subject": subject_of(p)} for p in eeg],
        "sleep_ecg": [{"path": str(p), "subject": sid(p)} for p in bins("ECG_sleep")],
        "exp_ecg": [{"path": str(p), "subject": sid(p)} for p in bins("ECG_experiment")],
        "has_scale": (root / "scale.xlsx").exists(),
        "has_subjects": (root / "experiment subjects.xlsx").exists(),
    }
    inv["subjects_eeg"] = sorted({r["subject"] for r in inv["eeg"] if r["subject"]})
    inv["subjects_sleep"] = sorted({r["subject"] for r in inv["sleep_ecg"] if r["subject"]})
    inv["subjects_both"] = sorted(set(inv["subjects_eeg"]) & set(inv["subjects_sleep"]))

    if verbose:
        print(f"\nECSMP inventory: {root}")
        print(f"  EEG recordings   : {len(inv['eeg'])}  "
              f"({len(inv['subjects_eeg'])} subjects)")
        print(f"  sleep ECG (.bin) : {len(inv['sleep_ecg'])}  "
              f"({len(inv['subjects_sleep'])} subjects)")
        print(f"  experiment ECG   : {len(inv['exp_ecg'])}")
        print(f"  scale.xlsx       : {inv['has_scale']}")
        print(f"  subjects.xlsx    : {inv['has_subjects']}")
        print(f"  EEG AND SLEEP    : {len(inv['subjects_both'])} subjects   "
              f"<-- relation-analysis sample size")
        if eeg:
            d, sr, ev, ch = read_eeg(eeg[0])
            segs = segment_emotions(ev, d.shape[1])
            print(f"\n  probe {Path(eeg[0]).name}: {d.shape[0]} ch @ {sr:.0f} Hz, "
                  f"{d.shape[1]/sr/60:.1f} min")
            print(f"  channels: {ch}")
            print(f"  emotion segments: {len(segs)}/6")
            for e, (a, b) in sorted(segs.items()):
                print(f"    {EMOTIONS[e]:8s} {a:>9d}-{b:<9d}  {(b-a)/sr:6.1f} s")
        if inv["sleep_ecg"]:
            s, fs = read_bin_ecg(inv["sleep_ecg"][0]["path"])
            print(f"\n  probe {Path(inv['sleep_ecg'][0]['path']).name}: "
                  f"{len(s)} samples @ {fs:.0f} Hz = {len(s)/fs/3600:.2f} h")
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        Path(save).write_text(json.dumps(inv, indent=2, default=str))
        print(f"\n  -> {save}")
    return inv


# ======================================================================
# Emotion epochs
# ======================================================================
EXCLUDE_CH = ("trigger", "veo", "heo", "m1", "m2", "ecg", "status")


def build_emotion(inv, out_dir, sfreq=128, epoch_sec=4, drop_onset_sec=10,
                  max_subjects=None, exclude=EXCLUDE_CH):
    from .data import _save, preprocess_block

    n_samp = int(sfreq * epoch_sec)
    Xs, ys, gs = [], [], []
    recs = inv["eeg"][:max_subjects] if max_subjects else inv["eeg"]

    for r in recs:
        sub = r["subject"]
        if sub is None:
            continue
        try:
            data, sr, ev, chans = read_eeg(r["path"])
        except Exception as e:
            print(f"    ! sub {sub}: {e}")
            continue
        if chans and len(chans) == data.shape[0]:
            keep = [i for i, c in enumerate(chans)
                    if not any(x in str(c).strip().lower() for x in exclude)]
            if keep:
                if sub == recs[0]["subject"]:
                    print(f"    channels kept: {[chans[i] for i in keep]}")
                    print(f"    channels DROPPED (leakage/artifact): "
                          f"{[c for i, c in enumerate(chans) if i not in keep]}")
                data = data[keep]
        segs = segment_emotions(ev, data.shape[1])
        if not segs:
            print(f"    ! sub {sub}: no emotion segments")
            continue
        kept = 0
        for emo, (a, b) in sorted(segs.items()):
            a2 = a + int(drop_onset_sec * sr)
            if b - a2 < sr * epoch_sec:
                continue
            seg = sps.resample(data[:, a2:b],
                               int((b - a2) * sfreq / sr), axis=-1)
            n_ep = seg.shape[1] // n_samp
            if n_ep == 0:
                continue
            ep = seg[:, :n_ep * n_samp].reshape(seg.shape[0], n_ep, n_samp)
            ep = np.transpose(ep, (1, 0, 2))
            ep = ep[np.isfinite(ep).all(axis=(1, 2))]
            if len(ep) == 0:
                continue
            Xs.append(ep.astype(np.float32))
            ys.append(np.full(len(ep), emo, np.int64))
            gs.append(np.full(len(ep), sub, np.int64))
            kept += len(ep)
        print(f"    sub {sub:03d}: {len(segs)}/6 segments, {kept} epochs")

    if not Xs:
        raise RuntimeError("no epochs built -- check outputs/ecsmp_inventory.json")
    cmin = min(x.shape[1] for x in Xs)
    X = np.concatenate([x[:, :cmin] for x in Xs])
    X = preprocess_block(X, sfreq, (0.5, 45.0), 50.0, 20.0)
    y = np.concatenate(ys)
    g = np.concatenate(gs)
    _save(out_dir, "ecsmp_emotion", X, y, g,
          meta=dict(classes=EMOTIONS, sfreq=sfreq, epoch_sec=epoch_sec,
                    n_channels=int(cmin), source="ECSMP EEG_downsample"))
    return X, y, g


# ======================================================================
# Sleep HRV
# ======================================================================
def build_sleep_hrv(inv, out_dir, segment_min=5, max_subjects=None):
    rows = []
    recs = inv["sleep_ecg"][:max_subjects] if max_subjects else inv["sleep_ecg"]
    for r in recs:
        sub = r["subject"]
        if sub is None:
            continue
        try:
            sig, fs = read_bin_ecg(r["path"])
        except Exception as e:
            print(f"    ! sub {sub}: {e}")
            continue
        if len(sig) < fs * 600:
            print(f"    ! sub {sub}: only {len(sig)/fs/60:.1f} min, skipped")
            continue
        rr = rr_intervals(sig, fs)
        f = hrv_features(rr)
        if not f:
            print(f"    ! sub {sub}: too few usable beats")
            continue
        seg = int(segment_min * 60 * fs)
        sl = [hrv_features(rr_intervals(sig[i:i + seg], fs))
              for i in range(0, max(len(sig) - seg, 1), seg)]
        sd = [s["sdnn"] for s in sl if s]
        rm = [s["rmssd"] for s in sl if s]
        hr = [s["mean_hr"] for s in sl if s]
        f.update(subject=sub, fs=float(fs),
                 night_hours=float(len(sig) / fs / 3600),
                 n_segments=len(sd),
                 sdnn_seg_mean=float(np.mean(sd)) if sd else np.nan,
                 sdnn_seg_std=float(np.std(sd)) if sd else np.nan,
                 rmssd_seg_mean=float(np.mean(rm)) if rm else np.nan,
                 rmssd_instability=float(np.std(rm)) if rm else np.nan,
                 hr_seg_std=float(np.std(hr)) if hr else np.nan)
        rows.append(f)
        print(f"    sub {sub:03d}: {f['night_hours']:.2f} h @ {fs:.0f} Hz, "
              f"{f['n_beats']} beats, SDNN {f['sdnn']:.1f}, HR {f['mean_hr']:.1f}")

    if not rows:
        raise RuntimeError("no sleep HRV extracted")
    df = pd.DataFrame(rows)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(out_dir) / "sleep_hrv.csv", index=False)
    print(f"    sleep HRV {df.shape} -> {out_dir}/sleep_hrv.csv")
    return df
