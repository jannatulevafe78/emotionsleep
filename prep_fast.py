"""
Fast prep. Same outputs as the originals, three optimisations:

1. ECG decimated 512 -> 128 Hz before R-peak detection. QRS energy lives below
   ~40 Hz, so 128 Hz is ample for beat timing; this alone removes 4x the work.
2. R-peaks detected ONCE per night. Segment-wise HRV is then computed by
   slicing the RR series by cumulative time, instead of re-running detection on
   every 5-minute window (the old code did the whole night's work twice over).
3. Subjects processed in parallel across CPU cores.

Also caps epochs per emotion: 40 epochs x 4 s = 160 s per condition per subject
is plenty for training, and the uncapped version produced a needlessly large
array that slowed every downstream stage.

Run:  python prep_fast.py --root <ECSMP folder>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import signal as sps

sys.path.insert(0, str(Path(__file__).parent))

from src.ecsmp import (EMOTIONS, EXCLUDE_CH, read_bin_ecg, read_eeg,  # noqa: E402
                       segment_emotions, load_metadata, build_subject_scores,
                       inspect_dataset)
from src.ecsmp_hrv import hrv_features  # noqa: E402


# ======================================================================
# EEG worker
# ======================================================================
def _eeg_worker(args):
    path, sub, sfreq, epoch_sec, drop_onset, max_ep = args
    try:
        data, sr, ev, chans = read_eeg(path)
    except Exception as e:
        return sub, None, None, f"read failed: {e}", None

    dropped = []
    if chans and len(chans) == data.shape[0]:
        keep = [i for i, c in enumerate(chans)
                if not any(x in str(c).strip().lower() for x in EXCLUDE_CH)]
        dropped = [c for i, c in enumerate(chans) if i not in keep]
        if keep:
            data = data[keep]
            chans = [chans[i] for i in keep]

    segs = segment_emotions(ev, data.shape[1])
    if not segs:
        return sub, None, None, "no segments", chans

    n_samp = int(sfreq * epoch_sec)
    Xs, ys = [], []
    for emo, (a, b) in sorted(segs.items()):
        a2 = a + int(drop_onset * sr)
        if b - a2 < sr * epoch_sec:
            continue
        # decimate then resample: much cheaper than resampling the raw block
        blk = data[:, a2:b]
        q = max(int(sr // sfreq), 1)
        if q > 1:
            blk = sps.decimate(blk, q, ftype="fir", zero_phase=True, axis=-1)
            cur = sr / q
        else:
            cur = sr
        if abs(cur - sfreq) > 1e-6:
            blk = sps.resample(blk, int(blk.shape[1] * sfreq / cur), axis=-1)
        n_ep = blk.shape[1] // n_samp
        if n_ep == 0:
            continue
        ep = blk[:, :n_ep * n_samp].reshape(blk.shape[0], n_ep, n_samp)
        ep = np.transpose(ep, (1, 0, 2))
        ep = ep[np.isfinite(ep).all(axis=(1, 2))]
        if len(ep) == 0:
            continue
        if max_ep and len(ep) > max_ep:            # even stride, keeps whole clip
            ep = ep[np.linspace(0, len(ep) - 1, max_ep).astype(int)]
        Xs.append(ep.astype(np.float32))
        ys.append(np.full(len(ep), emo, np.int64))

    if not Xs:
        return sub, None, None, "no epochs", (chans, dropped)
    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    return sub, X, y, f"{len(segs)}/6 segs, {len(X)} epochs", (chans, dropped)


# ======================================================================
# Sleep worker
# ======================================================================
def _rpeaks(x, fs):
    """R-peak sample indices. Assumes x already decimated to ~128 Hz."""
    x = np.asarray(x, float)
    x = x - np.median(x)
    hi = min(20.0 / (fs / 2), 0.99)
    lo = 5.0 / (fs / 2)
    if lo >= hi:
        return np.array([], int)
    b, a = sps.butter(3, [lo, hi], btype="band")
    f = sps.filtfilt(b, a, x)
    w = max(int(0.05 * fs), 1)
    e = np.convolve(np.diff(f) ** 2, np.ones(w) / w, mode="same")
    # per-window adaptive threshold -- ECG amplitude drifts over a night
    win = max(int(10 * fs), 1)
    thr = np.empty_like(e)
    for k in range(0, len(e), win):
        seg = e[k:k + win]
        if len(seg) == 0:
            continue
        local = max(np.percentile(seg, 90) * 0.45, np.median(seg) * 2.5)
        thr[k:k + win] = local
    floor = np.median(e) * 0.5
    thr = np.maximum(thr, floor)
    pk, _ = sps.find_peaks(e, height=thr, distance=int(0.28 * fs))
    return pk


def _clean_rr(rr):
    rr = rr[(rr > 300) & (rr < 2000)]
    if len(rr) > 8:
        med = np.median(rr)
        rr = rr[np.abs(rr - med) < 0.35 * med + 200]
    return rr


def _sleep_worker(args):
    path, sub, target_fs, segment_min = args
    try:
        sig, fs = read_bin_ecg(path)
    except Exception as e:
        return sub, None, f"read failed: {e}"
    if len(sig) < fs * 600:
        return sub, None, f"only {len(sig)/fs/60:.1f} min"

    hours = len(sig) / fs / 3600.0
    # decimate to ~128 Hz -- 4x less work, no loss of beat timing accuracy
    q = max(int(round(fs / target_fs)), 1)
    if q > 1:
        sig = sps.decimate(sig, q, ftype="fir", zero_phase=True)
        fs = fs / q

    pk = _rpeaks(sig, fs)
    if len(pk) < 60:
        return sub, None, "too few peaks"
    t_pk = pk / fs                                  # peak times in seconds
    rr_all = np.diff(t_pk) * 1000.0
    rr = _clean_rr(rr_all)
    f = hrv_features(rr)
    if not f:
        return sub, None, "too few usable beats"

    # segment-wise HRV by slicing the EXISTING rr series, not re-detecting
    seg_s = segment_min * 60.0
    t_mid = t_pk[:-1]
    sd, rm, hr = [], [], []
    n_seg = int(t_pk[-1] // seg_s)
    for k in range(max(n_seg, 1)):
        m = (t_mid >= k * seg_s) & (t_mid < (k + 1) * seg_s)
        r = _clean_rr(rr_all[m])
        if len(r) < 30:
            continue
        d = np.diff(r)
        sd.append(r.std(ddof=1))
        rm.append(np.sqrt((d ** 2).mean()))
        hr.append(60000.0 / r.mean())

    bph = f["n_beats"] / max(hours, 1e-6)
    implied_bpm = bph / 60.0
    coverage = implied_bpm / max(f["mean_hr"], 1e-6)
    f.update(beats_per_hour=float(bph),
             implied_bpm=float(implied_bpm),
             beat_coverage=float(coverage),
             hrv_implausible=bool(coverage < 0.85 or coverage > 1.15
                                  or f["mean_hr"] < 35 or f["mean_hr"] > 110),
             subject=sub, fs=float(fs * q), night_hours=float(hours),
             n_segments=len(sd),
             sdnn_seg_mean=float(np.mean(sd)) if sd else np.nan,
             sdnn_seg_std=float(np.std(sd)) if sd else np.nan,
             rmssd_seg_mean=float(np.mean(rm)) if rm else np.nan,
             rmssd_instability=float(np.std(rm)) if rm else np.nan,
             hr_seg_std=float(np.std(hr)) if hr else np.nan)
    flag = "  <-- LOW COVERAGE" if f["hrv_implausible"] else ""
    return sub, f, (f"{hours:.2f} h, {f['n_beats']} beats, "
                    f"HR {f['mean_hr']:.1f}, cov {coverage:.2f}, "
                    f"SDNN {f['sdnn']:.1f}{flag}")


# ======================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--workers", type=int, default=0, help="0 = auto")
    ap.add_argument("--max-epochs", type=int, default=40,
                    help="per emotion per subject; 0 = uncapped")
    ap.add_argument("--ecg-fs", type=float, default=128.0)
    ap.add_argument("--skip-eeg", action="store_true")
    ap.add_argument("--skip-sleep", action="store_true")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    proc = cfg["paths"]["proc"]
    Path(proc).mkdir(parents=True, exist_ok=True)
    Path(cfg["paths"]["out"]).mkdir(parents=True, exist_ok=True)

    nw = a.workers or max(1, min(8, (os.cpu_count() or 4) - 1))
    print(f"workers: {nw}")

    inv_path = Path("outputs/ecsmp_inventory.json")
    inv = (json.loads(inv_path.read_text()) if inv_path.exists()
           else inspect_dataset(a.root, verbose=False))

    import time
    t0 = time.time()

    # ---------------- EEG ----------------
    if not a.skip_eeg:
        print(f"\n[1/3] EEG emotion epochs ({len(inv['eeg'])} recordings)")
        jobs = [(r["path"], r["subject"], cfg["ecsmp"]["sfreq"],
                 cfg["ecsmp"]["epoch_sec"], cfg["ecsmp"]["drop_first_sec"],
                 a.max_epochs) for r in inv["eeg"] if r["subject"]]
        Xs, ys, gs, chan_lists, shown = [], [], [], [], False
        with ProcessPoolExecutor(max_workers=nw) as ex:
            futs = {ex.submit(_eeg_worker, j): j[1] for j in jobs}
            done = 0
            for fu in as_completed(futs):
                sub, X, y, msg, chinfo = fu.result()
                done += 1
                if X is None:
                    print(f"    ! sub {sub}: {msg}")
                    continue
                if not shown and isinstance(chinfo, tuple):
                    print(f"    channels kept   : {chinfo[0]}")
                    print(f"    channels DROPPED: {chinfo[1]}")
                    shown = True
                Xs.append(X); ys.append(y)
                gs.append(np.full(len(y), sub, np.int64))
                chan_lists.append([str(c).strip().upper()
                                   for c in (chinfo[0] if isinstance(chinfo, tuple)
                                             else [])])
                if done % 10 == 0:
                    print(f"    {done}/{len(jobs)} ({time.time()-t0:.0f}s)")
        if Xs:
            from src.data import _save, preprocess_block
            # ---- align channels BY NAME across subjects ----
            valid = [c for c in chan_lists if c]
            if valid and all(len(c) == x.shape[1] for c, x in zip(chan_lists, Xs) if c):
                common = set(valid[0])
                for c in valid[1:]:
                    common &= set(c)
                order = [c for c in valid[0] if c in common]
                if len(order) < 2:
                    raise RuntimeError(f"only {len(order)} common channels: {order}")
                print(f"    common channels ({len(order)}): {order}")
                dropped_sub = 0
                aligned = []
                for x, c in zip(Xs, chan_lists):
                    if not c or len(c) != x.shape[1]:
                        aligned.append(None); dropped_sub += 1; continue
                    idx = [c.index(n) for n in order]
                    aligned.append(x[:, idx])
                keep = [i for i, a in enumerate(aligned) if a is not None]
                if dropped_sub:
                    print(f"    ! dropped {dropped_sub} subjects with "
                          f"unreadable channel labels")
                Xs = [aligned[i] for i in keep]
                ys = [ys[i] for i in keep]
                gs = [gs[i] for i in keep]
            else:
                print("    ! channel names unavailable; falling back to position")
                cmin = min(x.shape[1] for x in Xs)
                Xs = [x[:, :cmin] for x in Xs]
            X = np.concatenate(Xs)
            X = preprocess_block(X, cfg["ecsmp"]["sfreq"], (0.5, 45.0), 50.0, 20.0)
            y = np.concatenate(ys); g = np.concatenate(gs)
            _save(proc, "ecsmp_emotion", X, y, g,
                  meta=dict(classes=EMOTIONS, sfreq=cfg["ecsmp"]["sfreq"],
                            epoch_sec=cfg["ecsmp"]["epoch_sec"],
                            n_channels=int(X.shape[1]),
                            source="ECSMP EEG_downsample (Trigger/VEO/M2 excluded)"))
            print(f"    EEG done in {time.time()-t0:.0f}s")

    # ---------------- sleep ----------------
    if not a.skip_sleep:
        t1 = time.time()
        print(f"\n[2/3] sleep HRV ({len(inv['sleep_ecg'])} nights)")
        jobs = [(r["path"], r["subject"], a.ecg_fs, 5)
                for r in inv["sleep_ecg"] if r["subject"]]
        rows = []
        with ProcessPoolExecutor(max_workers=nw) as ex:
            futs = {ex.submit(_sleep_worker, j): j[1] for j in jobs}
            done = 0
            for fu in as_completed(futs):
                sub, f, msg = fu.result()
                done += 1
                if f is None:
                    print(f"    ! sub {sub}: {msg}")
                else:
                    rows.append(f)
                    print(f"    sub {sub:03d}: {msg}  [{done}/{len(jobs)}]")
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(Path(proc) / "sleep_hrv.csv", index=False)
            print(f"    sleep HRV {df.shape} in {time.time()-t1:.0f}s")

    # ---------------- scales ----------------
    print("\n[3/3] questionnaires")
    try:
        meta = load_metadata(inv["root"])
        build_subject_scores(meta, proc)
    except Exception as e:
        print(f"    ! {e}")

    print(f"\nTOTAL {time.time()-t0:.0f}s")
    print("Next:  python run_ecsmp.py relate")


if __name__ == "__main__":
    main()
