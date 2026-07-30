"""HRV extraction from raw ECG. Shared by the sleep and experiment recordings."""
from __future__ import annotations

import numpy as np
from scipy import signal as sps


def rr_intervals(ecg, fs):
    """R-peak detection -> RR intervals in ms.

    Bandpass 5-20 Hz, squared derivative, moving-average energy, adaptive
    threshold, refractory period 300 ms. Deliberately simple: adequate for
    night-level HRV summaries, not validated for beat-level arrhythmia work.
    """
    x = np.asarray(ecg, float).ravel()
    x = x[np.isfinite(x)]
    if len(x) < fs * 30:
        return np.array([])
    x = x - np.median(x)
    hi = min(20.0 / (fs / 2), 0.99)
    lo = 5.0 / (fs / 2)
    if lo >= hi:
        return np.array([])
    b, a = sps.butter(3, [lo, hi], btype="band")
    f = sps.filtfilt(b, a, x)
    w = max(int(0.05 * fs), 1)
    e = np.convolve(np.diff(f) ** 2, np.ones(w) / w, mode="same")
    thr = np.percentile(e, 98) * 0.35
    pk, _ = sps.find_peaks(e, height=thr, distance=int(0.3 * fs))
    rr = np.diff(pk) / fs * 1000.0
    rr = rr[(rr > 300) & (rr < 2000)]
    if len(rr) > 8:
        med = np.median(rr)
        rr = rr[np.abs(rr - med) < 0.35 * med + 200]
    return rr


def hrv_features(rr):
    if len(rr) < 30:
        return {}
    d = np.diff(rr)
    out = {
        "mean_rr": float(rr.mean()),
        "mean_hr": float(60000.0 / rr.mean()),
        "sdnn": float(rr.std(ddof=1)),
        "rmssd": float(np.sqrt((d ** 2).mean())),
        "sdsd": float(d.std(ddof=1)),
        "pnn20": float((np.abs(d) > 20).mean() * 100),
        "pnn50": float((np.abs(d) > 50).mean() * 100),
        "cvnn": float(rr.std(ddof=1) / rr.mean()),
        "n_beats": int(len(rr)),
    }
    t = np.cumsum(rr) / 1000.0
    if len(t) > 60 and t[-1] > 120:
        fs_i = 4.0
        ti = np.arange(t[0], t[-1], 1 / fs_i)
        xi = np.interp(ti, t, rr)
        xi = xi - xi.mean()
        fr, ps = sps.welch(xi, fs=fs_i, nperseg=min(len(xi), int(fs_i * 120)))

        def band(a, b):
            m = (fr >= a) & (fr < b)
            return float(np.trapezoid(ps[m], fr[m])) if m.any() else 0.0

        vlf, lf, hf = band(0.0033, 0.04), band(0.04, 0.15), band(0.15, 0.4)
        out.update(vlf=vlf, lf=lf, hf=hf,
                   lf_hf=float(lf / hf) if hf > 0 else np.nan,
                   total_power=float(vlf + lf + hf),
                   lf_nu=float(lf / (lf + hf)) if (lf + hf) > 0 else np.nan)
    return out
