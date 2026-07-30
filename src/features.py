"""
Classical EEG features -- powers the RF / XGBoost / LightGBM baselines and
gives you the SHAP-interpretable feature table for the paper.

extract(X, sfreq) -> (F, names)
    X : (N, C, T) float32
    F : (N, D)    float32

Design note: every feature is computed per channel, then connectivity features
are added per channel-pair. With C=2 (Sleep-EDF) D is ~60. With C=61
(ds004902) the pairwise block explodes, so connectivity is capped by
`max_pairs` and computed on a fixed random subset of pairs (seeded).
"""
from __future__ import annotations

import math
import numpy as np
from scipy import signal as sps
from scipy.stats import skew, kurtosis

BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "sigma": (12.0, 16.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


# ---------------- time domain ----------------
def _hjorth(x, axis=-1):
    dx = np.diff(x, axis=axis)
    ddx = np.diff(dx, axis=axis)
    v0 = np.var(x, axis=axis) + 1e-12
    v1 = np.var(dx, axis=axis) + 1e-12
    v2 = np.var(ddx, axis=axis) + 1e-12
    mob = np.sqrt(v1 / v0)
    comp = np.sqrt(v2 / v1) / mob
    return v0, mob, comp


def _time_feats(X):
    f = [
        X.mean(-1), X.std(-1), skew(X, axis=-1), kurtosis(X, axis=-1),
        np.ptp(X, axis=-1),
        np.sqrt((X ** 2).mean(-1)),                       # RMS
        (np.diff(np.signbit(X), axis=-1) != 0).sum(-1),   # zero crossings
        np.abs(np.diff(X, axis=-1)).mean(-1),             # line length
    ]
    names = ["mean", "std", "skew", "kurt", "ptp", "rms", "zcr", "linelen"]
    a, m, c = _hjorth(X)
    f += [a, m, c]
    names += ["hjorth_act", "hjorth_mob", "hjorth_comp"]
    return np.stack(f, -1), names          # (N, C, F)


# ---------------- frequency domain ----------------
def _spectral_feats(X, sfreq):
    nper = min(X.shape[-1], int(4 * sfreq))
    freqs, psd = sps.welch(X, fs=sfreq, nperseg=nper, axis=-1)
    psd = psd + 1e-20
    total = np.trapezoid(psd, freqs, axis=-1) + 1e-20

    feats, names = [], []
    bp = {}
    for name, (lo, hi) in BANDS.items():
        m = (freqs >= lo) & (freqs < hi)
        p = np.trapezoid(psd[..., m], freqs[m], axis=-1)
        bp[name] = p
        feats += [np.log(p + 1e-20), p / total]
        names += [f"logbp_{name}", f"relbp_{name}"]

    for a, b in [("theta", "beta"), ("alpha", "theta"), ("delta", "alpha"),
                 ("delta", "beta"), ("sigma", "delta")]:
        feats.append(np.log((bp[a] + 1e-20) / (bp[b] + 1e-20)))
        names.append(f"ratio_{a}_{b}")

    # spectral shape descriptors
    pn = psd / psd.sum(-1, keepdims=True)
    centroid = (pn * freqs).sum(-1)
    spread = np.sqrt((pn * (freqs - centroid[..., None]) ** 2).sum(-1))
    entropy = -(pn * np.log(pn + 1e-20)).sum(-1) / np.log(pn.shape[-1])
    cum = np.cumsum(pn, -1)
    edge95 = freqs[np.argmax(cum >= 0.95, axis=-1)]
    feats += [centroid, spread, entropy, edge95, np.log(total)]
    names += ["spec_centroid", "spec_spread", "spec_entropy", "spec_edge95", "log_total_power"]
    return np.stack(feats, -1), names


# ---------------- nonlinear ----------------
def _perm_entropy(x, order=3, delay=1):
    n = x.shape[-1] - (order - 1) * delay
    if n <= 1:
        return np.zeros(x.shape[:-1])
    idx = np.arange(order) * delay
    emb = np.stack([x[..., i: i + n] for i in idx], -1)
    perms = np.argsort(emb, axis=-1)
    # encode each permutation as an integer via mixed radix
    code = (perms * (order ** np.arange(order))).sum(-1)
    out = np.zeros(x.shape[:-1])
    flat_code = code.reshape(-1, n)
    flat_out = np.empty(flat_code.shape[0])
    for i in range(flat_code.shape[0]):
        _, cnt = np.unique(flat_code[i], return_counts=True)
        p = cnt / cnt.sum()
        flat_out[i] = -(p * np.log(p)).sum() / np.log(math.factorial(order))
    return flat_out.reshape(out.shape)


def _lziv(x):
    """Lempel-Ziv complexity of the median-binarised signal."""
    b = (x > np.median(x, -1, keepdims=True)).astype(np.uint8)
    flat = b.reshape(-1, b.shape[-1])
    out = np.empty(flat.shape[0])
    for k in range(flat.shape[0]):
        s = flat[k].tobytes()
        i, c, ln = 0, 1, 1
        n = len(s)
        while i + ln < n:
            if s[i: i + ln] == s[i + ln: i + 2 * ln]:
                ln += 1
            else:
                c += 1
                i += ln
                ln = 1
        out[k] = c * np.log2(n) / n
    return out.reshape(x.shape[:-1])


def _higuchi_fd(x, kmax=8):
    N = x.shape[-1]
    flat = x.reshape(-1, N)
    out = np.empty(flat.shape[0])
    ks = np.arange(1, kmax + 1)
    for i in range(flat.shape[0]):
        L = []
        for k in ks:
            Lk = []
            for m in range(k):
                idx = np.arange(m, N, k)
                if len(idx) < 2:
                    continue
                lm = np.abs(np.diff(flat[i][idx])).sum()
                lm = lm * (N - 1) / (len(idx) - 1) / k
                Lk.append(lm)
            L.append(np.mean(Lk) if Lk else 1e-12)
        L = np.log(np.array(L) + 1e-20)
        out[i] = -np.polyfit(np.log(ks), L, 1)[0]
    return out.reshape(x.shape[:-1])


def _nonlinear_feats(X, fast=True):
    f = [_perm_entropy(X), _lziv(X)]
    names = ["perm_entropy", "lziv"]
    if not fast:
        f.append(_higuchi_fd(X))
        names.append("higuchi_fd")
    return np.stack(f, -1), names


# ---------------- connectivity ----------------
def _connectivity(X, sfreq, max_pairs=64, seed=42):
    N, C, T = X.shape
    if C < 2:
        return np.zeros((N, 0), np.float32), []
    pairs = [(i, j) for i in range(C) for j in range(i + 1, C)]
    if len(pairs) > max_pairs:
        rng = np.random.default_rng(seed)
        pairs = [pairs[k] for k in rng.choice(len(pairs), max_pairs, replace=False)]

    analytic = sps.hilbert(X, axis=-1)
    phase = np.angle(analytic)

    feats, names = [], []
    for i, j in pairs:
        dphi = phase[:, i] - phase[:, j]
        plv = np.abs(np.exp(1j * dphi).mean(-1))
        pli = np.abs(np.sign(np.sin(dphi)).mean(-1))
        a, b = X[:, i], X[:, j]
        a0 = a - a.mean(-1, keepdims=True)
        b0 = b - b.mean(-1, keepdims=True)
        corr = (a0 * b0).mean(-1) / (a0.std(-1) * b0.std(-1) + 1e-12)
        feats += [plv, pli, corr]
        names += [f"plv_{i}_{j}", f"pli_{i}_{j}", f"corr_{i}_{j}"]
    return np.stack(feats, -1).astype(np.float32), names


# ---------------- public API ----------------
def extract(X, sfreq, fast=True, max_pairs=64, batch=512):
    """Extract the full feature table. Batched so it never blows up RAM."""
    outs = []
    names = None
    for s in range(0, len(X), batch):
        xb = np.asarray(X[s: s + batch], dtype=np.float64)
        t, nt = _time_feats(xb)
        f, nf = _spectral_feats(xb, sfreq)
        nl, nnl = _nonlinear_feats(xb, fast=fast)
        per_ch = np.concatenate([t, f, nl], -1)              # (n, C, D)
        n, C, D = per_ch.shape
        flat = per_ch.reshape(n, C * D)
        conn, ncn = _connectivity(xb, sfreq, max_pairs=max_pairs)
        outs.append(np.concatenate([flat, conn], -1).astype(np.float32))
        if names is None:
            ch_names = [f"ch{c}_{nm}" for c in range(C) for nm in (nt + nf + nnl)]
            names = ch_names + ncn
    F = np.concatenate(outs, 0)
    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    return F, names
