"""
Training engine.

  augment / NTXent / pretrain_ssl   -- stage 1, self-supervised on unlabelled EEG
  FocalLoss                         -- stage 2 objective
  run_cv                            -- subject-grouped CV, the only honest split
  evaluate                          -- full metric suite + confusion matrix

Hard rule enforced here: splits are made on SUBJECT ids, never on epochs.
Random-splitting EEG epochs puts adjacent 30 s windows from the same night in
both train and test. That inflates accuracy by 10-25 points and is the single
most common reason EEG papers get rejected or fail to replicate.
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix,
                             f1_score, precision_score, recall_score,
                             roc_auc_score, average_precision_score)
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from torch.utils.data import DataLoader, Dataset

from . import models as M


# ======================================================================
# Dataset
# ======================================================================
class EEGDataset(Dataset):
    """Reads from a memmapped array -- keeps RAM flat regardless of dataset size."""
    def __init__(self, X, y=None, idx=None, train=False, aug_cfg=None):
        self.X, self.y = X, y
        self.idx = np.arange(len(X)) if idx is None else np.asarray(idx)
        self.train = train
        self.aug = aug_cfg or {}

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        x = torch.from_numpy(np.asarray(self.X[j], dtype=np.float32))
        if self.y is None:
            return x
        return x, int(self.y[j])


class SSLPairDataset(Dataset):
    """Returns two independently augmented views of the same epoch."""
    def __init__(self, X, idx=None, aug_cfg=None):
        self.X = X
        self.idx = np.arange(len(X)) if idx is None else np.asarray(idx)
        self.aug = aug_cfg or {}

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        x = torch.from_numpy(np.asarray(self.X[self.idx[i]], dtype=np.float32))
        return augment(x, self.aug), augment(x, self.aug)


# ======================================================================
# Augmentations  (physiologically plausible -- see note per transform)
# ======================================================================
def augment(x, cfg):
    """x: (C, T) tensor. Each transform preserves the label semantics."""
    C, T = x.shape
    g = cfg

    # amplitude scaling -- electrode impedance / gain varies between sessions
    lo, hi = g.get("scale_range", (0.8, 1.2))
    x = x * (lo + torch.rand(C, 1) * (hi - lo))

    # additive jitter -- sensor noise
    s = g.get("jitter_sigma", 0.05)
    if s > 0:
        x = x + torch.randn_like(x) * s

    # random temporal crop + resize -- tolerance to timing offsets
    cf = g.get("crop_frac", 0.85)
    if cf < 1.0:
        L = int(T * cf)
        st = int(torch.randint(0, T - L + 1, (1,)))
        x = F.interpolate(x[:, st:st + L].unsqueeze(0), size=T,
                          mode="linear", align_corners=False).squeeze(0)

    # channel-wise time masking -- robustness to dropout / movement artifact
    mf = g.get("mask_frac", 0.15)
    if mf > 0:
        L = int(T * mf)
        if L > 0:
            st = int(torch.randint(0, T - L + 1, (1,)))
            ch = int(torch.randint(0, C, (1,)))
            x = x.clone()
            x[ch, st:st + L] = 0.0

    # polarity flip -- EEG sign is reference-dependent, not information-bearing
    if torch.rand(1).item() < g.get("flip_prob", 0.3):
        x = -x
    return x


# ======================================================================
# SSL: NT-Xent (SimCLR)
# ======================================================================
def nt_xent(z1, z2, temperature=0.2):
    B = z1.size(0)
    z = torch.cat([z1, z2], 0)                        # (2B, D), pre-normalised
    sim = (z @ z.T) / temperature
    sim.fill_diagonal_(-1e9)
    targets = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(z.device)
    return F.cross_entropy(sim, targets)


def pretrain_ssl(X, cfg, backbone_name="cnn1d", n_ch=2, device="cuda",
                 out_path="outputs/ssl_encoder.pt", log_every=5):
    """Stage 1. Contrastive pretraining on UNLABELLED epochs.

    Run this on Sleep-EDF (large, unlabelled use), then load the weights into
    the supervised model for the emotion / sleep-deprivation task. That transfer
    step IS the paper's main claim -- keep the encoder identical between stages.
    """
    device = _device(device)
    scfg = cfg["ssl"]
    net = M.build(backbone_name, n_ch=n_ch, n_classes=2)
    ssl = M.SSLWrapper(net, scfg["proj_dim"]).to(device)

    dl = DataLoader(SSLPairDataset(X, aug_cfg=scfg["augment"]),
                    batch_size=scfg["batch_size"], shuffle=True,
                    num_workers=_workers(), drop_last=True, pin_memory=True)
    opt = torch.optim.AdamW(ssl.parameters(), lr=scfg["lr"],
                            weight_decay=scfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, scfg["epochs"])
    amp = cfg.get("amp", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    hist = []
    for ep in range(scfg["epochs"]):
        ssl.train(); tot = n = 0
        for v1, v2 in dl:
            v1, v2 = v1.to(device, non_blocking=True), v2.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp):
                loss = nt_xent(ssl(v1), ssl(v2), scfg["temperature"])
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(ssl.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            tot += loss.item() * v1.size(0); n += v1.size(0)
        sched.step()
        hist.append(tot / max(n, 1))
        if ep % log_every == 0 or ep == scfg["epochs"] - 1:
            print(f"  ssl epoch {ep:3d}  loss {hist[-1]:.4f}")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"backbone": net.state_dict(), "name": backbone_name,
                "n_ch": n_ch, "history": hist}, out_path)
    print(f"  saved encoder -> {out_path}")
    return hist


def load_pretrained(model, ckpt_path, verbose=True):
    """Load SSL weights, skipping the classification head (shape mismatch is expected)."""
    sd = torch.load(ckpt_path, map_location="cpu")["backbone"]
    own = model.state_dict()
    ok = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape
          and not k.startswith("head")}
    own.update(ok)
    model.load_state_dict(own)
    if verbose:
        print(f"  transferred {len(ok)}/{len(own)} tensors from {ckpt_path}")
    return model


# ======================================================================
# Losses
# ======================================================================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, smoothing=0.05):
        super().__init__()
        self.gamma, self.smoothing = gamma, smoothing
        self.register_buffer("weight", weight if weight is not None else torch.tensor([]))

    def forward(self, logits, target):
        K = logits.size(-1)
        logp = F.log_softmax(logits, -1)
        with torch.no_grad():
            t = torch.full_like(logp, self.smoothing / (K - 1))
            t.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        p = logp.exp()
        pt = (t * p).sum(-1)
        ce = -(t * logp).sum(-1)
        loss = ((1 - pt) ** self.gamma) * ce
        if self.weight.numel():
            loss = loss * self.weight.to(logits.device)[target]
        return loss.mean()


def class_weights(y, n_classes, scheme="inverse_freq"):
    cnt = np.bincount(y, minlength=n_classes).astype(np.float64)
    cnt[cnt == 0] = 1.0
    if scheme == "inverse_freq":
        w = cnt.sum() / (n_classes * cnt)
    elif scheme == "sqrt_inverse":
        w = np.sqrt(cnt.sum() / (n_classes * cnt))
    else:
        w = np.ones(n_classes)
    return torch.tensor(w / w.mean(), dtype=torch.float32)


# ======================================================================
# Metrics
# ======================================================================
def compute_metrics(y_true, y_pred, y_prob, n_classes):
    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "kappa": cohen_kappa_score(y_true, y_pred),
    }
    try:
        if n_classes == 2:
            m["roc_auc"] = roc_auc_score(y_true, y_prob[:, 1])
            m["pr_auc"] = average_precision_score(y_true, y_prob[:, 1])
        else:
            present = np.unique(y_true)
            m["roc_auc"] = roc_auc_score(y_true, y_prob[:, present],
                                         multi_class="ovr", average="macro",
                                         labels=present)
            m["pr_auc"] = np.mean([
                average_precision_score((y_true == c).astype(int), y_prob[:, c])
                for c in present])
    except Exception:
        m["roc_auc"] = float("nan"); m["pr_auc"] = float("nan")
    m["per_class_f1"] = f1_score(y_true, y_pred, average=None,
                                 labels=range(n_classes), zero_division=0).tolist()
    m["confusion"] = confusion_matrix(y_true, y_pred,
                                      labels=range(n_classes)).tolist()
    return m


# ======================================================================
# Single fold
# ======================================================================
def train_fold(X, y, tr_idx, va_idx, cfg, model_name, n_classes,
               device="cuda", pretrained=None, verbose=True):
    device = _device(device)
    tcfg = cfg["train"]
    n_ch = X.shape[1]

    model = M.build(model_name, n_ch=n_ch, n_classes=n_classes).to(device)
    if pretrained:
        load_pretrained(model, pretrained, verbose=verbose)

    w = class_weights(y[tr_idx], n_classes, tcfg["class_weighting"])
    crit = FocalLoss(tcfg["focal_gamma"], w, tcfg["label_smoothing"]).to(device)

    dl_tr = DataLoader(EEGDataset(X, y, tr_idx, train=True),
                       batch_size=tcfg["batch_size"], shuffle=True,
                       num_workers=_workers(), pin_memory=True, drop_last=True)
    dl_va = DataLoader(EEGDataset(X, y, va_idx), batch_size=tcfg["batch_size"] * 2,
                       shuffle=False, num_workers=_workers(), pin_memory=True)

    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"],
                            weight_decay=tcfg["weight_decay"])
    total = tcfg["epochs"]
    warm = tcfg["warmup_epochs"]

    def lr_at(e):
        if e < warm:
            return (e + 1) / max(warm, 1)
        p = (e - warm) / max(total - warm, 1)
        return 0.5 * (1 + np.cos(np.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    amp = cfg.get("amp", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    best_f1, best_state, patience, hist = -1.0, None, 0, []
    for ep in range(total):
        model.train(); tot = n = 0
        for xb, yb in dl_tr:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp):
                loss = crit(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
            scaler.step(opt); scaler.update()
            tot += loss.item() * xb.size(0); n += xb.size(0)
        sched.step()

        yt, yp, pr = predict(model, dl_va, device, amp)
        f1 = f1_score(yt, yp, average="macro", zero_division=0)
        hist.append({"epoch": ep, "train_loss": tot / max(n, 1), "val_f1": f1})
        if verbose and (ep % 5 == 0 or ep == total - 1):
            print(f"    ep {ep:3d} loss {tot/max(n,1):.4f} val_macroF1 {f1:.4f}")

        if f1 > best_f1:
            best_f1, patience = f1, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience += 1
            if patience >= tcfg["early_stop_patience"]:
                if verbose:
                    print(f"    early stop at epoch {ep}")
                break

    model.load_state_dict(best_state)
    yt, yp, pr = predict(model, dl_va, device, amp)
    return model, compute_metrics(yt, yp, pr, n_classes), hist


@torch.no_grad()
def predict(model, loader, device, amp=False):
    model.eval()
    P, Y = [], []
    for batch in loader:
        xb, yb = (batch if isinstance(batch, (list, tuple)) else (batch, None))
        xb = xb.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp):
            out = model(xb)
        P.append(torch.softmax(out.float(), -1).cpu().numpy())
        if yb is not None:
            Y.append(yb.numpy())
    P = np.concatenate(P)
    Y = np.concatenate(Y) if Y else None
    return Y, P.argmax(1), P


# ======================================================================
# Cross-validation
# ======================================================================
def run_cv(X, y, groups, cfg, model_name, n_classes, device="cuda",
           pretrained=None, tag="run", verbose=True):
    scheme = cfg["cv"]["scheme"]
    splitter = (LeaveOneGroupOut() if scheme == "loso"
                else GroupKFold(n_splits=cfg["cv"]["n_folds"]))
    folds, oof_pred = [], np.zeros(len(y), dtype=np.int64)
    oof_prob = np.zeros((len(y), n_classes), dtype=np.float32)
    t0 = time.time()

    for k, (tr, va) in enumerate(splitter.split(X, y, groups)):
        if verbose:
            print(f"  fold {k}: train {len(tr)} ep / {len(np.unique(groups[tr]))} subj"
                  f" | val {len(va)} ep / {len(np.unique(groups[va]))} subj")
        model, met, hist = train_fold(X, y, tr, va, cfg, model_name,
                                      n_classes, device, pretrained, verbose)
        met["fold"] = k
        met["val_subjects"] = np.unique(groups[va]).tolist()
        folds.append(met)
        _, yp, pr = predict(model, DataLoader(EEGDataset(X, y, va),
                                              batch_size=cfg["train"]["batch_size"] * 2),
                            _device(device))
        oof_pred[va] = yp
        oof_prob[va] = pr
        if verbose:
            print(f"  fold {k}: acc {met['accuracy']:.4f}  macroF1 "
                  f"{met['balanced_f1_macro']:.4f}  kappa {met['kappa']:.4f}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    keys = ["accuracy", "balanced_f1_macro", "precision_macro", "recall_macro",
            "kappa", "roc_auc", "pr_auc"]
    summary = {k: {"mean": float(np.nanmean([f[k] for f in folds])),
                   "std": float(np.nanstd([f[k] for f in folds]))} for k in keys}
    result = {"tag": tag, "model": model_name, "scheme": scheme,
              "n_folds": len(folds), "elapsed_sec": time.time() - t0,
              "summary": summary, "folds": folds,
              "oof": compute_metrics(y, oof_pred, oof_prob, n_classes)}

    out = Path(cfg["paths"]["out"]) / f"cv_{tag}_{model_name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    if verbose:
        print(f"  == {model_name}: acc {summary['accuracy']['mean']:.4f}"
              f" +/- {summary['accuracy']['std']:.4f} | macroF1 "
              f"{summary['balanced_f1_macro']['mean']:.4f}"
              f" +/- {summary['balanced_f1_macro']['std']:.4f}"
              f"  ({result['elapsed_sec']:.0f}s)")
    return result, oof_pred, oof_prob


# ======================================================================
def _device(d):
    if isinstance(d, torch.device):
        return d
    if d == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(d)


def _workers():
    import os
    return min(4, max(0, (os.cpu_count() or 2) - 1))


def set_seed(s=42):
    import random, os
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    os.environ["PYTHONHASHSEED"] = str(s)
    torch.backends.cudnn.deterministic = False   # True halves speed; keep False
    torch.backends.cudnn.benchmark = True
