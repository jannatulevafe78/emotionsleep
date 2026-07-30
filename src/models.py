"""
Architectures. All share one contract:

    forward(x) -> logits            x : (B, C, T)
    encode(x)  -> embedding (B, E)  used for SSL, t-SNE, transfer

Parameter counts are deliberately small (0.1M - 3M). On <=6 GB VRAM with
30 s @ 100 Hz epochs (T=3000) and batch 64, all of these fit comfortably in
mixed precision. Do not scale these up before you have a baseline number --
capacity is not your bottleneck, subject-level generalisation is.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# building blocks
# ======================================================================
class SEBlock(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        h = max(c // r, 4)
        self.fc = nn.Sequential(nn.Linear(c, h), nn.SiLU(), nn.Linear(h, c), nn.Sigmoid())

    def forward(self, x):
        w = self.fc(x.mean(-1))
        return x * w.unsqueeze(-1)


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, k=7, s=1, dilation=1, drop=0.1):
        super().__init__()
        pad = dilation * (k - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(cin, cout, k, stride=s, padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm1d(cout), nn.SiLU(), nn.Dropout(drop),
        )
        self.se = SEBlock(cout)
        self.skip = (nn.Identity() if (cin == cout and s == 1)
                     else nn.Conv1d(cin, cout, 1, stride=s, bias=False))

    def forward(self, x):
        return self.se(self.net(x)) + self.skip(x)


class MultiScaleStem(nn.Module):
    """Parallel small/medium/large kernels -- EEG carries information at very
    different timescales (spindles ~0.5 s, slow waves ~1-2 s, K-complexes)."""
    def __init__(self, cin, cout):
        super().__init__()
        per = cout // 3
        rest = cout - 2 * per
        self.b1 = nn.Conv1d(cin, per, 7, stride=2, padding=3, bias=False)
        self.b2 = nn.Conv1d(cin, per, 25, stride=2, padding=12, bias=False)
        self.b3 = nn.Conv1d(cin, rest, 101, stride=2, padding=50, bias=False)
        self.bn = nn.BatchNorm1d(cout)

    def forward(self, x):
        return F.silu(self.bn(torch.cat([self.b1(x), self.b2(x), self.b3(x)], 1)))


# ======================================================================
# 1. CNN1D
# ======================================================================
class CNN1D(nn.Module):
    def __init__(self, n_ch=2, n_classes=5, width=32, drop=0.2):
        super().__init__()
        w = width
        self.stem = MultiScaleStem(n_ch, w)
        self.body = nn.Sequential(
            ConvBlock(w, w, 7), nn.MaxPool1d(4),
            ConvBlock(w, w * 2, 7), ConvBlock(w * 2, w * 2, 7, dilation=2), nn.MaxPool1d(4),
            ConvBlock(w * 2, w * 4, 5), ConvBlock(w * 4, w * 4, 5, dilation=4), nn.MaxPool1d(4),
            ConvBlock(w * 4, w * 4, 3, dilation=8),
        )
        self.emb_dim = w * 8
        self.drop = nn.Dropout(drop)
        self.head = nn.Linear(self.emb_dim, n_classes)

    def encode(self, x):
        h = self.body(self.stem(x))
        return torch.cat([h.mean(-1), h.amax(-1)], -1)   # avg + max pooling

    def forward(self, x):
        return self.head(self.drop(self.encode(x)))


# ======================================================================
# 2. CRNN  (CNN feature extractor + BiGRU over time)
# ======================================================================
class CRNN(nn.Module):
    def __init__(self, n_ch=2, n_classes=5, width=32, hidden=64, drop=0.2):
        super().__init__()
        w = width
        self.stem = MultiScaleStem(n_ch, w)
        self.body = nn.Sequential(
            ConvBlock(w, w * 2, 7), nn.MaxPool1d(4),
            ConvBlock(w * 2, w * 4, 5), nn.MaxPool1d(4),
            ConvBlock(w * 4, w * 4, 3), nn.MaxPool1d(2),
        )
        self.rnn = nn.GRU(w * 4, hidden, num_layers=2, batch_first=True,
                          bidirectional=True, dropout=drop)
        self.emb_dim = hidden * 4
        self.drop = nn.Dropout(drop)
        self.head = nn.Linear(self.emb_dim, n_classes)

    def encode(self, x):
        h = self.body(self.stem(x)).transpose(1, 2)      # (B, L, C)
        o, _ = self.rnn(h)
        return torch.cat([o.mean(1), o.amax(1)], -1)

    def forward(self, x):
        return self.head(self.drop(self.encode(x)))


# ======================================================================
# 3. Transformer over conv patches
# ======================================================================
class PosEnc(nn.Module):
    def __init__(self, d, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class EEGTransformer(nn.Module):
    def __init__(self, n_ch=2, n_classes=5, d=128, depth=4, heads=4,
                 patch=50, drop=0.1):
        super().__init__()
        self.tok = nn.Sequential(
            MultiScaleStem(n_ch, 64),
            ConvBlock(64, d, 7), nn.MaxPool1d(4),
            ConvBlock(d, d, 5), nn.MaxPool1d(4),
        )
        self.pos = PosEnc(d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.trunc_normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d, heads, dim_feedforward=d * 4, dropout=drop,
            activation="gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(d)
        self.emb_dim = d * 2
        self.head = nn.Linear(self.emb_dim, n_classes)
        self._attn = None

    def encode(self, x, keep_tokens=False):
        h = self.tok(x).transpose(1, 2)
        h = self.pos(h)
        h = torch.cat([self.cls.expand(h.size(0), -1, -1), h], 1)
        h = self.norm(self.enc(h))
        if keep_tokens:
            return h
        return torch.cat([h[:, 0], h[:, 1:].mean(1)], -1)

    def forward(self, x):
        return self.head(self.encode(x))


# ======================================================================
# 4. Channel-graph GNN  (spatial relations between electrodes)
# ======================================================================
class GraphConv(nn.Module):
    """Dense graph conv with a LEARNED adjacency over channels.

    Learning A rather than fixing it to electrode distance is the point: the
    adjacency you learn is itself a result -- it is the functional connectivity
    the model considers useful, and you can plot it as a figure.
    """
    def __init__(self, din, dout, n_nodes):
        super().__init__()
        self.lin = nn.Linear(din, dout)
        self.adj = nn.Parameter(torch.randn(n_nodes, n_nodes) * 0.05)
        self.bn = nn.BatchNorm1d(n_nodes)

    def forward(self, x):                      # x: (B, N, D)
        a = torch.softmax(self.adj + self.adj.T, dim=-1)
        return F.silu(self.bn(self.lin(a @ x)))


class EEGGNN(nn.Module):
    def __init__(self, n_ch=2, n_classes=5, width=32, d=64, drop=0.2):
        super().__init__()
        self.n_ch = n_ch
        # shared per-channel temporal encoder (weights shared across electrodes)
        self.temporal = nn.Sequential(
            MultiScaleStem(1, width),
            ConvBlock(width, width * 2, 7), nn.MaxPool1d(4),
            ConvBlock(width * 2, width * 2, 5), nn.MaxPool1d(4),
            ConvBlock(width * 2, d, 3),
        )
        self.g1 = GraphConv(d, d, n_ch)
        self.g2 = GraphConv(d, d, n_ch)
        self.emb_dim = d * 2
        self.drop = nn.Dropout(drop)
        self.head = nn.Linear(self.emb_dim, n_classes)

    def encode(self, x):
        B, C, T = x.shape
        h = self.temporal(x.reshape(B * C, 1, T))       # (B*C, d, L)
        h = h.mean(-1).reshape(B, C, -1)                # (B, C, d)  node features
        h = self.g2(self.g1(h)) + h
        return torch.cat([h.mean(1), h.amax(1)], -1)

    def forward(self, x):
        return self.head(self.drop(self.encode(x)))

    def adjacency(self):
        a = self.g1.adj.detach()
        return torch.softmax(a + a.T, -1).cpu().numpy()


# ======================================================================
# 5. Fusion  (conv trunk -> transformer -> graph, with uncertainty head)
# ======================================================================
class FusionNet(nn.Module):
    def __init__(self, n_ch=2, n_classes=5, width=32, d=96, heads=4,
                 depth=2, drop=0.15):
        super().__init__()
        self.n_ch = n_ch
        self.temporal = nn.Sequential(
            MultiScaleStem(1, width),
            ConvBlock(width, width * 2, 7), nn.MaxPool1d(4),
            ConvBlock(width * 2, d, 5), nn.MaxPool1d(4),
            ConvBlock(d, d, 3),
        )
        layer = nn.TransformerEncoderLayer(d, heads, d * 4, drop,
                                           activation="gelu",
                                           batch_first=True, norm_first=True)
        self.temporal_tf = nn.TransformerEncoder(layer, depth)
        self.graph = GraphConv(d, d, n_ch) if n_ch > 1 else None
        self.emb_dim = d * 2
        self.drop = nn.Dropout(drop)
        self.gate = nn.Sequential(nn.Linear(self.emb_dim, self.emb_dim), nn.Sigmoid())
        self.head = nn.Linear(self.emb_dim, n_classes)
        self.unc = nn.Sequential(nn.Linear(self.emb_dim, 64), nn.SiLU(),
                                 nn.Linear(64, n_classes), nn.Softplus())

    def encode(self, x):
        B, C, T = x.shape
        h = self.temporal(x.reshape(B * C, 1, T))        # (B*C, d, L)
        h = h.transpose(1, 2)                            # (B*C, L, d)
        h = self.temporal_tf(h).mean(1)                  # (B*C, d)
        h = h.reshape(B, C, -1)
        if self.graph is not None:
            h = self.graph(h) + h
        return torch.cat([h.mean(1), h.amax(1)], -1)

    def forward(self, x, return_unc=False):
        z = self.drop(self.encode(x))
        z = z * self.gate(z)
        logits = self.head(z)
        if return_unc:
            return logits, self.unc(z)
        return logits


# ======================================================================
# 6. SSL wrapper: projection head for contrastive pretraining
# ======================================================================
class SSLWrapper(nn.Module):
    def __init__(self, backbone, proj_dim=128):
        super().__init__()
        self.backbone = backbone
        e = backbone.emb_dim
        self.proj = nn.Sequential(nn.Linear(e, e), nn.BatchNorm1d(e),
                                  nn.SiLU(), nn.Linear(e, proj_dim))

    def forward(self, x):
        return F.normalize(self.proj(self.backbone.encode(x)), dim=-1)


# ======================================================================
REGISTRY = {
    "cnn1d": CNN1D,
    "crnn": CRNN,
    "transformer": EEGTransformer,
    "gnn": EEGGNN,
    "fusion": FusionNet,
}


def build(name, n_ch, n_classes, **kw):
    if name not in REGISTRY:
        raise KeyError(f"unknown model '{name}'. options: {list(REGISTRY)}")
    return REGISTRY[name](n_ch=n_ch, n_classes=n_classes, **kw)


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
