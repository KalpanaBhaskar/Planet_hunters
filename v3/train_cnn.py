"""
v3 multi-view 1D CNN.

Inputs per star:
    - raw high-pass/low-pass/centroid light curves (4 x 315)
    - folded profiles at four periods (16 x 32)
    - stellar metadata (29 values)

The CNN learns shapes that handcrafted features miss: transit ingress/egress,
V-shaped eclipses, flare decay, starspot harmonics, and centroid motion.

Outputs:
    v3/cache/oof_cnn.npy
    v3/cache/test_cnn.npy
"""

from __future__ import annotations

import copy
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

ROOT = Path(r"C:\Users\Kalpana\Documents\GitHub\Planet_hunters")
C = ROOT / "v3" / "cache"
V2 = ROOT / "v2" / "cache"

torch.set_num_threads(6)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
torch.manual_seed(42)
np.random.seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 12        # 22
PATIENCE = 7
BATCH = 128
LR = 1.5e-3

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
raw_train = np.load(C / ".." / ".." / "v2" / "cache" / "deep_train.npy").astype(np.float32)
raw_test = np.load(C / ".." / ".." / "v2" / "cache" / "deep_test.npy").astype(np.float32)
fold_train = np.load(C / "fold_train.npy").astype(np.float32).reshape(len(raw_train), -1, 32)
fold_test = np.load(C / "fold_test.npy").astype(np.float32).reshape(len(raw_test), -1, 32)
meta_train_df = pd.read_pickle(C / ".." / ".." / "v2" / "cache" / "meta_train.pkl")
meta_test_df = pd.read_pickle(C / ".." / ".." / "v2" / "cache" / "meta_test.pkl")
meta_test_df = meta_test_df.reindex(columns=meta_train_df.columns, fill_value=0.0)
meta_train = meta_train_df.values.astype(np.float32)
meta_test = meta_test_df.values.astype(np.float32)

y = np.load(V2 / "y.npy", allow_pickle=True)
classes = np.load(V2 / "classes.npy", allow_pickle=True)
yi = np.searchsorted(classes, y).astype(np.int64)

# Standardize metadata using training statistics only.
meta_mu = meta_train.mean(0, keepdims=True)
meta_sd = meta_train.std(0, keepdims=True) + 1e-6
meta_train = (meta_train - meta_mu) / meta_sd
meta_test = (meta_test - meta_mu) / meta_sd
meta_train = np.nan_to_num(meta_train, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
meta_test = np.nan_to_num(meta_test, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

Xraw = torch.tensor(raw_train, dtype=torch.float32, device=DEVICE)
Xfold = torch.tensor(fold_train, dtype=torch.float32, device=DEVICE)
M = torch.tensor(meta_train, dtype=torch.float32, device=DEVICE)
y_t = torch.tensor(yi, dtype=torch.long, device=DEVICE)
Xraw_test = torch.tensor(raw_test, dtype=torch.float32, device=DEVICE)
Xfold_test = torch.tensor(fold_test, dtype=torch.float32, device=DEVICE)
M_test = torch.tensor(meta_test, dtype=torch.float32, device=DEVICE)

print(
    "cnn data",
    Xraw.shape,
    Xfold.shape,
    M.shape,
    "device",
    DEVICE,
    flush=True,
)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class Net(nn.Module):
    def __init__(self, nmeta: int):
        super().__init__()
        self.raw = nn.Sequential(
            nn.Conv1d(4, 32, 9, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.fold = nn.Sequential(
            nn.Conv1d(16, 32, 5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.meta = nn.Sequential(
            nn.Linear(nmeta, 64),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(256 + 128 + 64, 192),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(192, 8),
        )

    def forward(self, xr, xf, m):
        zr = self.raw(xr)
        zr = torch.cat([zr.mean(-1), zr.amax(-1)], dim=1)
        zf = self.fold(xf)
        zf = torch.cat([zf.mean(-1), zf.amax(-1)], dim=1)
        zm = self.meta(m)
        return self.head(torch.cat([zr, zf, zm], dim=1))


def augment_raw(x):
    """Per-sample time roll, amplitude jitter, noise, and channel dropout."""
    b = x.shape[0]
    shifts = torch.randint(-60, 61, (b,), device=x.device)
    out = torch.stack([torch.roll(x[i], int(shifts[i]), dims=-1) for i in range(b)])
    out = out * (1.0 + 0.10 * torch.randn(b, 1, 1, device=x.device))
    out = out + 0.03 * torch.randn_like(out)
    mask = (torch.rand(b, 4, 1, device=x.device) > 0.10).float()
    return out * mask


def augment_fold(x):
    """Small phase roll, noise, and channel dropout for folded profiles."""
    b = x.shape[0]
    shifts = torch.randint(-2, 3, (b,), device=x.device)
    out = torch.stack([torch.roll(x[i], int(shifts[i]), dims=-1) for i in range(b)])
    out = out + 0.03 * torch.randn_like(out)
    mask = (torch.rand(b, x.shape[1], 1, device=x.device) > 0.10).float()
    return out * mask


def predict(model, xr, xf, m, tta=True):
    """Predict with light test-time augmentation."""
    model.eval()
    outs = []
    shifts = (-4, 0, 4) if tta else (0,)
    with torch.no_grad():
        for s in shifts:
            xr_s = torch.roll(xr, s, dims=-1) if s else xr
            xf_s = torch.roll(xf, max(-2, min(2, s // 4)), dims=-1) if s else xf
            for i in range(0, len(xr_s), 256):
                logits = model(xr_s[i:i + 256], xf_s[i:i + 256], m[i:i + 256])
                outs.append(torch.softmax(logits, dim=1))
    return torch.cat(outs, dim=0).view(len(shifts), len(xr), 8).mean(0).cpu().numpy()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
splits = list(skf.split(np.zeros(len(yi)), yi))

counts = np.bincount(yi, minlength=8).astype(np.float32)
class_w = len(yi) / (8.0 * np.maximum(counts, 1.0))
class_w = class_w / class_w.mean()
loss_fn = nn.CrossEntropyLoss(
    weight=torch.tensor(class_w, dtype=torch.float32, device=DEVICE),
    label_smoothing=0.03,
)

oof = np.zeros((len(yi), 8), dtype=np.float32)
testp = np.zeros((len(raw_test), 8), dtype=np.float32)

for fold, (tr, va) in enumerate(splits, 1):
    print(f"\n===== CNN fold {fold}/5 =====", flush=True)
    tr_t = torch.tensor(tr, dtype=torch.long, device=DEVICE)
    va_t = torch.tensor(va, dtype=torch.long, device=DEVICE)
    net = Net(M.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=3e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    best_f1 = -1.0
    best_state = None
    bad = 0
    t0 = time.time()

    for ep in range(EPOCHS):
        net.train()
        perm = torch.randperm(len(tr_t), device=DEVICE)
        for bi in range(0, len(tr_t), BATCH):
            b = tr_t[perm[bi:bi + BATCH]]
            xr = augment_raw(Xraw[b])
            xf = augment_fold(Xfold[b])
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(net(xr, xf, M[b]), y_t[b])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
        sched.step()

        pv = predict(net, Xraw[va_t], Xfold[va_t], M[va_t], tta=False)
        score = f1_score(yi[va], pv.argmax(1), average="macro")
        print(
            f"fold {fold} epoch {ep:02d} sec {time.time() - t0:6.1f} "
            f"loss {float(loss):.4f} valF1 {score:.4f}",
            flush=True,
        )
        if score > best_f1 + 1e-5:
            best_f1 = score
            best_state = copy.deepcopy(net.state_dict())
            bad = 0
        else:
            bad += 1
        if bad >= PATIENCE:
            print("early stop", flush=True)
            break

    if best_state is not None:
        net.load_state_dict(best_state)
    oof[va] = predict(net, Xraw[va_t], Xfold[va_t], M[va_t], tta=True)
    testp += predict(net, Xraw_test, Xfold_test, M_test, tta=True) / len(splits)
    print("fold", fold, "best valF1", round(best_f1, 4), "sec", round(time.time() - t0, 1), flush=True)

print("CNN OOF macro-F1", round(f1_score(yi, oof.argmax(1), average="macro"), 4), flush=True)
np.save(C / "oof_cnn.npy", oof)
np.save(C / "test_cnn.npy", testp)
print("cnn training complete", flush=True)
