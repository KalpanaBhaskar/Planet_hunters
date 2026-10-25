"""
v3 final blend / stack.

Inputs (missing optional files are skipped):
    v3/cache/oof_lgb.npy,     test_lgb.npy
    v3/cache/oof_lgb2.npy,    test_lgb2.npy
    v3/cache/oof_et.npy,      test_et.npy
    v3/cache/oof_cnn.npy,     test_cnn.npy
    v2/cache/oof_v2.npy,      test_proba_v2.npy
    v3/cache/oof_bin_<class>.npy, test_bin_<class>.npy

Steps:
    1. Optimize a convex weight blend on OOF macro-F1.
    2. Train a small LightGBM stacker on all model probabilities plus the
       one-vs-rest specialist probabilities.
    3. Choose the best of blend, stacker, or a convex combination of both.
    4. Tune per-class multipliers on OOF macro-F1.
    5. Write v3/submission.csv with exactly 7,000 rows.

This is deliberately conservative: if the stacker overfits, the weighted
blend is still available and includes v2 as one of its components.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

ROOT = Path(r"C:\Users\Kalpana\Documents\GitHub\Planet_hunters")
C = ROOT / "v3" / "cache"
V2 = ROOT / "v2" / "cache"
D = ROOT / "dataset_and_problem_statement"

y = np.load(V2 / "y.npy", allow_pickle=True)
classes = np.load(V2 / "classes.npy", allow_pickle=True)
yi = np.searchsorted(classes, y)
n_class = len(classes)


def normalize(p):
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, 1e-9, 1.0)
    return p / p.sum(axis=1, keepdims=True)


def macro_f1(y_true, p):
    return float(f1_score(classes[y_true], classes[p.argmax(1)], average="macro"))


# ---------------------------------------------------------------------------
# Load model probabilities
# ---------------------------------------------------------------------------
specs = [
    ("lgb", C / "oof_lgb.npy", C / "test_lgb.npy"),
    ("lgb2", C / "oof_lgb2.npy", C / "test_lgb2.npy"),
    ("et", C / "oof_et.npy", C / "test_et.npy"),
    ("cnn", C / "oof_cnn.npy", C / "test_cnn.npy"),
    ("v2", V2 / "oof_v2.npy", V2 / "test_proba_v2.npy"),
]

names = []
O = []
T = []
for name, op, tp in specs:
    if op.exists() and tp.exists():
        o = normalize(np.load(op))
        t = normalize(np.load(tp))
        if len(o) != len(yi):
            print("skip", name, "bad OOF shape", o.shape, flush=True)
            continue
        names.append(name)
        O.append(o)
        T.append(t)
        print(name, "OOF", round(macro_f1(yi, o), 4), flush=True)
    else:
        print("missing optional model", name, flush=True)

if not O:
    raise SystemExit("No model probability files found. Run train_tabular.py (and optionally train_cnn.py).")

O = np.stack(O, axis=0)  # (M, N, 8)
T = np.stack(T, axis=0)  # (M, Ntest, 8)
n_models = len(names)
n_test = T.shape[1]


def blend_with_weights(w):
    w = np.maximum(np.asarray(w, dtype=np.float64), 0.0)
    if w.sum() <= 0:
        w = np.ones(n_models)
    w = w / w.sum()
    return np.tensordot(w, O, axes=(0, 0)), np.tensordot(w, T, axes=(0, 0))


# ---------------------------------------------------------------------------
# 1. Random-simplex weight search (more robust than optimizing one model at a time)
# ---------------------------------------------------------------------------
rng = np.random.default_rng(42)
best_w = np.ones(n_models) / n_models
best_score = macro_f1(yi, blend_with_weights(best_w)[0])

# Single-model baselines
for i in range(n_models):
    w = np.zeros(n_models)
    w[i] = 1.0
    s = macro_f1(yi, blend_with_weights(w)[0])
    if s > best_score:
        best_score, best_w = s, w

# Dirichlet random search
for _ in range(2500):
    w = rng.dirichlet(np.ones(n_models) * 0.8)
    s = macro_f1(yi, blend_with_weights(w)[0])
    if s > best_score:
        best_score, best_w = s, w

print("weight blend OOF", round(best_score, 4), "weights", dict(zip(names, np.round(best_w, 3))), flush=True)
oof_blend, test_blend = blend_with_weights(best_w)

# ---------------------------------------------------------------------------
# 2. Stacker on model + specialist probabilities
# ---------------------------------------------------------------------------
Ztr = np.hstack(O)  # (N, M*8)
Zte = np.hstack(T)

specialist_names = []
for c in classes:
    op = C / f"oof_bin_{c}.npy"
    tp = C / f"test_bin_{c}.npy"
    if op.exists() and tp.exists():
        o = np.asarray(np.load(op), dtype=np.float64).reshape(-1, 1)
        t = np.asarray(np.load(tp), dtype=np.float64).reshape(-1, 1)
        if len(o) == len(yi) and len(t) == n_test:
            specialist_names.append(str(c))
            Ztr = np.hstack([Ztr, o])
            Zte = np.hstack([Zte, t])

print("stacker features", Ztr.shape, "specialists", specialist_names, flush=True)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
meta_oof = np.zeros((len(yi), n_class), dtype=np.float32)
meta_test = np.zeros((n_test, n_class), dtype=np.float32)

for fold, (tr, va) in enumerate(skf.split(Ztr, yi), 1):
    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=n_class,
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=30,
        subsample=0.90,
        colsample_bytree=0.80,
        reg_lambda=5.0,
        n_estimators=500,
        n_jobs=6,
        verbosity=-1,
        class_weight="balanced",
        random_state=42 + fold,
        force_col_wise=True,
    )
    model.fit(
        Ztr[tr],
        yi[tr],
        eval_set=[(Ztr[va], yi[va])],
        eval_metric="multi_logloss",
        callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)],
    )
    meta_oof[va] = model.predict_proba(Ztr[va])
    meta_test += model.predict_proba(Zte) / 5
    print("stacker fold", fold, "iter", model.best_iteration_, flush=True)

meta_score = macro_f1(yi, meta_oof)
print("stacker OOF", round(meta_score, 4), flush=True)

# ---------------------------------------------------------------------------
# 3. Choose blend / stacker / mixture
# ---------------------------------------------------------------------------
candidates = [
    ("weight_blend", oof_blend, test_blend, macro_f1(yi, oof_blend)),
    ("stacker", meta_oof, meta_test, meta_score),
]
best_name, best_oof, best_test, best_final = max(candidates, key=lambda z: z[3])

# Try a convex mixture of the two, even if one is weaker.
alpha_best = 0.0
for alpha in np.linspace(0.0, 1.0, 21):
    p = alpha * meta_oof + (1.0 - alpha) * oof_blend
    s = macro_f1(yi, p)
    if s > best_final:
        best_final = s
        best_name = "mixture"
        best_oof = p
        best_test = alpha * meta_test + (1.0 - alpha) * test_blend
        alpha_best = float(alpha)

print("chosen", best_name, "alpha", alpha_best, "OOF", round(best_final, 4), flush=True)

# ---------------------------------------------------------------------------
# 4. Class multipliers (coordinate ascent)
# ---------------------------------------------------------------------------
mult = np.ones(n_class, dtype=np.float64)


def apply_mult(p, m):
    q = p * m[None, :]
    return q / np.maximum(q.sum(axis=1, keepdims=True), 1e-12)


for _ in range(4):
    for c in range(n_class):
        best_v = mult[c]
        best_s = macro_f1(yi, apply_mult(best_oof, mult))
        for v in np.linspace(0.35, 2.4, 23):
            trial = mult.copy()
            trial[c] = v
            s = macro_f1(yi, apply_mult(best_oof, trial))
            if s > best_s + 1e-6:
                best_s, best_v = s, v
        mult[c] = best_v

final_oof = apply_mult(best_oof, mult)
final_test = apply_mult(best_test, mult)
final_score = macro_f1(yi, final_oof)
print("final OOF", round(final_score, 4), "multipliers", dict(zip(classes, np.round(mult, 3))), flush=True)

# Per-class report
pred_oof = classes[final_oof.argmax(1)]
report = {}
for c in classes:
    m = y == c
    report[str(c)] = round(float(f1_score(m, pred_oof == c, average="binary")), 4)
print("per-class F1", report, flush=True)

# ---------------------------------------------------------------------------
# 5. Submission
# ---------------------------------------------------------------------------
sample = pd.read_csv(D / "sample_submission.csv")
ids = sample["id"].astype(str).values
if len(ids) != len(final_test):
    raise SystemExit(f"submission length mismatch: {len(ids)} vs {len(final_test)}")

pred = classes[final_test.argmax(1)]
sub = pd.DataFrame({"id": ids, "label": pred})
sub.to_csv(ROOT / "v3" / "submission.csv", index=False)

assert len(sub) == 7000
assert list(sub.columns) == ["id", "label"]
assert (sub["id"].values == ids).all()

metrics = {
    "model_names": names,
    "model_oof": {n: round(macro_f1(yi, O[i]), 4) for i, n in enumerate(names)},
    "weight_blend_oof": round(best_score, 4),
    "weights": {n: round(float(w), 5) for n, w in zip(names, best_w)},
    "stacker_oof": round(meta_score, 4),
    "specialists": specialist_names,
    "chosen": best_name,
    "mixture_alpha": alpha_best,
    "final_oof_macro_f1": round(final_score, 4),
    "class_multipliers": {str(c): round(float(m), 4) for c, m in zip(classes, mult)},
    "per_class_f1": report,
    "test_prediction_counts": {str(k): int(v) for k, v in pd.Series(pred).value_counts().items()},
}
with open(ROOT / "v3" / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

np.save(C / "oof_final.npy", final_oof)
np.save(C / "test_final.npy", final_test)
print("wrote", ROOT / "v3" / "submission.csv", "rows", len(sub), flush=True)
