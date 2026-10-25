"""
v3 tabular models.

Trains three diverse tabular models with the same 5-fold split:
    1. LightGBM (main)
    2. LightGBM (different depth/feature sampling)
    3. ExtraTrees

The out-of-fold (OOF) probabilities and test probabilities are saved to
v3/cache for the final stacking script.
"""

from __future__ import annotations

import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

ROOT = Path(r"C:\Users\Kalpana\Documents\GitHub\Planet_hunters")
C = ROOT / "v3" / "cache"
V2 = ROOT / "v2" / "cache"

Xtr = pd.read_pickle(C / "X_train_tab.pkl").reset_index(drop=True)
Xte = pd.read_pickle(C / "X_test_tab.pkl").reset_index(drop=True)
Xte = Xte.reindex(columns=Xtr.columns, fill_value=0.0)

y = np.load(V2 / "y.npy", allow_pickle=True)
classes = np.load(V2 / "classes.npy", allow_pickle=True)
yi = np.searchsorted(classes, y)
n_class = len(classes)

print("tabular shapes", Xtr.shape, Xte.shape, "classes", classes.tolist(), flush=True)

FAST = os.environ.get("V3_FAST", "0") == "1"
skf = StratifiedKFold(n_splits=3 if FAST else 5, shuffle=True, random_state=42)
splits = list(skf.split(Xtr, yi))


def macro_f1(y_true, prob):
    return f1_score(classes[y_true], classes[np.argmax(prob, axis=1)], average="macro")


def run_lightgbm(name, params):
    """Train one LightGBM model with early stopping and save OOF/test probs."""
    oof = np.zeros((len(Xtr), n_class), dtype=np.float32)
    testp = np.zeros((len(Xte), n_class), dtype=np.float32)
    t0 = time.time()

    for fold, (tr, va) in enumerate(splits, 1):
        model = lgb.LGBMClassifier(**params)
        model.fit(
            Xtr.iloc[tr],
            yi[tr],
            eval_set=[(Xtr.iloc[va], yi[va])],
            eval_metric="multi_logloss",
            callbacks=[
                lgb.early_stopping(100, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        oof[va] = model.predict_proba(Xtr.iloc[va])
        testp += model.predict_proba(Xte) / len(splits)
        print(name, "fold", fold, "iter", model.best_iteration_, "sec", round(time.time() - t0, 1), flush=True)

    score = macro_f1(yi, oof)
    print(name, "OOF macro-F1", round(score, 4), flush=True)
    np.save(C / f"oof_{name}.npy", oof)
    np.save(C / f"test_{name}.npy", testp)
    return score


def run_extra_trees(name):
    """ExtraTrees is a high-variance, low-bias complement to LightGBM.

    ExtraTrees cannot handle NaN/inf, unlike LightGBM.  Impute with the
    training median so the two models see the same underlying information.
    """
    Xtr_et = Xtr.replace([np.inf, -np.inf], np.nan)
    med = Xtr_et.median(numeric_only=True)
    Xtr_et = Xtr_et.fillna(med).fillna(0.0)
    Xte_et = Xte.replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0)

    oof = np.zeros((len(Xtr), n_class), dtype=np.float32)
    testp = np.zeros((len(Xte), n_class), dtype=np.float32)
    t0 = time.time()
    for fold, (tr, va) in enumerate(splits, 1):
        model = ExtraTreesClassifier(
            n_estimators=700,
            max_features=0.30,
            min_samples_leaf=2,
            n_jobs=6,
            random_state=42,
            class_weight="balanced",
            bootstrap=False,
        )
        model.fit(Xtr_et.iloc[tr], yi[tr])
        oof[va] = model.predict_proba(Xtr_et.iloc[va])
        testp += model.predict_proba(Xte_et) / len(splits)
        print(name, "fold", fold, "sec", round(time.time() - t0, 1), flush=True)
    score = macro_f1(yi, oof)
    print(name, "OOF macro-F1", round(score, 4), flush=True)
    np.save(C / f"oof_{name}.npy", oof)
    np.save(C / f"test_{name}.npy", testp)
    return score


if __name__ == "__main__":
    fast = FAST
    base_params = dict(
        objective="multiclass",
        num_class=n_class,
        learning_rate=0.035,
        num_leaves=95,
        min_child_samples=18,
        subsample=0.82,
        colsample_bytree=0.68,
        reg_lambda=8.0,
        n_estimators=500 if fast else 2000,
        n_jobs=6,
        verbosity=-1,
        class_weight="balanced",
        random_state=42,
        force_col_wise=True,
    )
    run_lightgbm("lgb", base_params)

    if not fast:
        alt_params = dict(
            objective="multiclass",
            num_class=n_class,
            learning_rate=0.045,
            num_leaves=31,
            min_child_samples=30,
            subsample=0.90,
            colsample_bytree=0.88,
            reg_lambda=3.0,
            n_estimators=1600,
            n_jobs=6,
            verbosity=-1,
            class_weight="balanced",
            random_state=202,
            force_col_wise=True,
        )
        run_lightgbm("lgb2", alt_params)
        run_extra_trees("et")
    print("tabular training complete", flush=True)
