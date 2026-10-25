"""
v3 one-vs-rest specialist models.

The multiclass LightGBM already predicts all eight labels, but the hardest
confusions are asymmetric.  For example, planet vs quiet_star is much harder
than quiet_star vs eclipsing_binary.  A dedicated binary model for each class
gives the stacker a second, differently-biased opinion.

Outputs (one pair per class):
    v3/cache/oof_bin_<class>.npy
    v3/cache/test_bin_<class>.npy
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
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
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def run_specialist(target: str):
    yb = (y == target).astype(np.int32)
    oof = np.zeros(len(Xtr), dtype=np.float32)
    testp = np.zeros(len(Xte), dtype=np.float32)
    t0 = time.time()

    for fold, (tr, va) in enumerate(skf.split(Xtr, yb), 1):
        model = lgb.LGBMClassifier(
            objective="binary",
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.90,
            colsample_bytree=0.80,
            reg_lambda=3.0,
            n_estimators=600,
            n_jobs=6,
            verbosity=-1,
            class_weight="balanced",
            random_state=42 + fold,
            force_col_wise=True,
        )
        model.fit(
            Xtr.iloc[tr],
            yb[tr],
            eval_set=[(Xtr.iloc[va], yb[va])],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(70, verbose=False), lgb.log_evaluation(0)],
        )
        oof[va] = model.predict_proba(Xtr.iloc[va])[:, 1]
        testp += model.predict_proba(Xte)[:, 1] / 5
        print(target, "fold", fold, "iter", model.best_iteration_, "sec", round(time.time() - t0, 1), flush=True)

    auc = roc_auc_score(yb, oof)
    print(target, "OOF AUC", round(float(auc), 4), flush=True)
    np.save(C / f"oof_bin_{target}.npy", oof)
    np.save(C / f"test_bin_{target}.npy", testp)


if __name__ == "__main__":
    for target in classes:
        run_specialist(str(target))
    print("specialist training complete", flush=True)
