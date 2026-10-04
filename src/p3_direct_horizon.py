"""
Paper 3 / experiment (a),(c): CAUSAL direct multi-horizon predictor.

For every lead k = 1..K_MAX we fit a separate classifier
    features at hour t  ->  P(alert in oblast at hour t+k)
on the Paper-2 feature matrix and the Paper-2 temporal split (70/15/15).
Only information available at decision time t is used, so these probabilities
are deployable as-is (no look-up of future feature rows, unlike the
retrospective multi-horizon lookup in ml/predictor.py).

Model: sklearn HistGradientBoosting (the routing technology is model-agnostic;
Paper 2 showed tree ensembles match the DNN+RF under the same calibration),
followed by isotonic calibration fitted on the validation block.
v2: leads k = 1..19, embargo at the split boundaries, rel_rate_24h normalised on train data.

Outputs (results/p3/):
  direct_probs.parquet   long form: hour, oblast, k, p   (validation + test hours)
  direct_metrics.json    per-k AUC / Brier on the test block
Run from project root:
    python src/p3_direct_horizon.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

sys.path.insert(0, str(Path(__file__).parent))
import p3_features as F  # noqa: E402

import p3_core as C  # noqa: E402

OUT = C.P3
K_MAX = C.K_MAX + 1          # v2: leads 0..18 after departure need models k = 1..19 on row t-1
SEED = 42


def hgb():
    return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08, max_leaf_nodes=63,
                                          l2_regularization=1.0, early_stopping=False, random_state=SEED)


def fit_direct(df, X, itr, iva, iout, t_val, t_test, make=hgb, ks=range(1, K_MAX + 1), tag=""):
    """Direct model per lead k with an embargo: training rows need target hour t+k < t_val,
    calibration rows need t+k < t_test.  Returns long-form calibrated probabilities on `iout`
    and per-k test AUC/Brier (rows of `iout` with hour >= t_test)."""
    g = df.groupby("oblast")["alert_occurred"]
    hour = df["hour"]
    rows, metrics = [], {}
    for k in ks:
        t0 = time.time()
        y = g.shift(-k).values
        tgt = hour + pd.Timedelta(hours=k)
        tr = itr[(~np.isnan(y[itr])) & (tgt.iloc[itr] < t_val).values]
        va = iva[(~np.isnan(y[iva])) & (tgt.iloc[iva] < t_test).values]
        clf = make().fit(X[tr], y[tr].astype(int))
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
        iso.fit(clf.predict_proba(X[va])[:, 1], y[va])
        p = iso.predict(clf.predict_proba(X[iout])[:, 1])
        m = (hour.iloc[iout] >= t_test).values & ~np.isnan(y[iout])
        yt = y[iout][m].astype(int)
        metrics[k] = {"auc": float(roc_auc_score(yt, p[m])), "brier": float(brier_score_loss(yt, p[m])),
                      "base_rate": float(yt.mean()), "n_train": int(len(tr)), "n_cal": int(len(va)),
                      "sec": round(time.time() - t0, 1)}
        print(tag, k, metrics[k], flush=True)
        rows.append(pd.DataFrame({"hour": hour.values[iout], "oblast": df["oblast"].values[iout],
                                  "k": np.int8(k), "p": p.astype(np.float32)}))
    return pd.concat(rows, ignore_index=True), metrics


def main():
    # P3_LEAKY=1: look-ahead variant for the ablation (full-sample rel_rate normalisation; used with row t)
    leaky = os.environ.get("P3_LEAKY") == "1"
    df = F.build_frame(cutoff=None if leaky else "train").sort_values(["oblast", "hour"]).reset_index(drop=True)
    cols = F.get_feature_columns(df)
    tr, va, te = F.temporal_split(df)
    t_val, t_test = va["hour"].min(), te["hour"].min()
    X = df[cols].values.astype(np.float32)
    iout = np.r_[va.index.values, te.index.values]
    out, metrics = fit_direct(df, X, tr.index.values, va.index.values, iout, t_val, t_test)
    OUT.mkdir(parents=True, exist_ok=True)
    tag = "_leaky" if leaky else ""
    out.to_parquet(OUT / f"direct_probs{tag}.parquet")
    (OUT / f"direct_metrics{tag}.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
