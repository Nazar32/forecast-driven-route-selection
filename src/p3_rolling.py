"""
Paper 3: ROLLING-ORIGIN evaluation (robustness over time).

Four consecutive 6-month test windows; for each window the direct per-lead predictor
(p3_direct_horizon.py, same features and model) is RE-TRAINED on all data before the window
(the last 3 months before it are the isotonic-calibration block), the static baseline is re-chosen
on pre-window data, and every strategy is evaluated on the window only.

  W1 test 2023-10-01 .. 2024-04-01     W2 test 2024-04-01 .. 2024-10-01
  W3 test 2024-10-01 .. 2025-04-01     W4 test 2025-04-01 .. 2025-10-13
Evaluated on: the 4 ORS corridors and the 78 OSM-primary OD pairs (K = 10 penalty candidates).

Phase 1 (train, resumable):  python src/p3_rolling.py --phase train
Phase 2 (evaluate):          python src/p3_rolling.py --phase eval
Outputs: results/p3/rolling/probs_W*.parquet, results/p3/rolling_summary.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).parent))
import p3_core as C  # noqa: E402
import p3_features as F  # noqa: E402

OUT = C.P3 / "rolling"
K_MAX = C.K_MAX
WINDOWS = [
    ("W0", "2023-01-01", "2023-04-01", "2023-10-01"),
    ("W1", "2023-07-01", "2023-10-01", "2024-04-01"),
    ("W2", "2024-01-01", "2024-04-01", "2024-10-01"),
    ("W3", "2024-07-01", "2024-10-01", "2025-04-01"),
    ("W4", "2025-01-01", "2025-04-01", "2025-10-14"),
    ("W3c", "2024-07-01", "2024-10-01", "2025-01-01"),   # v5: W3 models, test cut at 31.12.2024
]
# v4: four windows inside the oblast-declaration period (W4 overlaps the raion-level transition)
if os.environ.get("P3_WINDOWS"):
    WINDOWS = [w for w in WINDOWS if w[0] in os.environ["P3_WINDOWS"].split(",")]   # name, validation start, test start, test end (exclusive)


def ts(x):
    return pd.Timestamp(x, tz="UTC")


def train():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from p3_direct_horizon import fit_direct
    OUT.mkdir(parents=True, exist_ok=True)
    base = F.build_frame(cutoff=None)
    for name, vs, t0, t1 in WINDOWS:
        path = OUT / f"probs_{name}.parquet"
        if path.exists():
            print(name, "exists, skipping", flush=True)
            continue
        # v2: rel_rate_24h normalised by pre-window (training) means only
        df = F.fix_rel_rate(base, ts(vs)).sort_values(["oblast", "hour"]).reset_index(drop=True)
        cols = F.get_feature_columns(df)
        X = df[cols].values.astype(np.float32)
        hour = df["hour"]
        itr = np.where(hour < ts(vs))[0]
        iva = np.where((hour >= ts(vs)) & (hour < ts(t0)))[0]
        iout = np.where((hour >= ts(t0) - pd.Timedelta(hours=1)) & (hour < ts(t1)))[0]
        make = lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=63,
                                                      l2_regularization=1.0, early_stopping=False,
                                                      random_state=42)
        out, met = fit_direct(df, X, itr, iva, iout, ts(vs), ts(t0), make=make, tag=name)
        out.to_parquet(path)
        (OUT / f"auc_{name}.json").write_text(json.dumps({k: v["auc"] for k, v in met.items()}))


def probs_tensor(name, hours, oblasts, act=None):
    return C.causal_P(C.read_probs(OUT / f"probs_{name}.parquet"), hours, oblasts,
                      C.active_at(hours, oblasts))


def evaluate():
    import p3_geo as G
    from p3_eval import select
    os.environ.setdefault("P3_GRAPH_DIR", str(C.P3_IN / "roadgraph_osm_primary"))
    import p3_graph_candidates as GC
    from p3_graph_eval import od_pairs
    act_all = C.presence()
    oblasts = list(act_all.columns)
    g, cities = GC.load_graph()
    csr = GC.Csr(g)
    pairs = od_pairs(g, cities)
    cand_routes = {}
    for a, b in pairs:
        cand = GC.penalty_lo(g, cities[a], cities[b], 10, csr=csr)
        cand_routes[(a, b)] = [GC.path_route(g, p, f"{a}-{b}#{i}") for i, p in enumerate(cand)]
    summary = {}
    for name, vs, t0, t1 in WINDOWS:
        hours = pd.date_range(ts(t0), ts(t1) - pd.Timedelta(hours=K_MAX + 1), freq="h", tz="UTC")
        pre = pd.date_range(C.DATA_START, ts(t0) - pd.Timedelta(hours=K_MAX + 1), freq="h", tz="UTC")
        Pall = probs_tensor(name, hours, oblasts, act_all)
        auc = json.loads((OUT / f"auc_{name}.json").read_text())
        res = {"n_hours": len(hours), "auc_k1": auc["1"], "auc_k6": auc["6"], "auc_k7": auc["7"],
               "corridors": {}, "osm": {}}
        CUR = C.active_at(hours, oblasts)
        rows = np.arange(len(hours))
        for key in G.CORRIDORS:
            c = C.Corridor(key, act_all)
            oi = [oblasts.index(o) for o in c.obl]
            P = Pall[:, :, oi]
            DUA, DUAp = c.metric(hours), c.metric(pre)          # v2: km under alert
            i_stat = int(DUAp.mean(0).argmin())
            base = DUA[:, i_stat]
            ch = {"reactive": select(c.current_metric(CUR[:, oi]), i_stat),
                  "pred_sched": select(c.expected_metric(P), i_stat),
                  "oracle": DUA.argmin(1)}
            e = {"static": c.ids[i_stat], "static_KUA": float(base.mean())}
            for n, x in ch.items():
                d = DUA[rows, x]
                e[f"{n}_impr_pct"] = C.impr(base, d)
                if n == "pred_sched":
                    e["pred_ci"] = C.block_bootstrap_impr(base, d)
                    e["pred_p"] = float(wilcoxon(base, d).pvalue) if np.any(base != d) else 1.0
                    e["pred_dT_min"] = float(60 * (c.T[x].mean() - c.T[i_stat]))
            res["corridors"][key] = e
        # OSM primary, K = 10 penalty candidates
        per = []
        for (a, b), routes in cand_routes.items():
            c = C.Corridor(f"{a}-{b}", act_all, routes=routes)
            oi = [oblasts.index(o) for o in c.obl]
            P = Pall[:, :, oi]
            DUA, DUAp = c.metric(hours), c.metric(pre)
            i_stat = int(DUAp.mean(0).argmin())
            base = DUA[:, i_stat]
            dp = DUA[rows, select(c.expected_metric(P), i_stat)]
            dr = DUA[rows, select(c.current_metric(CUR[:, oi]), i_stat)]
            do = DUA.min(1)
            per.append((base.mean(), dp.mean(), dr.mean(), do.mean(),
                        float(wilcoxon(base, dp).pvalue) if np.any(base != dp) else 1.0))
        B, Pp, R, O, pv = map(np.array, zip(*per))
        pool = lambda x: float(100 * (B.sum() - x.sum()) / B.sum())
        res["osm"] = {"n_od": len(per), "pred_pooled_pct": pool(Pp), "reactive_pooled_pct": pool(R),
                      "oracle_pooled_pct": pool(O), "share_od_better": float(np.mean(Pp < B)),
                      "share_od_sig": float(np.mean((pv < 0.05) & (Pp < B)))}
        summary[name] = res
        print(name, json.dumps({k: (round(v["pred_sched_impr_pct"], 2) if isinstance(v, dict) and "pred_sched_impr_pct" in v else v)
                                for k, v in res["corridors"].items()}), json.dumps(res["osm"]), flush=True)
    (C.P3 / "rolling_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["train", "eval"], required=True)
    a = ap.parse_args()
    train() if a.phase == "train" else evaluate()
