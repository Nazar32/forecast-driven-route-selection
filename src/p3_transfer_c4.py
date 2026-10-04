"""
Paper 3 (v4): US severe-weather transfer with the rolling baselines and the blend of the decision layer.

The anchored model is NOT retrained for the US (the US ML is the transfer model of v3, results/<P3_RESULTS>/transfer);
the US forecast is recalibrated per region daily and averaged 50/50 with a rolling Markov chain (42 days), exactly
as the decision layer of C4. Reported next to the v3 ML, full-history Markov, rolling Markov, rolling static,
reactive and oracle. Output: results/<P3_RESULTS>/transfer_c4.json, transfer_c4_per_od.csv
"""
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import p3_core as C  # noqa: E402
import p3_transfer as T  # noqa: E402
import p3_raion_exp as X  # noqa: E402

H1 = pd.Timedelta(hours=1)


def main():
    os.environ["P3_GRAPH_DIR"] = str(T.GDIR)
    import networkx as nx
    import p3_graph_candidates as GC
    from p3_eval import select
    pres = T.presence()
    reg = list(pres.columns)
    sp = json.loads((T.OUT / "split.json").read_text())
    t_test = pd.Timestamp(sp["t_test"])
    hours = pd.date_range(t_test, T.END - pd.Timedelta(hours=T.K_MAX), freq="h", tz="UTC")
    pre = pd.date_range(T.START, t_test - pd.Timedelta(hours=T.K_MAX + 1), freq="h", tz="UTC")
    dp = C.read_probs(T.OUT / "direct_probs.parquet")
    rows = pd.date_range(dp["hour"].min(), dp["hour"].max(), freq="h", tz="UTC")
    Q = np.full((len(rows), X.NK, len(reg)), np.nan, np.float32)
    rp = pd.Series(np.arange(len(rows)), index=rows)
    ri = {r: j for j, r in enumerate(reg)}
    Q[rp[dp["hour"]].values, dp["k"].values.astype(int) - 1, dp["oblast"].map(ri).values] = dp["p"].values
    act0 = T.active_at(hours, reg).astype(np.float64)
    Qr = X.recalibrate(Q, rows, pres, t_test)
    F = {"pred": X.causal(Q, rows, hours, act0), "pred_recal": X.causal(Qr, rows, hours, act0)}
    F["markov_roll"] = X.markov_rolling(pres, hours, act0)
    pre_mask = (pres.index >= T.START) & (pres.index < t_test)
    import p3_raion_model as RM
    F["markov"], _ = RM.markov(pres, hours, pre_mask, act0)
    F["blend50"] = 0.5 * F["pred_recal"] + 0.5 * F["markov_roll"]
    F["blend50"][:, 0] = np.maximum(F["blend50"][:, 0], act0)
    g, cities = GC.load_graph()
    csr = GC.Csr(g)
    pairs = []
    for a, b in itertools.combinations([c for c in T.CENTERS if c in cities], 2):
        tt = GC.path_time(g, nx.shortest_path(g, cities[a], cities[b], weight="time_s")) / 3600
        if T.T_MIN_H <= tt <= T.T_MAX_H:
            pairs.append((a, b))
    rr = np.arange(len(hours))
    per, pooled, dts = [], {}, {}
    for a, b in pairs:
        cand = GC.penalty_lo(g, cities[a], cities[b], 10, csr=csr)
        c = C.Corridor("od", pres, routes=[GC.path_route(g, p, f"{a}-{b}#{i}") for i, p in enumerate(cand)])
        oi = [reg.index(o) for o in c.obl]
        K = c.metric(hours)
        i_s = int(c.metric(pre).mean(0).argmin())
        ch = {"static": np.full(len(hours), i_s)}
        for n, P in F.items():
            ch[n] = select(c.expected_metric(P[:, :, oi]), i_s)
        ch["reactive"] = select(c.current_metric(act0[:, oi]), i_s)
        ch["oracle"] = K.argmin(1)
        hh = pd.date_range(hours[0] - pd.Timedelta(days=29), hours[-1], freq="h", tz="UTC")
        Kx = c.metric(hh)
        cs = np.vstack([np.zeros((1, Kx.shape[1])), np.cumsum(Kx, 0)])
        e_ = np.arange(len(hours)) + 29 * 24 - T.K_MAX
        ch["static_roll"] = select(cs[e_] - cs[e_ - 28 * 24], i_s)
        d = {k: K[rr, v] for k, v in ch.items()}
        for k, v in d.items():
            pooled.setdefault(k, []).append(v)
            dts.setdefault(k, []).append(float(60 * (c.T[ch[k]].mean() - c.T[i_s])))
        per.append({"od": f"{a}-{b}", **{k: C.impr(d["static"], v) for k, v in d.items() if k != "static"}})
    per = pd.DataFrame(per)
    per.to_csv(C.P3 / "transfer_c4_per_od.csv", index=False)
    S = {k: np.sum(v, axis=0) for k, v in pooled.items()}
    out = {"n_od": len(per), "n_hours": len(hours), "test_start": str(t_test),
           "pooled": {k: C.impr(S["static"], v) for k, v in S.items() if k != "static"},
           "ci_vs_static": {k: C.block_bootstrap_impr(S["static"], v, block=72) for k, v in S.items() if k != "static"},
           "better": {k: float((per[k] > 0).mean()) for k in per.columns if k != "od"},
           "dT_min": {k: float(np.mean(v)) for k, v in dts.items()},
           "paired": {f"{a}-markov_roll": X.paired_ci(S["static"], S[a], S["markov_roll"])
                      for a in ("pred", "pred_recal", "blend50", "static_roll")}}
    (C.P3 / "transfer_c4.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
