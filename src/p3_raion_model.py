"""
Paper 3, raion-level technology on the extended alert log (15.03.2022 -- 06.09.2026).

Why: from May 2025 (Dnipropetrovsk) and August 2025 (Kyiv, Poltava, Kharkiv, Sumy, Chernihiv) and in all
oblasts from November 2025 alerts are declared per raion, so the oblast-level presence of the paper no longer
describes where alerts are. Here the region units are raions (post-2020, OSM boundaries; city of Kyiv as one
unit), and the technology is applied exactly as in the paper and in the US transfer:
  presence  a(u, unit) = oblast-level alert of its oblast OR raion-level alert of the raion (hromada not used)
  forecast  19 direct per-lead HGB models + isotonic calibration, same hyperparameters, chronological
            70/15/15 split, embargo; generic features (own history, neighbouring raions, raions of the same
            oblast, national activity, calendar) with the unit as a categorical feature
  decision  causal (row t-1, model k=j+1, lead 0 = max(active at t:00, g_1)), schedule-aware score on
            raion-level time profiles; corridors (ORS routes) and OSM primary (K = 10 penalty candidates,
            same 78 OD pairs); static baseline = fewest pre-test km under alert
Nothing is tuned on the test block.

  P3_RESULTS=p3v3 python p3_raion_model.py --phase data|train|eval
Outputs: results/p3v3/raion/{presence.parquet, adjacency.json, direct_probs.parquet, split.json, summary.json, ...}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import p3_core as C  # noqa: E402
import p3_geo as G  # noqa: E402
import p3_raion as RA  # noqa: E402

LOG2 = C.DATA / "alerts" / "official_data_uk_2026-09-07_dedup.csv.gz"
OUT = C.P3 / os.environ.get("RAION_OUT", "raion")
# optional fixed split "t_val,t_test" (rolling-origin style check); default: chronological 70/15/15
SPLIT = os.environ.get("RAION_SPLIT")
SEED = 42
K_MAX = C.K_MAX
START = C.DATA_START
END = pd.Timestamp("2026-09-06 23:00", tz="UTC")
NO_DATA = {"Луганська область", "Автономна Республіка Крим", "Севастополь"}
LEVELS = ("oblast", "raion")


def units():
    obl = sorted({f for f in RA.raion_polygons()} | {"м. Київ"})
    obl = [o for o in obl if o not in NO_DATA]
    return RA.all_units(obl)


# ----------------------------------------------------------------------------- data
def adjacency(us):
    import json as js
    from shapely.geometry import shape
    g = js.loads(RA.RAIONS.read_text())
    geom = {}
    for f in g["features"]:
        p = f["properties"]
        obl = RA.OSM2LOG_OBL.get(p["oblast"], p["oblast"])
        if p.get("raion"):
            geom[RA.unit(obl, RA.norm(p["raion"]))] = shape(f["geometry"]).buffer(0)
        elif obl == "м. Київ":
            geom[RA.unit(obl)] = shape(f["geometry"]).buffer(0)
    gs = {u: geom[u].buffer(0.01) for u in us}
    return {u: sorted(v for v in us if v != u and gs[u].intersects(geom[v])) for u in us}


def phase_data():
    OUT.mkdir(parents=True, exist_ok=True)
    us = units()
    log = RA.read_log(LOG2)
    pres = RA.presence_units(us, LEVELS, log, START, END + pd.Timedelta(hours=K_MAX + 2))
    pres.to_parquet(OUT / "presence.parquet")
    adj = adjacency(us)
    (OUT / "adjacency.json").write_text(json.dumps(adj, ensure_ascii=False))
    inb = pres.loc[START:END]
    m = inb.mean(axis=1).groupby(inb.index.tz_convert(None).to_period("Q").astype(str)).mean()
    print("units", len(us), "isolated", [u for u, v in adj.items() if not v])
    print("presence by quarter", m.round(3).to_dict())


def presence():
    return pd.read_parquet(OUT / "presence.parquet")


# ----------------------------------------------------------------------------- features + models
def split_times(idx):
    if SPLIT:
        a, b = SPLIT.split(",")
        return pd.Timestamp(a, tz="UTC"), pd.Timestamp(b, tz="UTC")
    n = len(idx)
    return idx[int(0.70 * n)], idx[int(0.85 * n)]


def features(pres, adj):
    A = pres.loc[START:END].astype(np.float32)
    us = list(A.columns)
    roll = lambda X, w: X.rolling(w, min_periods=1).mean()
    f = {"a0": A, "a1": A.shift(1), "a2": A.shift(2), "a5": A.shift(5),
         "r3": roll(A, 3), "r6": roll(A, 6), "r24": roll(A, 24), "r72": roll(A, 72), "r168": roll(A, 168)}
    idx = np.arange(len(A))[:, None] * np.ones((1, A.shape[1]))
    lt = pd.DataFrame(np.where(A.values > 0, idx, np.nan), index=A.index, columns=us).ffill()
    f["since"] = pd.DataFrame(np.minimum(idx - lt.values, 720), index=A.index, columns=us).fillna(720)
    M = np.zeros((len(us), len(us)), np.float32)
    for i, u in enumerate(us):
        nb = [us.index(q) for q in adj[u] if q in us] or [i]
        M[i, nb] = 1.0 / len(nb)
    O = np.zeros((len(us), len(us)), np.float32)                 # other raions of the same oblast
    for i, u in enumerate(us):
        mates = [j for j, v in enumerate(us) if RA.oblast_of(v) == RA.oblast_of(u) and j != i] or [i]
        O[i, mates] = 1.0 / len(mates)
    def mix(X, W):
        Y = pd.DataFrame(X.values @ W.T, index=X.index, columns=us)
        return Y
    f["nb0"], f["nb6"], f["nb24"] = mix(A, M), mix(roll(A, 6), M), mix(roll(A, 24), M)
    f["ob0"], f["ob6"], f["ob24"] = mix(A, O), mix(roll(A, 6), O), mix(roll(A, 24), O)
    nat0 = A.mean(1)
    nat24 = nat0.rolling(24, min_periods=1).mean()
    rows = []
    for j, u in enumerate(us):
        d = pd.DataFrame({k: v[u].values for k, v in f.items()}, index=A.index)
        d["nat0"], d["nat24"] = nat0.values, nat24.values
        d["region"] = j
        rows.append(d.astype(np.float32))
    X = pd.concat(rows, keys=us, names=["r", "hour"]).reset_index()
    h = X.hour
    X["hs"], X["hc"] = np.sin(2 * np.pi * h.dt.hour / 24), np.cos(2 * np.pi * h.dt.hour / 24)
    doy = h.dt.dayofyear
    X["ds"], X["dc"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    X["dow"] = h.dt.dayofweek
    return X


def phase_train():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.isotonic import IsotonicRegression
    pres = presence()
    adj = json.loads((OUT / "adjacency.json").read_text())
    X = features(pres, adj)
    hours = pres.loc[START:END].index
    t_val, t_test = split_times(hours)
    (OUT / "split.json").write_text(json.dumps({"t_val": str(t_val), "t_test": str(t_test)}))
    print("split", t_val, t_test, flush=True)
    cols = [c for c in X.columns if c not in ("r", "hour")]
    us = list(pres.columns)
    rj = {u: j for j, u in enumerate(us)}
    Apos = pd.Series(np.arange(len(pres)), index=pres.index)
    hi = Apos[X.hour].values
    ri = X.r.map(rj).values
    Av = pres.values
    Xv = X[cols].values.astype(np.float32)
    cat = [cols.index("region")]
    tr = (X.hour < t_val).values
    va = ((X.hour >= t_val) & (X.hour < t_test)).values
    out_m = (X.hour >= t_val - pd.Timedelta(hours=1)).values
    sub = (X.hour.dt.hour % 2 == 0).values                         # every second hour for training
    hour_out, r_out = X.hour.values[out_m], X.r.values[out_m]
    res = []
    for k in range(1, K_MAX + 2):
        y = Av[hi + k, ri]
        tgt = X.hour + pd.Timedelta(hours=k)
        m_tr = tr & sub & (tgt < t_val).values
        m_va = va & (tgt < t_test).values
        clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08, max_leaf_nodes=63,
                                             l2_regularization=1.0, early_stopping=False, random_state=SEED,
                                             categorical_features=cat)
        clf.fit(Xv[m_tr], y[m_tr])
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(
            clf.predict_proba(Xv[m_va])[:, 1], y[m_va])
        p = iso.predict(clf.predict_proba(Xv[out_m])[:, 1]).astype(np.float32)
        res.append(pd.DataFrame({"hour": hour_out, "oblast": r_out, "k": np.int8(k), "p": p}))
        print("k", k, "train", int(m_tr.sum()), flush=True)
    pd.concat(res, ignore_index=True).to_parquet(OUT / "direct_probs.parquet")


# ----------------------------------------------------------------------------- evaluation
def active_now(hours, us, log):
    from p3_raion_eval import active_units
    return active_units(hours, us, log, LEVELS)


def markov(pres, hours, pre_mask, act0):
    preA = pres.values[pre_mask]
    a0, a1 = preA[:-1], preA[1:]
    p11 = (a0 * a1).sum(0) / np.maximum(a0.sum(0), 1)
    p01 = ((1 - a0) * a1).sum(0) / np.maximum((1 - a0).sum(0), 1)
    pi = p01 / np.maximum(1 - p11 + p01, 1e-9)
    lam = p11 - p01
    prev = pres.reindex(hours - pd.Timedelta(hours=1)).values
    jj = np.arange(K_MAX + 1)
    PM = pi[None, None] + (prev[:, None, :] - pi[None, None]) * lam[None, None] ** (jj[None, :, None] + 1)
    PM[:, 0, :] = np.maximum(PM[:, 0, :], act0)
    return PM, float(np.mean(lam))


def calendar(pres, hours, pre_mask):
    Ap = pres.values
    posA = pd.Series(np.arange(len(pres)), index=pres.index)
    how = lambda ix: (pres.index[ix].dayofweek * 24 + pres.index[ix].hour).values
    pm = np.where(pre_mask)[0]
    rate = pd.DataFrame(Ap[pm]).groupby(how(pm)).mean().reindex(range(168)).values
    ii = posA[hours].values
    return np.stack([rate[how(ii + j)] for j in range(K_MAX + 1)], axis=1)


def phase_eval():
    from scipy.stats import wilcoxon
    from sklearn.metrics import roc_auc_score
    import networkx as nx
    from p3_eval import select
    pres = presence()
    us = list(pres.columns)
    log = RA.read_log(LOG2)
    sp = json.loads((OUT / "split.json").read_text())
    t_test = pd.Timestamp(sp["t_test"])
    hours = pd.date_range(t_test, END - pd.Timedelta(hours=K_MAX), freq="h", tz="UTC")
    pre = pd.date_range(START, t_test - pd.Timedelta(hours=K_MAX + 1), freq="h", tz="UTC")
    pre_mask = (pres.index >= START) & (pres.index < t_test)
    import pyarrow.parquet as pq
    import pyarrow.compute as pc
    tb = pq.read_table(OUT / "direct_probs.parquet", read_dictionary=["oblast"])
    tb = tb.filter(pc.greater_equal(tb["hour"], (hours[0] - pd.Timedelta(hours=1)).tz_convert(None).to_pydatetime()))
    dp = tb.to_pandas()
    dp["hour"] = pd.to_datetime(dp["hour"], utc=True)
    act0 = active_now(hours, us, log)
    P = C.causal_P(dp, hours, us, act0)
    PM, lam = markov(pres, hours, pre_mask, act0)
    PC = calendar(pres, hours, pre_mask)
    Ap = pres.values
    ii = pd.Series(np.arange(len(pres)), index=pres.index)[hours].values
    lead = {}
    for j in (0, 1, 3, 6, 12, 18):
        y = Ap[ii + j].ravel()
        base = y.mean()
        bs = lambda q: float(1 - np.mean((q - y) ** 2) / np.mean((base - y) ** 2))
        lead[j] = {"auc_ml": float(roc_auc_score(y, P[:, j].ravel())),
                   "auc_markov": float(roc_auc_score(y, PM[:, j].ravel())),
                   "bss_ml": bs(P[:, j].ravel()), "base": float(base)}
    print("lead", {j: (round(v["auc_ml"], 3), round(v["auc_markov"], 3), round(v["bss_ml"], 3)) for j, v in lead.items()},
          flush=True)
    names = ["pred", "markov", "calendar", "reactive", "oracle"]
    rr = np.arange(len(hours))

    def run(c):
        oi = [us.index(o) for o in c.obl]
        K = c.metric(hours)
        i_s = int(c.metric(pre).mean(0).argmin())
        ch = {"static": np.full(len(hours), i_s),
              "pred": select(c.expected_metric(P[:, :, oi]), i_s),
              "markov": select(c.expected_metric(PM[:, :, oi]), i_s),
              "calendar": select(c.expected_metric(PC[:, :, oi]), i_s),
              "reactive": select(c.current_metric(act0[:, oi]), i_s),
              "oracle": K.argmin(1)}
        return {k: K[rr, v] for k, v in ch.items()}, ch, i_s

    summary = {"n_hours": len(hours), "test_start": str(t_test), "test_end": str(hours[-1]), "split": sp,
               "n_units": len(us), "base_rate_test": float(pres.loc[hours[0]:hours[-1]].values.mean()),
               "markov_lambda_mean": lam, "lead": lead, "corridors": {}}
    m = hours.tz_convert(None).to_period("M").astype(str)
    for key in G.CORRIDORS:
        c0 = C.Corridor(key, C.presence())                     # same candidate set and ids as the paper
        routes = [r for r in RA.corridor_routes_raion(key) if r["id"] in c0.ids]
        c = C.Corridor(key, pres, routes=routes)
        d, ch, i_s = run(c)
        res = {"static_route": c.ids[i_s], "static_KUA": float(d["static"].mean())}
        for n in names:
            res[n] = {"impr": C.impr(d["static"], d[n]), "ci": C.block_bootstrap_impr(d["static"], d[n]),
                      "dT_min": float(60 * (c.T[ch[n]].mean() - c.T[i_s])),
                      "monthly": {mm: C.impr(d["static"][m == mm], d[n][m == mm]) for mm in sorted(set(m))}}
        summary["corridors"][key] = res
        print(key, res["static_route"], round(res["static_KUA"], 1),
              {n: (round(res[n]["impr"], 2), [round(q, 1) for q in res[n]["ci"]]) for n in names}, flush=True)

    os.environ.setdefault("P3_GRAPH_DIR", str(C.P3_IN / "roadgraph_osm_primary"))
    import p3_graph_candidates as GC
    from p3_graph_eval import od_pairs
    g, cities = GC.load_graph()
    E = pd.read_csv(C.P3_IN / "roadgraph_osm_primary_raion" / "edges.csv")
    gr = nx.Graph()
    for r in E.itertuples():
        gr.add_edge(int(r.u), int(r.v), length_m=r.length_m, time_s=r.time_s,
                    profile=json.loads(r.profile_json), fwd=(int(r.u), int(r.v)))
    csr = GC.Csr(g)
    rows, pooled = [], {k: [] for k in ["static"] + names}
    for a_, b_ in od_pairs(g, cities):
        cand = GC.penalty_lo(g, cities[a_], cities[b_], 10, csr=csr)
        c = C.Corridor("od", pres, routes=[GC.path_route(gr, p, f"{a_}-{b_}#{i}") for i, p in enumerate(cand)])
        d, ch, i_s = run(c)
        for k in pooled:
            pooled[k].append(d[k])
        diff = d["static"] - d["pred"]
        pval = float(wilcoxon(d["static"], d["pred"]).pvalue) if np.any(diff) else 1.0
        rows.append({"od": f"{a_}-{b_}", "n_cand": len(c.ids), "static_km": float(d["static"].mean()),
                     **{f"{k}_red_pct": C.impr(d["static"], d[k]) for k in names}, "pred_p": pval,
                     "dT_min": float(60 * (c.T[ch["pred"]].mean() - c.T[i_s]))})
    per = pd.DataFrame(rows)
    per.to_csv(OUT / "per_od_osm_primary.csv", index=False)
    S = {k: np.sum(v, axis=0) for k, v in pooled.items()}
    summary["osm_primary"] = {
        "n_od": len(per), "pooled_red_pct": {k: C.impr(S["static"], S[k]) for k in names},
        "pooled_ci_pred": C.block_bootstrap_impr(S["static"], S["pred"]),
        "pooled_ci_markov": C.block_bootstrap_impr(S["static"], S["markov"]),
        "share_better": float((per.pred_red_pct > 0).mean()),
        "share_signif": float(((per.pred_p < 0.05) & (per.pred_red_pct > 0)).mean()),
        "share_signif_neg": float(((per.pred_p < 0.05) & (per.pred_red_pct < 0)).mean()),
        "pred_beats_markov_share": float((per.pred_red_pct > per.markov_red_pct).mean()),
        "mean_dT_min": float(per.dT_min.mean()), "static_km_mean": float(per.static_km.mean())}
    print("osm", json.dumps(summary["osm_primary"], indent=1), flush=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True)
    ph = ap.parse_args().phase
    {"data": phase_data, "train": phase_train, "eval": phase_eval}[ph]()
