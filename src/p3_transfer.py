"""
Paper 3, transfer experiment: the same technology on a different hazard and country.

Hazard  : road-relevant severe-weather events of the NOAA NCEI Storm Events Database, 2021-2025,
          aggregated to an hourly presence a(u, i) per state (48 CONUS states + DC), exactly as the
          oblast-level air-alert presence of the case study.
Network : Natural Earth 1:10m US highway graph (p3_transfer_graph.py), state profiles.
Forecast: 19 direct per-lead models (same learner, hyperparameters, split rule, embargo and isotonic
          calibration as p3_direct_horizon.py) on generic features (own history, neighbouring states,
          national activity, calendar); nothing is tuned on the test block.
Decision: the causal rule of the paper (row t-1, model k=j+1, lead 0 = max(active at t:00, g_1)),
          K = 10 limited-overlap penalty candidates, static baseline = fewest pre-test km under event.

  python p3_transfer.py --phase data|train|eval
Outputs: results/p3v2/transfer/{presence.parquet, direct_probs.parquet, transfer_summary.json, per_od.csv}
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import p3_core as C  # noqa: E402

DATA = C.DATA / "transfer"
OUT = C.P3 / "transfer"
GDIR = OUT / "roadgraph_us"
SEED = 42
K_MAX = C.K_MAX                       # 18: leads 0..18, direct models k = 1..19
START = pd.Timestamp("2021-01-01 00:00", tz="UTC")
END = pd.Timestamp("2025-12-31 23:00", tz="UTC")
# hazard set fixed before any routing run: events that stop or slow road freight
TYPES = {"Blizzard", "Winter Storm", "Ice Storm", "Heavy Snow", "Lake-Effect Snow", "Freezing Fog",
         "Dense Fog", "Dust Storm", "High Wind", "Flash Flood", "Tornado"}
MIN_H = 1                             # point events cover the hour they occur in
CENTERS = ["New York", "Chicago", "Philadelphia", "Dallas", "Atlanta", "Boston", "Houston",
           "Washington,  D.C.", "Detroit", "Minneapolis", "Denver", "St. Louis", "Kansas City",
           "Indianapolis", "Pittsburgh", "Cleveland", "Cincinnati", "Nashville", "Memphis",
           "Oklahoma City", "Omaha", "Charlotte", "Columbus", "Milwaukee", "Louisville"]
T_MIN_H, T_MAX_H = 4.0, 16.0


# ----------------------------------------------------------------------------- data
def tz_offset(s):
    """'CST-6' -> +6 h to UTC; 'EST-5' -> +5 h."""
    s = str(s)
    num = "".join(ch for ch in s if ch.isdigit() or ch == "-").split("-")[-1]
    return int(num) if num else 0


def load_events():
    fs = sorted(glob.glob(str(DATA / "StormEvents_details-ftp_v1.0_d20*.csv.gz")))
    assert fs, "put the NCEI StormEvents_details files for 2021-2025 into data/transfer/"
    ev = pd.concat([pd.read_csv(f, low_memory=False) for f in fs], ignore_index=True)
    ev = ev[ev.EVENT_TYPE.isin(TYPES)].copy()
    off = pd.to_timedelta(ev.CZ_TIMEZONE.map(tz_offset), unit="h")
    def ts(ym, d, hm):
        ym, d, hm = ym.astype(int), d.astype(int), hm.astype(int)
        return pd.to_datetime(dict(year=ym // 100, month=ym % 100, day=d, hour=hm // 100, minute=hm % 100))
    ev["s"] = (ts(ev.BEGIN_YEARMONTH, ev.BEGIN_DAY, ev.BEGIN_TIME) + off).dt.tz_localize("UTC")
    ev["f"] = (ts(ev.END_YEARMONTH, ev.END_DAY, ev.END_TIME) + off).dt.tz_localize("UTC")
    ev.loc[ev.f < ev.s, "f"] = ev.s
    return ev


def regions():
    import geopandas as gpd
    st = gpd.read_file(DATA / "ne_10m_admin_1_states_provinces.shp")
    st = st[(st.adm0_a3 == "USA") & ~st.postal.isin({"AK", "HI"})].to_crs("EPSG:5070")
    name2 = {n.upper(): p for n, p in zip(st.name, st.postal)}
    name2["DISTRICT OF COLUMBIA"] = "DC"
    adj = {p: sorted(q for q, g2 in zip(st.postal, st.geometry) if q != p and g.buffer(2000).intersects(g2))
           for p, g in zip(st.postal, st.geometry)}
    return sorted(st.postal), name2, adj


def phase_data():
    OUT.mkdir(parents=True, exist_ok=True)
    reg, name2, adj = regions()
    ev = load_events()
    ev["r"] = ev.STATE.str.upper().map(name2)
    ev = ev.dropna(subset=["r"])
    full = pd.date_range(START, END + pd.Timedelta(hours=K_MAX + 2), freq="h", tz="UTC")
    pos = pd.Series(np.arange(len(full)), index=full)
    A = np.zeros((len(full), len(reg)), np.int8)
    col = {r: j for j, r in enumerate(reg)}
    s0 = ev.s.dt.floor("h"); f0 = (ev.f - pd.Timedelta(seconds=1)).dt.floor("h")
    f0 = np.maximum(f0, s0 + pd.Timedelta(hours=MIN_H - 1))
    for r_, a_, b_ in zip(ev.r, s0, f0):
        if b_ < full[0] or a_ > full[-1]:
            continue
        A[pos[max(a_, full[0])]:pos[min(b_, full[-1])] + 1, col[r_]] = 1
    pres = pd.DataFrame(A, index=full, columns=reg)
    pres.to_parquet(OUT / "presence.parquet")
    ev[["r", "s", "f", "EVENT_TYPE"]].to_parquet(OUT / "events.parquet")
    (OUT / "adjacency.json").write_text(json.dumps(adj))
    inb = pres.loc[START:END]
    print("events", len(ev), "regions", len(reg), "base rate %.3f" % inb.values.mean(),
          "per-region min/med/max %.3f %.3f %.3f" % tuple(np.percentile(inb.mean().values, [0, 50, 100])))
    print(ev.EVENT_TYPE.value_counts().to_string())


def presence():
    return pd.read_parquet(OUT / "presence.parquet")


def active_at(hours, reg):
    ev = pd.read_parquet(OUT / "events.parquet")
    out = np.zeros((len(hours), len(reg)))
    col = {r: j for j, r in enumerate(reg)}
    hi = pd.Series(np.arange(len(hours)), index=hours)
    for r_, s, f in zip(ev.r, ev.s, ev.f):
        if r_ not in col:
            continue
        f = max(f, s + pd.Timedelta(minutes=1))
        a = s.ceil("h"); b = (f - pd.Timedelta(seconds=1)).floor("h")   # instants h:00 with s <= h < f
        if b < a or b < hours[0] or a > hours[-1]:
            continue
        sl = hi[max(a, hours[0]):min(b, hours[-1])].values
        out[sl, col[r_]] = 1
    return out


# ----------------------------------------------------------------------------- features + models
def split_times(idx):
    n = len(idx); return idx[int(0.70 * n)], idx[int(0.85 * n)]


def features(pres, adj):
    A = pres.loc[START:END].astype(np.float32)
    reg = list(A.columns)
    roll = lambda w: A.rolling(w, min_periods=1).mean()
    f = {"a0": A, "a1": A.shift(1), "a2": A.shift(2), "a5": A.shift(5),
         "r3": roll(3), "r6": roll(6), "r24": roll(24), "r72": roll(72), "r168": roll(168)}
    # hours since last event (capped at 30 days)
    last = A.copy() * np.nan
    last[A > 0] = 0
    idx = np.arange(len(A))[:, None] * np.ones((1, A.shape[1]))
    lt = pd.DataFrame(np.where(A.values > 0, idx, np.nan), index=A.index, columns=reg).ffill()
    f["since"] = pd.DataFrame(np.minimum(idx - lt.values, 720), index=A.index, columns=reg).fillna(720)
    M = np.zeros((len(reg), len(reg)), np.float32)
    for i, r in enumerate(reg):
        nb = [reg.index(q) for q in adj[r] if q in reg] or [i]
        M[i, nb] = 1.0 / len(nb)
    f["nb0"] = A @ M.T; f["nb0"].columns = reg
    f["nb6"] = (roll(6) @ M.T); f["nb6"].columns = reg
    f["nb24"] = (roll(24) @ M.T); f["nb24"].columns = reg
    nat0 = A.mean(1); nat24 = nat0.rolling(24, min_periods=1).mean()
    rows = []
    for j, r in enumerate(reg):
        d = pd.DataFrame({k: v[r].values for k, v in f.items()}, index=A.index)
        d["nat0"], d["nat24"] = nat0.values, nat24.values
        d["region"] = j
        rows.append(d)
    X = pd.concat(rows, keys=reg, names=["r", "hour"]).reset_index()
    h = X.hour
    X["hs"], X["hc"] = np.sin(2 * np.pi * h.dt.hour / 24), np.cos(2 * np.pi * h.dt.hour / 24)
    doy = h.dt.dayofyear
    X["ds"], X["dc"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    X["dow"] = h.dt.dayofweek
    return X


def phase_train():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.isotonic import IsotonicRegression
    pres = presence(); adj = json.loads((OUT / "adjacency.json").read_text())
    X = features(pres, adj)
    hours = pres.loc[START:END].index
    t_val, t_test = split_times(hours)
    cols = [c for c in X.columns if c not in ("r", "hour")]
    reg = list(pres.columns); rj = {r: j for j, r in enumerate(reg)}
    Apos = pd.Series(np.arange(len(pres)), index=pres.index)
    hi = Apos[X.hour].values; ri = X.r.map(rj).values
    Av = pres.values
    Xv = X[cols].values.astype(np.float32)
    cat = [cols.index("region")]
    tr = (X.hour < t_val).values; va = ((X.hour >= t_val) & (X.hour < t_test)).values
    out_m = (X.hour >= t_val - pd.Timedelta(hours=1)).values
    sub = (X.hour.dt.hour % 2 == 0).values                      # every second hour for training
    res, met = [], {}
    for k in range(1, K_MAX + 2):
        y = Av[hi + k, ri]
        tgt = X.hour + pd.Timedelta(hours=k)
        m_tr = tr & sub & (tgt < t_val).values
        m_va = va & (tgt < t_test).values
        clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08, max_leaf_nodes=63,
                                             l2_regularization=1.0, early_stopping=False, random_state=SEED,
                                             categorical_features=cat)
        clf.fit(Xv[m_tr], y[m_tr])
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(clf.predict_proba(Xv[m_va])[:, 1], y[m_va])
        p = iso.predict(clf.predict_proba(Xv[out_m])[:, 1])
        res.append(pd.DataFrame({"hour": X.hour.values[out_m], "oblast": X.r.values[out_m], "k": k, "p": p}))
        print("k", k, "train", int(m_tr.sum()), flush=True)
    dp = pd.concat(res, ignore_index=True)
    dp.to_parquet(OUT / "direct_probs.parquet")
    (OUT / "split.json").write_text(json.dumps({"t_val": str(t_val), "t_test": str(t_test)}))


# ----------------------------------------------------------------------------- evaluation
def phase_eval():
    from scipy.stats import wilcoxon
    from sklearn.metrics import roc_auc_score
    os.environ["P3_GRAPH_DIR"] = str(GDIR)
    import networkx as nx
    import p3_graph_candidates as GC
    from p3_eval import select
    pres = presence(); reg = list(pres.columns)
    sp = json.loads((OUT / "split.json").read_text())
    t_test = pd.Timestamp(sp["t_test"])
    hours = pd.date_range(t_test, END - pd.Timedelta(hours=K_MAX), freq="h", tz="UTC")
    pre = pd.date_range(START, t_test - pd.Timedelta(hours=K_MAX + 1), freq="h", tz="UTC")
    dp = C.read_probs(OUT / "direct_probs.parquet")
    act0 = active_at(hours, reg)
    P = C.causal_P(dp, hours, reg, act0)                       # H x J x R
    # Markov persistence
    preA = pres.loc[(pres.index >= START) & (pres.index < t_test)].values
    a0, a1 = preA[:-1], preA[1:]
    p11 = (a0 * a1).sum(0) / np.maximum(a0.sum(0), 1); p01 = ((1 - a0) * a1).sum(0) / np.maximum((1 - a0).sum(0), 1)
    pi = p01 / np.maximum(1 - p11 + p01, 1e-9); lam = p11 - p01
    prev = pres.reindex(hours - pd.Timedelta(hours=1)).values
    jj = np.arange(K_MAX + 1)
    PM = pi[None, None] + (prev[:, None, :] - pi[None, None]) * lam[None, None] ** (jj[None, :, None] + 1)
    PM[:, 0, :] = np.maximum(PM[:, 0, :], act0)
    # climatology (hour of week)
    Ap = pres.values; posA = pd.Series(np.arange(len(pres)), index=pres.index)
    how = lambda ix: (pres.index[ix].dayofweek * 24 + pres.index[ix].hour).values
    pm = np.where((pres.index >= START) & (pres.index < t_test))[0]
    rate = pd.DataFrame(Ap[pm]).groupby(how(pm)).mean().reindex(range(168)).values
    ii = posA[hours].values
    PC = np.stack([rate[how(ii + j)] for j in range(K_MAX + 1)], axis=1)
    # forecast quality by lead
    lead = {}
    for j in (0, 1, 3, 6, 12, 18):
        y = Ap[ii + j].ravel()
        lead[j] = {"auc_ml": float(roc_auc_score(y, P[:, j].ravel())), "auc_markov": float(roc_auc_score(y, PM[:, j].ravel())),
                   "base": float(y.mean())}
    print("lead", lead, flush=True)
    g, cities = GC.load_graph(); csr = GC.Csr(g)
    pairs = []
    for a, b in itertools.combinations([c for c in CENTERS if c in cities], 2):
        T = GC.path_time(g, nx.shortest_path(g, cities[a], cities[b], weight="time_s")) / 3600
        if T_MIN_H <= T <= T_MAX_H:
            pairs.append((a, b, T))
    rows, pooled = [], {k: [] for k in ("static", "pred", "markov", "clim", "reactive", "oracle")}
    for a, b, T in pairs:
        cand = GC.penalty_lo(g, cities[a], cities[b], 10, csr=csr)
        c = C.Corridor("od", pres, routes=[GC.path_route(g, p, f"{a}-{b}#{i}") for i, p in enumerate(cand)])
        oi = [reg.index(o) for o in c.obl]
        K = c.metric(hours); i_s = int(c.metric(pre).mean(0).argmin())
        rr = np.arange(len(hours))
        ch = {"static": np.full(len(hours), i_s),
              "pred": select(c.expected_metric(P[:, :, oi]), i_s),
              "markov": select(c.expected_metric(PM[:, :, oi]), i_s),
              "clim": select(c.expected_metric(PC[:, :, oi]), i_s),
              "reactive": select(c.current_metric(act0[:, oi]), i_s),
              "oracle": K.argmin(1)}
        d = {k: K[rr, v] for k, v in ch.items()}
        for k in pooled:
            pooled[k].append(d[k])
        diff = d["static"] - d["pred"]
        pval = float(wilcoxon(d["static"], d["pred"]).pvalue) if np.any(diff) else 1.0
        rows.append({"od": f"{a}-{b}", "T_fast_h": T, "n_cand": len(cand), "static_km": float(d["static"].mean()),
                     **{f"{k}_red_pct": C.impr(d["static"], d[k]) for k in ch if k != "static"},
                     "pred_p": pval, "dT_min": float(60 * (c.T[ch["pred"]].mean() - c.T[i_s]))})
        print(rows[-1]["od"], round(rows[-1]["pred_red_pct"], 1), flush=True)
    per = pd.DataFrame(rows); per.to_csv(OUT / "per_od.csv", index=False)
    S = {k: np.sum(v, axis=0) for k, v in pooled.items()}
    summ = {"n_od": len(per), "n_hours": len(hours), "test_start": str(t_test), "types": sorted(TYPES),
            "base_rate_test": float(pres.loc[hours[0]:hours[-1]].values.mean()), "lead": lead,
            "pooled_red_pct": {k: C.impr(S["static"], S[k]) for k in S if k != "static"},
            "pooled_ci_pred": C.block_bootstrap_impr(S["static"], S["pred"]),
            "pooled_ci_markov": C.block_bootstrap_impr(S["static"], S["markov"]),
            "share_better": float((per.pred_red_pct > 0).mean()),
            "share_signif": float(((per.pred_p < 0.05) & (per.pred_red_pct > 0)).mean()),
            "share_signif_neg": float(((per.pred_p < 0.05) & (per.pred_red_pct < 0)).mean()),
            "pred_beats_markov_share": float((per.pred_red_pct > per.markov_red_pct).mean()),
            "median_pred_pct": float(per.pred_red_pct.median()), "mean_dT_min": float(per.dT_min.mean()),
            "static_km_mean": float(per.static_km.mean())}
    (OUT / "transfer_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps({k: summ[k] for k in summ if k not in ("lead", "types")}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--phase", required=True)
    ph = ap.parse_args().phase
    {"data": phase_data, "train": phase_train, "eval": phase_eval}[ph]()
