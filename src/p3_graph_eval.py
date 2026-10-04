"""
Paper 3 / experiment (d), step 3: does graph-native candidate generation + dynamic
risk weighting raise the achievable exposure reduction?  79 origin-destination pairs
between oblast centres (fastest trip 4-16 h) on the junction-level road graph.

Per OD pair:
  candidates  C_K = first K limited-overlap Yen paths (theta = 0.7, stretch <= 1.5),
              K in {1, 2, 3, 5, 10}
  strategies  (as in p3_eval.py, all causal except oracle / hindsight)
     fastest            C_1
     static_pretest(K)  fixed path in C_K with lowest pre-test DUA   (deployable static)
     reactive(K)        min distance share currently under alert, ties -> static
     pred_sched(K)      min expected DUA from the direct per-lead predictor
     oracle(K)          min realized DUA (ceiling of selection within C_K)
     dyn_path(mu)       hourly risk-weighted shortest path on the WHOLE graph:
                          w_e(t) = time_e[h] + mu * E[km of e driven under alert](t) / 100
                        where the lead used for each piece of e is its arrival time on the
                        origin's fastest-time tree (static per OD, so the weight is a plain
                        per-hour edge property -- exactly what Neo4j GDS Dijkstra consumes).
Metrics: realized DUA (share of km under alert), km under alert per trip, travel time.
Outputs: results/p3/graph_eval_per_od.csv, results/p3/graph_eval_summary.json
"""
from __future__ import annotations

import itertools
import json
import os
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.stats import wilcoxon
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).parent))
import p3_core as C  # noqa: E402
import p3_graph_candidates as GC  # noqa: E402
from p3_eval import select  # noqa: E402

CENTERS = ["Lviv", "Kyiv", "Odesa", "Kharkiv", "Dnipro", "Ternopil", "Rivne", "Zhytomyr",
           "Vinnytsia", "Khmelnytskyi", "Cherkasy", "Kropyvnytskyi", "Poltava", "Mykolaiv", "Lutsk"]
KS = [1, 2, 3, 5, 10]
# configuration for large graphs (defaults reproduce the corridor-graph run)
CAND = os.environ.get("P3_CAND", "yen")                    # yen | penalty
SUFFIX = os.environ.get("P3_OUT_SUFFIX", "")
DYN_STEP = int(os.environ.get("P3_DYN_STEP", "1"))          # evaluate dyn_path every n-th hour
MUS = [float(x) for x in os.environ.get("P3_MUS", "0.5,1,2,4,8,16,32,64").split(",")]
LAMBDAS = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
T_MIN_H, T_MAX_H = 4.0, 16.0


def od_pairs(g, cities):
    out = []
    for a, b in itertools.combinations(CENTERS, 2):
        p = nx.shortest_path(g, cities[a], cities[b], weight="time_s")
        T = GC.path_time(g, p) / 3600
        if T_MIN_H <= T <= T_MAX_H:
            out.append((a, b))
    return out


def full_probs(hours, oblasts, act=None, path=None):
    """v2 causal forecast tensor for all oblasts (lead 0 = max(active at h:00, k=1 forecast))."""
    dp = C.read_probs(path or (C.P3 / "direct_probs.parquet"))
    return C.causal_P(dp, hours, oblasts, C.active_at(hours, oblasts))


def edge_lead_tables(g, src, oblasts, n_lead):
    """Per edge: matrix M_e[lead, oblast] = km of e driven at that lead (from src's fastest tree)."""
    tau = nx.single_source_dijkstra_path_length(g, src, weight="time_s")
    oi = {o: j for j, o in enumerate(oblasts)}
    tabs = {}
    for a, b in g.edges:
        u, v = (a, b) if tau[a] <= tau[b] else (b, a)
        t = tau[u]
        M = np.zeros((n_lead, len(oblasts)), dtype=np.float32)
        for o, d, dt in GC.oriented_profile(g, u, v):
            j = min(int((t + dt / 2) // 3600), n_lead - 1)
            M[j, oi[o]] += d / 1000.0
            t += dt
        tabs[(a, b)] = M
    return tabs


def dyn_paths(g, src, dst, P, oblasts, mu):
    nodes = list(g.nodes)
    ni = {n: i for i, n in enumerate(nodes)}
    E = list(g.edges)
    tabs = edge_lead_tables(g, src, oblasts, P.shape[1])
    Mstack = np.stack([tabs[e] for e in E])                     # E x J x O
    risk_km = np.einsum("hjo,ejo->he", P, Mstack)               # H x E expected km under alert
    base = np.array([g.edges[e]["time_s"] / 3600.0 for e in E])
    rows = [ni[a] for a, b in E] + [ni[b] for a, b in E]
    cols = [ni[b] for a, b in E] + [ni[a] for a, b in E]
    paths = []
    s, d = ni[src], ni[dst]
    for h in range(P.shape[0]):
        w = base + mu * risk_km[h] / 100.0
        m = csr_matrix((np.r_[w, w], (rows, cols)), shape=(len(nodes), len(nodes)))
        _, pred = dijkstra(m, directed=True, indices=s, return_predecessors=True)
        p, x = [], d
        while x != -9999 and x != s:
            p.append(nodes[x]); x = pred[x]
        p.append(src)
        paths.append(tuple(p[::-1]))
    return paths


def dyn_setup(csr, g, src, oblasts, n_lead):
    """Sparse E x (lead*oblast) km matrix; leads from the origin's fastest-time tree."""
    from scipy.sparse.csgraph import dijkstra
    tau = dijkstra(csr.matrix(csr.t), directed=True, indices=csr.ni[src])
    oi = {o: j for j, o in enumerate(oblasts)}
    rows, cols, vals = [], [], []
    for k, (a, b) in enumerate(csr.E):
        ia, ib = csr.ni[a], csr.ni[b]
        u, v = (a, b) if tau[ia] <= tau[ib] else (b, a)
        t = min(tau[ia], tau[ib])
        for o, d, dt in GC.oriented_profile(g, u, v):
            rows.append(k); cols.append(min(int((t + dt / 2) // 3600), n_lead - 1) * len(oblasts) + oi[o])
            vals.append(d / 1000.0)
            t += dt
    return csr_matrix((vals, (rows, cols)), shape=(len(csr.E), n_lead * len(oblasts)))


def dyn_paths_fast(csr, M, src, dst, P, mu):
    base = csr.t / 3600.0
    out = []
    for h in range(P.shape[0]):
        w = base + mu * (M @ P[h].ravel()) / 100.0
        p = csr.path(w, src, dst)
        out.append(tuple(csr.nodes[i] for i in p))
    return out


def main():
    t0 = time.time()
    g, cities = GC.load_graph()
    act_all = C.presence()
    oblasts = list(act_all.columns)
    hours = pd.date_range(C.TEST_START, C.TEST_END - pd.Timedelta(hours=C.K_MAX), freq="h", tz="UTC")
    pre = pd.date_range(C.DATA_START, C.TEST_START - pd.Timedelta(hours=C.K_MAX + 1), freq="h", tz="UTC")
    Pall = full_probs(hours, oblasts, act_all)
    pairs = od_pairs(g, cities)
    csr = GC.Csr(g)
    dyn_idx = np.arange(0, len(hours), DYN_STEP)
    print(f"graph {GC.GDIR.name}: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges; "
          f"{len(pairs)} OD pairs; candidates={CAND}; dyn every {DYN_STEP} h, mu={MUS}", flush=True)
    rows, per_od = [], []
    for (a, b) in pairs:
        s, t = cities[a], cities[b]
        cand = GC.yen_lo(g, s, t, max(KS)) if CAND == "yen" else GC.penalty_lo(g, s, t, max(KS), csr=csr)
        routes = [GC.path_route(g, p, f"{a}-{b}#{i}") for i, p in enumerate(cand)]
        c = C.Corridor(f"{a}-{b}", act_all, routes=routes)
        oi = [oblasts.index(o) for o in c.obl]
        P = Pall[:, :, oi]
        SH = c.dua(hours)                                 # share (secondary)
        DUA, DUAp = c.metric(hours), c.metric(pre)        # v2: km under alert (objective)
        E = c.expected_metric(P)
        cur = C.active_at(hours, c.obl)
        rws = np.arange(len(hours))
        rec = {"od": f"{a}-{b}", "n_cand": len(c.ids), "T_fast_h": float(c.T[0]),
               "D_fast_km": float(c.D[0])}
        i_stat10 = int(DUAp.mean(0).argmin())
        base10 = DUA[:, i_stat10]
        rec["static10_KUA"] = float(base10.mean())
        rec["static10_DUA"] = float(SH[:, i_stat10].mean())
        rec["static10_T"] = float(c.T[i_stat10])
        rec["fastest_KUA"] = float(DUA[:, 0].mean())
        rec["fastest_DUA"] = float(SH[:, 0].mean())
        for K in KS:
            k = min(K, len(c.ids))
            sub = slice(0, k)
            i_stat = int(DUAp[:, sub].mean(0).argmin())
            for name, ch in {
                "static": np.full(len(hours), i_stat),
                "reactive": select(c.current_metric(cur)[:, sub], i_stat),
                "pred_sched": select(E[:, sub], i_stat),
                "oracle": DUA[:, sub].argmin(1),
            }.items():
                d = DUA[rws, ch]
                rec[f"{name}_K{K}_DUA"] = float(SH[rws, ch].mean())
                rec[f"{name}_K{K}_KUA"] = float(d.mean())
                rec[f"{name}_K{K}_T"] = float(c.T[ch].mean())
                if name == "pred_sched" and K == 10:
                    diff = base10 - d
                    rec["pred_K10_p"] = float(wilcoxon(base10, d).pvalue) if np.any(diff) else 1.0
        # Lagrangian front for candidate selection (K = 10): min E[DUA] + lam * (T/T_fast - 1)
        for lam in LAMBDAS:
            ch = select(E / c.D[i_stat10] + lam * (c.T / c.T[0] - 1)[None, :], i_stat10)
            d = DUA[rws, ch]
            rec[f"cand_lam{lam}_DUA"] = float(SH[rws, ch].mean())
            rec[f"cand_lam{lam}_KUA"] = float(d.mean())
            rec[f"cand_lam{lam}_T"] = float(c.T[ch].mean())
        # dynamic whole-graph risk-weighted path (every DYN_STEP-th hour)
        cache = {}
        M = dyn_setup(csr, g, s, oblasts, Pall.shape[1])
        base_sub = base10[dyn_idx]
        for mu in MUS:
            ps = dyn_paths_fast(csr, M, s, t, Pall[dyn_idx], mu)
            uniq = sorted(set(ps))
            for p in uniq:
                if p not in cache:
                    r = GC.path_route(g, list(p), "dyn")
                    cc = C.Corridor("dyn", act_all, routes=[r])
                    cache[p] = (cc.dua(hours)[:, 0], cc.D[0], cc.T[0])
            d = np.array([cache[p][0][h] for h, p in zip(dyn_idx, ps)])
            D = np.array([cache[p][1] for p in ps]); T = np.array([cache[p][2] for p in ps])
            rec[f"dyn_mu{mu}_DUA"] = float(d.mean())
            rec[f"dyn_mu{mu}_static_KUA"] = float(base_sub.mean())
            rec[f"dyn_mu{mu}_KUA"] = float((d * D).mean())
            rec[f"dyn_mu{mu}_T"] = float(T.mean())
            rec[f"dyn_mu{mu}_n_paths"] = len(uniq)
        per_od.append(rec)
        print(f"{a}-{b}: n={len(c.ids)} static10={rec['static10_KUA']:.2f}km "
              f"predK10={rec['pred_sched_K10_KUA']:.2f} oracleK10={rec['oracle_K10_KUA']:.2f} "
              f"dyn(mu={MUS[-1]})={rec[f'dyn_mu{MUS[-1]}_KUA']:.2f} ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(per_od)
    df.to_csv(C.P3 / f"graph_eval_per_od{SUFFIX}.csv", index=False)
    summ = {"n_od": len(df), "n_hours": len(hours)}
    def rel(col, base="static10_KUA"):
        v = 100 * (df[base] - df[col]) / df[base]
        return {"mean_pct": float(v.mean()), "median_pct": float(v.median()),
                "pooled_pct": float(100 * (df[base].sum() - df[col].sum()) / df[base].sum()),
                "share_od_better": float((v > 0).mean())}
    for K in KS:
        for n in ("static", "reactive", "pred_sched", "oracle"):
            summ[f"{n}_K{K}"] = rel(f"{n}_K{K}_KUA") | {"mean_T_h": float(df[f"{n}_K{K}_T"].mean())} | \
                {"share_pooled_pct": rel(f"{n}_K{K}_DUA", base="static10_DUA")["pooled_pct"]}
    for mu in MUS:
        summ[f"dyn_mu{mu}"] = rel(f"dyn_mu{mu}_KUA", base=f"dyn_mu{mu}_static_KUA") | {"mean_T_h": float(df[f"dyn_mu{mu}_T"].mean())}
    for lam in LAMBDAS:
        summ[f"cand_lam{lam}"] = rel(f"cand_lam{lam}_KUA") | {"mean_T_h": float(df[f"cand_lam{lam}_T"].mean())}
    summ["pooled_DUA"] = {c: float(df[c].mean()) for c in df.columns if c.endswith("_DUA")}
    summ["pooled_KUA"] = {c: float(df[c].mean()) for c in df.columns if c.endswith("_KUA")}
    summ["pooled_T"] = {c: float(df[c].mean()) for c in df.columns if c.endswith("_T") or c.endswith("_T_h")}
    summ["static10_mean_T_h"] = float(df["static10_T"].mean())
    summ["fastest_mean_T_h"] = float(df["T_fast_h"].mean())
    summ["share_od_pred_K10_sig_0.05"] = float((df["pred_K10_p"] < 0.05).mean())
    (C.P3 / f"graph_eval_summary{SUFFIX}.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
