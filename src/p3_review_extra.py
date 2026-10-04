"""
Paper 3, pre-submission experiments motivated by the Paper-2 review history.

1. markov    a statistical forecaster instead of the ML models: a two-state Markov chain per region
             estimated on pre-test hourly presence, p_i(t,j) = pi_i + (a(t-1,i) - pi_i) * lam_i^(j+1)
             (a(t-1,i): last complete hour; lead 0 also max with alerts active at t:00),
             used with the same schedule-aware scoring; corridors + OSM primary (K = 10)
2. tail      tail risk of km under alert per trip: P90, P95, CVaR95 (mean of the worst 5 % of trips),
             share of trips above 100 km; corridors
3. onset     km under alert split into alerts already active at departure and still running when the
             vehicle is in the region ("ongoing") and all others ("new"); static vs predictive choice
Output: results/p3v2/review_extra.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
import p3_core as C  # noqa: E402
import p3_geo as G  # noqa: E402
from p3_eval import load_direct, select  # noqa: E402
from p3_tradeoff import nl_tradeoff  # noqa: E402


def markov_params(act, oblasts):
    pre = act.loc[(act.index >= C.DATA_START) & (act.index < C.TEST_START), oblasts].values
    a0, a1 = pre[:-1], pre[1:]
    p11 = (a0 * a1).sum(0) / np.maximum(a0.sum(0), 1)
    p01 = ((1 - a0) * a1).sum(0) / np.maximum((1 - a0).sum(0), 1)
    pi = p01 / np.maximum(1 - p11 + p01, 1e-9)
    return pi, p11 - p01


def markov_P(act, hours, oblasts):
    pi, lam = markov_params(act, oblasts)
    prev = act.reindex(hours - pd.Timedelta(hours=1))[oblasts].values
    j = np.arange(C.K_MAX + 1)
    P = pi[None, None, :] + (prev[:, None, :] - pi[None, None, :]) * lam[None, None, :] ** (j[None, :, None] + 1)
    P[:, 0, :] = np.maximum(P[:, 0, :], C.active_at(hours, oblasts))
    return P


def tail(x):
    q95 = np.quantile(x, 0.95)
    return {"mean": float(x.mean()), "p90": float(np.quantile(x, 0.90)), "p95": float(q95),
            "cvar95": float(x[x >= q95].mean()), "share_gt100": float(np.mean(x > 100))}


def main():
    act = C.presence()
    hours = pd.date_range(C.TEST_START, C.TEST_END - pd.Timedelta(hours=C.K_MAX), freq="h", tz="UTC")
    pre = pd.date_range(C.DATA_START, C.TEST_START - pd.Timedelta(hours=C.K_MAX + 1), freq="h", tz="UTC")
    rows = np.arange(len(hours))
    out = {"markov": {"corridors": {}}, "tail": {}, "onset": {}}

    # Markov forecast quality by lead on the test block (all regions)
    obl_all = list(act.columns)
    Pm = markov_P(act, hours, obl_all)
    Pml = load_direct(hours, obl_all)
    A = act.values; pos = pd.Series(np.arange(len(act)), index=act.index); i0 = pos[hours].values
    qual = {}
    for j in [0, 1, 3, 6, 12, 18]:
        y = A[i0 + j].ravel()
        qual[j] = {"auc_markov": float(roc_auc_score(y, Pm[:, j, :].ravel())),
                   "auc_ml": float(roc_auc_score(y, Pml[:, j, :].ravel())),
                   "brier_markov": float(np.mean((Pm[:, j, :].ravel() - y) ** 2)),
                   "brier_ml": float(np.mean((Pml[:, j, :].ravel() - y) ** 2))}
    out["markov"]["lead_quality"] = qual
    print("lead quality", {j: (round(v["auc_markov"], 3), round(v["auc_ml"], 3)) for j, v in qual.items()}, flush=True)

    for key in G.CORRIDORS:
        c = C.Corridor(key, act)
        K = c.metric(hours)
        i_stat = int(c.metric(pre).mean(0).argmin())
        base = K[:, i_stat]
        P_ml = load_direct(hours, c.obl)
        P_mk = markov_P(act, hours, c.obl)
        E_ml, E_mk = c.expected_metric(P_ml), c.expected_metric(P_mk)
        cur = C.active_at(hours, c.obl)
        how = lambda idx: (c.act.index[idx].dayofweek * 24 + c.act.index[idx].hour).values
        pre_mask = (c.act.index >= C.DATA_START) & (c.act.index < C.TEST_START)
        rate = pd.DataFrame(c.A[pre_mask]).groupby(how(np.where(pre_mask)[0])).mean().reindex(range(168)).values
        ii = c.pos[hours].values
        Pclim = np.stack([rate[how(ii + j)] for j in range(C.K_MAX + 1)], axis=1)
        ch = {"static": np.full(len(hours), i_stat),
              "reactive": select(c.current_metric(cur), i_stat),
              "climatology": select(c.expected_metric(Pclim), i_stat),
              "markov": select(E_mk, i_stat),
              "predictive": select(E_ml, i_stat),
              "nl_0.8": select(nl_tradeoff(E_ml, c.T, 0.8), i_stat),
              "oracle": K.argmin(1)}
        mk = {}
        for n in ("markov", "predictive"):
            d = K[rows, ch[n]]
            mk[n] = {"impr": C.impr(base, d), "ci": C.block_bootstrap_impr(base, d),
                     "dT_min": float(60 * (c.T[ch[n]].mean() - c.T[i_stat]))}
        out["markov"]["corridors"][key] = mk
        # tail
        out["tail"][key] = {n: tail(K[rows, x]) for n, x in ch.items()}
        # onset decomposition: region-hour (t+j, i) is "ongoing" if i was active at t:00 and has been under
        # alert in every hour t..t+j (the alert that was running at departure has not ended yet)
        W = c.Wh * c.D[:, None, None]                           # R x J x O km
        J = W.shape[1]
        run = cur.astype(bool).copy()                           # H x O
        ong = np.zeros((len(hours), J, len(c.obl)), bool)
        for j in range(J):
            run &= c.A[ii + j].astype(bool)
            ong[:, j, :] = run
        Aj = np.stack([c.A[ii + j] for j in range(J)], axis=1)  # H x J x O
        dec = {}
        for n in ("static", "predictive", "markov", "reactive"):
            Wc = W[ch[n]]                                       # H x J x O
            ongoing = (Wc * Aj * ong).sum((1, 2)); total = (Wc * Aj).sum((1, 2))
            dec[n] = {"ongoing_km": float(ongoing.mean()), "new_km": float((total - ongoing).mean()),
                      "total_km": float(total.mean())}
        s, p = dec["static"], dec["predictive"]
        dec["gain_from_ongoing_km"] = s["ongoing_km"] - p["ongoing_km"]
        dec["gain_from_new_km"] = s["new_km"] - p["new_km"]
        dec["share_hours_active_at_departure_on_static"] = float(np.mean((cur @ c.Wd[i_stat]) > 0))
        out["onset"][key] = dec
        print(key, "markov", round(mk["markov"]["impr"], 1), [round(x, 1) for x in mk["markov"]["ci"]],
              "ml", round(mk["predictive"]["impr"], 1),
              "| tail cvar95 static/pred", round(out["tail"][key]["static"]["cvar95"], 1),
              round(out["tail"][key]["predictive"]["cvar95"], 1),
              "| onset gain ongoing/new", round(dec["gain_from_ongoing_km"], 2), round(dec["gain_from_new_km"], 2),
              flush=True)

    # network, OSM primary, K = 10 penalty candidates
    os.environ.setdefault("P3_GRAPH_DIR", str(C.P3_IN / "roadgraph_osm_primary"))
    import p3_graph_candidates as GC
    from p3_graph_eval import od_pairs
    g, cities = GC.load_graph()
    csr = GC.Csr(g)
    obl = list(act.columns)
    Pml_all = load_direct(hours, obl)
    Pmk_all = markov_P(act, hours, obl)
    per, names = [], []
    for a_, b_ in od_pairs(g, cities):
        cand = GC.penalty_lo(g, cities[a_], cities[b_], 10, csr=csr)
        c = C.Corridor("od", act, routes=[GC.path_route(g, p, f"{a_}-{b_}#{i}") for i, p in enumerate(cand)])
        oi = [obl.index(o) for o in c.obl]
        K = c.metric(hours); i_s = int(c.metric(pre).mean(0).argmin())
        d_ml = K[rows, select(c.expected_metric(Pml_all[:, :, oi]), i_s)]
        d_mk = K[rows, select(c.expected_metric(Pmk_all[:, :, oi]), i_s)]
        per.append((K[:, i_s].mean(), d_ml.mean(), d_mk.mean(),
                    tail(K[:, i_s])["cvar95"], tail(d_ml)["cvar95"]))
        names.append(f"{a_}-{b_}")
    B, Dml, Dmk, Cs, Cp = map(np.array, zip(*per))
    pd.DataFrame({"od": names, "static_KUA": B, "pred_KUA": Dml, "markov_KUA": Dmk, "cvar95_static": Cs,
                  "cvar95_pred": Cp}).to_csv(C.P3 / "review_extra_per_od_osm_primary.csv", index=False)
    pool = lambda x, b=B: float(100 * (b.sum() - x.sum()) / b.sum())
    out["markov"]["osm_primary"] = {"n_od": len(per), "ml_pooled_pct": pool(Dml), "markov_pooled_pct": pool(Dmk),
                                    "ml_better_than_markov_share": float(np.mean(Dml < Dmk)),
                                    "share_better_ml": float(np.mean(Dml < B)), "share_better_markov": float(np.mean(Dmk < B))}
    out["tail"]["osm_primary"] = {"cvar95_static_mean": float(Cs.mean()), "cvar95_pred_mean": float(Cp.mean()),
                                  "cvar95_reduction_pooled_pct": pool(Cp, Cs),
                                  "share_od_cvar_better": float(np.mean(Cp < Cs))}
    print("osm", out["markov"]["osm_primary"], out["tail"]["osm_primary"], flush=True)
    (C.P3 / "review_extra.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
