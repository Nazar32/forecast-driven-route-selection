"""
Paper 3: sensitivity of graph-native candidate generation (OSM graphs, penalty method with
limited overlap).  Varies the overlap limit theta, the stretch bound, the penalty factor and the
candidate count K (up to 20); evaluation identical to p3_graph_eval.py (causal schedule-aware rule,
static baseline = best fixed path among the same candidates on pre-test data).
Usage: P3_GRAPH_DIR=results/p3/roadgraph_osm_primary \
       python p3_sens.py --theta 0.7 --stretch 1.5 --factor 1.2 --kmax 20 --tag base
Output: results/p3/sens/<graph>_<tag>.json
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
import p3_core as C  # noqa
import p3_graph_candidates as GC  # noqa
from p3_eval import select  # noqa
from p3_graph_eval import od_pairs, full_probs  # noqa

KS = [1, 2, 3, 5, 10, 15, 20]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--theta", type=float, default=0.7)
    ap.add_argument("--stretch", type=float, default=1.5)
    ap.add_argument("--factor", type=float, default=1.2)
    ap.add_argument("--kmax", type=int, default=20)
    ap.add_argument("--maxiter", type=int, default=200)
    ap.add_argument("--tag", default="base")
    ap.add_argument("--method", default="penalty", choices=["penalty", "yenraw"])
    ap.add_argument("--od", default="", help="slice start:end of the OD list")
    a = ap.parse_args()
    t0 = time.time()
    g, cities = GC.load_graph()
    act = C.presence()
    obl = list(act.columns)
    hours = pd.date_range(C.TEST_START, C.TEST_END - pd.Timedelta(hours=C.K_MAX), freq="h", tz="UTC")
    pre = pd.date_range(C.DATA_START, C.TEST_START - pd.Timedelta(hours=C.K_MAX + 1), freq="h", tz="UTC")
    Pall = full_probs(hours, obl)
    CUR = C.active_at(hours, obl)
    pairs = od_pairs(g, cities)
    if a.od:
        i0, i1 = (int(x) for x in a.od.split(":")); pairs = pairs[i0:i1]
    csr = GC.Csr(g)
    recs = []
    for (o, d) in pairs:
        s, t = cities[o], cities[d]
        if a.method == "yenraw":
            cand = GC.yen_raw(g, s, t, a.kmax)
        else:
            cand = GC.penalty_lo(g, s, t, a.kmax, theta=a.theta, stretch=a.stretch, factor=a.factor,
                                 max_iter=a.maxiter, csr=csr)
        routes = [GC.path_route(g, p, f"{o}-{d}#{i}") for i, p in enumerate(cand)]
        c = C.Corridor(f"{o}-{d}", act, routes=routes)
        oi = [obl.index(x) for x in c.obl]
        P = Pall[:, :, oi]
        DUA, DUAp = c.metric(hours), c.metric(pre)      # v2: km under alert
        E = c.expected_metric(P)
        cur = CUR[:, oi]
        rws = np.arange(len(hours))
        # pairwise overlap (shared length / shorter length) of the first 10 candidates
        k10 = min(10, len(cand))
        ov = [GC.shared_len(g, cand[i], cand[j]) / min(GC.path_len(g, cand[i]), GC.path_len(g, cand[j]))
              for i in range(k10) for j in range(i + 1, k10)]
        rec = {"od": f"{o}-{d}", "n_cand": len(c.ids), "T_fast": float(c.T[0]), "overlap10": float(np.mean(ov)) if ov else np.nan}
        i10 = int(DUAp[:, :min(10, len(c.ids))].mean(0).argmin())
        rec["base10_KUA"] = float(DUA[:, i10].mean()); rec["base10_T"] = float(c.T[i10])
        for K in KS:
            k = min(K, len(c.ids)); sub = slice(0, k)
            i_s = int(DUAp[:, sub].mean(0).argmin())
            for n, ch in {"static": np.full(len(hours), i_s), "reactive": select(c.current_metric(cur)[:, sub], i_s),
                          "pred": select(E[:, sub], i_s), "oracle": DUA[:, sub].argmin(1)}.items():
                rec[f"{n}_K{K}_KUA"] = float(DUA[rws, ch].mean()); rec[f"{n}_K{K}_T"] = float(c.T[ch].mean())
        recs.append(rec)
        print(f"{o}-{d} n={len(c.ids)} ({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(recs)
    b = df["base10_KUA"].sum()
    out = {"args": vars(a), "graph": GC.GDIR.name, "n_od": len(df), "mean_n_cand": float(df.n_cand.mean()),
           "share_od_10": float((df.n_cand >= 10).mean()), "share_od_20": float((df.n_cand >= 20).mean()),
           "overlap10": float(df.overlap10.mean()), "base10_T": float(df.base10_T.mean()), "sec": time.time() - t0}
    for K in KS:
        for n in ("static", "reactive", "pred", "oracle"):
            out[f"{n}_K{K}"] = {"pooled_pct": float(100 * (b - df[f"{n}_K{K}_KUA"].sum()) / b),
                                "share_better": float((df[f"{n}_K{K}_KUA"] < df["base10_KUA"] - 1e-12).mean()),
                                "T_h": float(df[f"{n}_K{K}_T"].mean())}
        # gain relative to the static baseline of the same K
        bk = df[f"static_K{K}_KUA"].sum()
        out[f"pred_vs_staticK_K{K}"] = float(100 * (bk - df[f"pred_K{K}_KUA"].sum()) / bk)
        out[f"oracle_vs_staticK_K{K}"] = float(100 * (bk - df[f"oracle_K{K}_KUA"].sum()) / bk)
    od = C.P3 / "sens"; od.mkdir(exist_ok=True)
    df.to_csv(od / f"{GC.GDIR.name}_{a.tag}.csv", index=False)
    (od / f"{GC.GDIR.name}_{a.tag}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k.startswith(("pred_K10", "oracle_K10", "mean_n", "overlap", "sec"))}))


if __name__ == "__main__":
    main()
