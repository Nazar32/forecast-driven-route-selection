"""
Paper 3: additional analyses for the manuscript (all causal, same evaluation as p3_eval.py).
  lead      per-lead AUC/Brier/Brier-skill and reliability bins of the direct models (test block)
  depart    DUA of static vs schedule-aware choice by local departure hour (Europe/Kyiv)
  subgroup  reduction on hour subsets (exposed static route, candidates differ, national activity)
  wtl       share of hours the predictive choice is better / equal / worse than static
  hyst      switching threshold (hysteresis) on E[DUA]: reduction vs number of route changes
  case      the Dnipro-Kyiv departure with the largest km-under-alert saving (illustration)
Output: results/p3/extra_summary.json
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).parent))
import p3_core as C  # noqa
import p3_geo as G   # noqa
from p3_eval import load_direct, select  # noqa

THETAS = [0.0, 0.0025, 0.005, 0.01, 0.02]


def boot_subset(base, x, mask, block=24, n=2000, seed=7):
    rng = np.random.default_rng(seed)
    H = len(base); nb = int(np.ceil(H / block))
    st = rng.integers(0, H - block + 1, size=(n, nb))
    idx = (st[:, :, None] + np.arange(block)[None, None, :]).reshape(n, -1)[:, :H]
    m = mask[idx]
    b = np.where(m, base[idx], 0).sum(1) / np.maximum(m.sum(1), 1)
    xx = np.where(m, x[idx], 0).sum(1) / np.maximum(m.sum(1), 1)
    v = 100 * (b - xx) / np.where(b > 0, b, np.nan)
    return [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]


def main():
    act = C.presence()
    hours = pd.date_range(C.TEST_START, C.TEST_END - pd.Timedelta(hours=C.K_MAX), freq="h", tz="UTC")
    pre = pd.date_range(C.DATA_START, C.TEST_START - pd.Timedelta(hours=C.K_MAX + 1), freq="h", tz="UTC")
    out = {"lead": {}, "corridors": {}}

    # ---------------- per-lead predictor quality on the test block
    # v2: forecast issued at departure h:00 for lead j = model k=j+1 on feature row h-1
    dp = C.read_probs(C.P3 / "direct_probs.parquet")
    dp = dp[(dp.hour >= C.TEST_START - pd.Timedelta(hours=1)) &
            (dp.hour <= C.TEST_END - pd.Timedelta(hours=C.K_MAX + 1))]
    A = act
    pos = pd.Series(np.arange(len(A)), index=A.index)
    col = {o: j for j, o in enumerate(A.columns)}
    for k, g in dp.groupby("k"):
        g = g[g.oblast.isin(col)]
        y = A.values[pos[g.hour].values + int(k), g.oblast.map(col).values]
        k = int(k) - 1                     # report by lead after departure (0..18)
        p = g.p.values
        base = y.mean(); brier = float(np.mean((p - y) ** 2))
        bins = np.minimum((p * 10).astype(int), 9)
        rel = [{"bin": int(b), "p_mean": float(p[bins == b].mean()), "y_mean": float(y[bins == b].mean()),
                "n": int((bins == b).sum())} for b in range(10) if (bins == b).sum() > 0]
        out["lead"][int(k)] = {"auc": float(roc_auc_score(y, p)), "brier": brier,
                               "bss": float(1 - brier / (base * (1 - base))), "base_rate": float(base),
                               "ece": float(sum(r["n"] * abs(r["p_mean"] - r["y_mean"]) for r in rel) / len(p)),
                               "reliability": rel}
    print("lead", {k: round(v["auc"], 3) for k, v in out["lead"].items()})

    # national activity at decision hour (number of oblasts under alert)
    nat = C.active_at(hours, list(A.columns)).sum(1)     # regions under alert at departure h:00
    for key in G.CORRIDORS:
        c = C.Corridor(key, act)
        DUA = c.metric(hours)                  # v2: km under alert per trip
        SH = c.dua(hours)
        i_stat = int(c.metric(pre).mean(0).argmin())
        P = load_direct(hours, c.obl)
        E = c.expected_metric(P)
        rows = np.arange(len(hours))
        ch = select(E, i_stat)
        base, d = DUA[:, i_stat], DUA[rows, ch]
        res = {}
        # departure hour (local)
        lh = hours.tz_convert("Europe/Kyiv").hour
        res["depart"] = [{"h": int(h), "static": float(base[lh == h].mean()), "pred": float(d[lh == h].mean()),
                          "oracle": float(DUA[lh == h].min(1).mean())} for h in range(24)]
        # subgroups
        spread = DUA.max(1) - DUA.min(1)
        q75 = np.quantile(nat, 0.75)
        groups = {"all": np.ones(len(hours), bool), "static_exposed": base > 0, "candidates_differ": spread > 0,
                  "national_high": nat >= q75, "national_low": nat < np.quantile(nat, 0.5)}
        res["subgroup"] = {n: {"share": float(m.mean()), "impr": C.impr(base[m], d[m]),
                               "ci": boot_subset(base, d, m)} for n, m in groups.items()}
        res["subgroup_q75_regions"] = float(q75)
        # win / tie / loss
        res["wtl"] = {"better": float(np.mean(d < base - 1e-12)), "equal": float(np.mean(np.abs(d - base) <= 1e-12)),
                      "worse": float(np.mean(d > base + 1e-12))}
        res["km_saved_when_better"] = float(np.mean((base - d)[d < base - 1e-12]))
        res["km_lost_when_worse"] = float(np.mean((d - base)[d > base + 1e-12]))
        res["share_static"] = float(SH[:, i_stat].mean()); res["share_pred"] = float(SH[rows, ch].mean())
        # hysteresis
        hy = []
        for th in THETAS:
            cur = i_stat; seq = []
            for t in range(len(hours)):
                b = int(np.argmin(E[t]))
                if (E[t, cur] - E[t, b]) / c.D[i_stat] > th:      # threshold on the share scale
                    cur = b
                seq.append(cur)
            seq = np.array(seq); dd = DUA[rows, seq]
            hy.append({"theta": th, "impr": C.impr(base, dd), "ci": C.block_bootstrap_impr(base, dd),
                       "churn": int(np.sum(seq[1:] != seq[:-1])), "T_h": float(c.T[seq].mean())})
        res["hyst"] = hy
        # case study (Dnipro-Kyiv): largest km saving
        if key == os.environ.get("P3_CASE", "dnipro_kyiv"):
            sav = base - d
            t = int(np.argmax(np.where(ch != i_stat, sav, -np.inf)))
            def trav(ri):
                r = c.routes[ri]
                return [{"oblast": s["oblast"], "t0": s["t_enter_s"] / 3600, "t1": s["t_exit_s"] / 3600,
                         "km": s["distance_m"] / 1000} for s in r["traversal"]]
            i0 = c.pos[hours[t]]
            alert = {o: [int(c.A[i0 + j, c.obl.index(o)]) for j in range(C.K_MAX + 1)] for o in c.obl}
            prob = {o: [float(P[t, j, c.obl.index(o)]) for j in range(C.K_MAX + 1)] for o in c.obl}
            res["case"] = {"hour_utc": str(hours[t]), "hour_local": str(hours[t].tz_convert("Europe/Kyiv")),
                           "static": c.ids[i_stat], "chosen": c.ids[int(ch[t])],
                           "static_T_h": float(c.T[i_stat]), "chosen_T_h": float(c.T[ch[t]]),
                           "static_km_ua": float(base[t]), "chosen_km_ua": float(d[t]),
                           "static_E": float(E[t, i_stat]), "chosen_E": float(E[t, ch[t]]),
                           "static_trav": trav(i_stat), "chosen_trav": trav(int(ch[t])),
                           "alert": alert, "prob": prob}
        out["corridors"][key] = res
        print(key, "wtl", {k: round(v, 3) for k, v in res["wtl"].items()},
              "sub", {k: (round(v["share"], 2), round(v["impr"], 1)) for k, v in res["subgroup"].items()},
              "hyst", [(h["theta"], round(h["impr"], 1), h["churn"]) for h in hy])
    (C.P3 / "extra_summary.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
