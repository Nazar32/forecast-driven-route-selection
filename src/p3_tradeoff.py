"""
Paper 3: nonlinear trade-off (Kharchenko & Pysarchuk scheme) as the per-hour decision rule
over the two criteria travel time T and expected distance under alert E[DUA].

For decision hour t and candidate r, criteria are sum-normalised within the candidate set
(as in the manuscript's Eq. 1), y_k(r) = c_k(r) / sum_j c_k(r_j), and the route is chosen by
    r*(t) = argmin_r  sum_k gamma_k / (1 - y_k(r)),   gamma_T + gamma_E = 1.
E[DUA] uses the same causal per-lead forecasts as pred_sched. Evaluation is identical to
p3_eval.py (realized DUA vs the deployable static route, block bootstrap, Wilcoxon).
Output: results/p3/tradeoff_summary.json
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
sys.path.insert(0, str(Path(__file__).parent))
import p3_core as C  # noqa
import p3_geo as G   # noqa
from p3_eval import load_direct, select  # noqa

GAMMAS_E = [0.5, 0.6, 0.7, 0.8, 0.9]


def nl_tradeoff(E, T, gE):
    sE = E.sum(1, keepdims=True)
    yE = np.divide(E, sE, out=np.zeros_like(E), where=sE > 0)
    yT = (T / T.sum())[None, :]
    return gE / (1 - yE) + (1 - gE) / (1 - yT)


def main():
    act_all = C.presence()
    hours = pd.date_range(C.TEST_START, C.TEST_END - pd.Timedelta(hours=C.K_MAX), freq="h", tz="UTC")
    pre = pd.date_range(C.DATA_START, C.TEST_START - pd.Timedelta(hours=C.K_MAX + 1), freq="h", tz="UTC")
    out = {}
    for key in G.CORRIDORS:
        c = C.Corridor(key, act_all)
        DUA = c.metric(hours)                       # v2: km under alert per trip
        SH = c.dua(hours)
        i_stat = int(c.metric(pre).mean(0).argmin())
        P = load_direct(hours, c.obl)               # causal, lead 0 = max(active, k=1)
        E = c.expected_metric(P)
        rows = np.arange(len(hours))
        base = DUA[:, i_stat]
        res = {"static": c.ids[i_stat], "static_T_h": float(c.T[i_stat]), "static_KUA": float(base.mean()),
               "static_share": float(SH[:, i_stat].mean()),
               "fastest_T_h": float(c.T.min())}
        ch = select(E, i_stat)
        res["pred_sched"] = {"KUA": float(DUA[rows, ch].mean()), "share": float(SH[rows, ch].mean()), "T_h": float(c.T[ch].mean()),
                             "impr_pct": C.impr(base, DUA[rows, ch])}
        for gE in GAMMAS_E:
            ch = select(nl_tradeoff(E, c.T, gE), i_stat)
            d = DUA[rows, ch]
            res[f"nl_gE{gE}"] = {"KUA": float(d.mean()), "share": float(SH[rows, ch].mean()), "T_h": float(c.T[ch].mean()),
                                 "dT_vs_static_min": float(60 * (c.T[ch].mean() - c.T[i_stat])),
                                 "impr_pct": C.impr(base, d), "ci": C.block_bootstrap_impr(base, d),
                                 "wilcoxon_p": float(wilcoxon(base, d).pvalue) if np.any(base != d) else 1.0,
                                 "churn": int(np.sum(ch[1:] != ch[:-1]))}
        out[key] = res
        print(key, json.dumps({k: (v if not isinstance(v, dict) else {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()}) for k, v in res.items()}))
    (C.P3 / "tradeoff_summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
