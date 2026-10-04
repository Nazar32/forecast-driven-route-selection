"""
Paper 3: multi-criteria selection with the static criteria of the infological model.

Criteria of candidate r at departure hour t (all minimised):
  T_r, E[KUA](r,t) (dynamic, causal per-lead forecasts), and the static infological criteria
  rail, junction, settle_km, bridges, poor_km (p3_infocriteria.py).
Nonlinear trade-off scheme (Eq. 4 of the paper) over all criteria:
  y_k(r) = f_k(r) / sum_q f_k(q),   F(r,t) = sum_k gamma_k / (1 - y_k(r)),   sum_k gamma_k = 1,
  with gamma_T, gamma_E and gamma_S split equally over the five static criteria.
Rules
  static_mc   the infological model with a STATIC safety criterion: E[KUA] replaced by the route's
              mean km under alert on pre-test data -> one fixed route per corridor
  dynamic_mc  the same scheme with the hourly forecast E[KUA](r,t)
  exposure    min E[KUA] (reference)
Output: results/p3v2/multicrit_summary.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import p3_core as C  # noqa: E402
import p3_geo as G  # noqa: E402
from p3_eval import load_direct, select  # noqa: E402

STATIC = ["rail", "junction", "settle_km", "bridges", "poor_km"]
GAMMA_T = 0.2
GAMMA_S = [0.0, 0.1, 0.2, 0.3, 0.4]        # gamma_E = 1 - gamma_T - gamma_S
MAIN_S = 0.2


def ynorm(v):
    s = v.sum(-1, keepdims=True)
    return np.divide(v, s, out=np.zeros_like(v, dtype=float), where=s > 0)


def nl_score(T, E, S, gS):
    """T: R, E: H x R, S: R x 5 -> H x R integrated criterion."""
    gE = 1 - GAMMA_T - gS
    F = GAMMA_T / (1 - ynorm(T))[None, :] + gE / (1 - ynorm(E))
    for j in range(S.shape[1]):
        F = F + (gS / S.shape[1]) / (1 - ynorm(S[:, j]))[None, :]
    return F


def main():
    act = C.presence()
    ic = pd.read_csv(C.P3 / "infocriteria.csv").set_index("route")
    hours = pd.date_range(C.TEST_START, C.TEST_END - pd.Timedelta(hours=C.K_MAX), freq="h", tz="UTC")
    pre = pd.date_range(C.DATA_START, C.TEST_START - pd.Timedelta(hours=C.K_MAX + 1), freq="h", tz="UTC")
    rows = np.arange(len(hours))
    out = {}
    for key in G.CORRIDORS:
        c = C.Corridor(key, act)
        K = c.metric(hours)
        Kpre = c.metric(pre).mean(0)
        i_stat = int(Kpre.argmin())
        E = c.expected_metric(load_direct(hours, c.obl))
        S = ic.loc[c.ids, STATIC].values.astype(float)
        T = c.T
        res = {"routes": c.ids, "static_criteria": S.tolist(), "static_km": c.ids[i_stat]}

        def describe(ch, base):
            d = K[rows, ch]
            e = {"KUA": float(d.mean()), "T_h": float(T[ch].mean()),
                 "impr_vs_base_pct": C.impr(K[rows, base], d),
                 "ci": C.block_bootstrap_impr(K[rows, base], d),
                 "churn": int(np.sum(ch[1:] != ch[:-1])),
                 "routes_used": sorted({c.ids[int(x)] for x in np.unique(ch)})}
            for j, n in enumerate(STATIC):
                e[n] = float(S[ch, j].mean())
            return e

        for gS in GAMMA_S:
            Fs = nl_score(T, np.tile(Kpre, (1, 1)), S, gS)[0]       # static safety criterion
            i_mc = int(Fs.argmin())
            base = np.full(len(hours), i_mc)
            dyn = select(nl_score(T, E, S, gS), i_mc)
            res[f"gS{gS}"] = {"static_mc_route": c.ids[i_mc],
                              "static_mc": describe(base, base),
                              "dynamic_mc": describe(dyn, base),
                              "dynamic_mc_vs_static_km": C.impr(K[:, i_stat], K[rows, dyn])}
        res["exposure_only"] = describe(select(E, i_stat), np.full(len(hours), i_stat))
        res["static_km_baseline"] = describe(np.full(len(hours), i_stat), np.full(len(hours), i_stat))
        out[key] = res
        m = res[f"gS{MAIN_S}"]
        print(key, "static MC:", m["static_mc_route"], round(m["static_mc"]["KUA"], 1), round(m["static_mc"]["T_h"], 2),
              "| dynamic MC:", round(m["dynamic_mc"]["KUA"], 1), round(m["dynamic_mc"]["T_h"], 2),
              round(m["dynamic_mc"]["impr_vs_base_pct"], 1), [round(x, 1) for x in m["dynamic_mc"]["ci"]],
              {n: (round(m["static_mc"][n], 1), round(m["dynamic_mc"][n], 1)) for n in STATIC}, flush=True)
        for gS in GAMMA_S:
            r = res[f"gS{gS}"]
            print("   gS", gS, r["static_mc_route"], "dyn impr", round(r["dynamic_mc"]["impr_vs_base_pct"], 1),
                  "T", round(r["dynamic_mc"]["T_h"], 2), "vs km-static", round(r["dynamic_mc_vs_static_km"], 1),
                  {n: round(r["dynamic_mc"][n], 1) for n in STATIC})
    (C.P3 / "multicrit_summary.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
