"""
Paper 3 -- mandatory experiments (a) deployable baselines + causal signals,
(b) cost of safety / Pareto trade-off, (c) time-resolved exposure (DUA).

Per corridor (Lviv-Kyiv, Odesa-Kyiv, Kharkiv-Lviv, Dnipro-Kyiv), routes and
oblast segments are recomputed from the ORS geometries (p3_geo.py) so all corridors
use one consistent segmentation, and exact-duplicate candidates are removed.

Decision hours: every test hour t with t + 18 h inside the test block (n = 4692).

Strategies (all CAUSAL unless marked):
  fastest / shortest            fixed route, min T / min D
  static_pretest                fixed route with the lowest realized exposure on all
                                PRE-TEST data (2022-03-15 .. 2025-03-31): the deployable
                                "static safety" baseline
  static_hindsight  [non-causal] fixed route best on the TEST period (upper bound for
                                any fixed choice; this is what the manuscript's R3 is)
  reactive                      min distance share currently under alert (ties -> static)
  climatology                   min expected DUA from pre-test hour-of-week alert rates
  pred_h6_retro     [non-causal] manuscript signal: Paper-2 q_calib looked up at t+5
  pred_h6                       direct 6-h-ahead model, distance-weighted (manuscript rule)
  pred_sched                    direct per-lead models, expected DUA along the route's
                                own time profile (lead j used for the hour the vehicle
                                is in each oblast; current status for j = 0)
  oracle            [non-causal] min realized exposure per hour
Outputs: results/p3/eval_summary.json, results/p3/pareto_<corridor>.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).parent))
import p3_core as C  # noqa: E402
import p3_geo as G  # noqa: E402

EPS = [0.0, 0.02, 0.05, 0.10, 0.20, 0.50, None]
LAMBDAS = [0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]


def select(score, pref):
    """argmin per row; exact ties resolved in favour of route index `pref`."""
    s = score.copy()
    s += 1e-9 * (np.arange(s.shape[1]) != pref)[None, :]
    return s.argmin(1)


_DP = {}


def load_direct(hours, oblasts, path=None):
    """v2 causal forecast tensor incl. lead 0 (see C.causal_P)."""
    path = path or (C.P3 / "direct_probs.parquet")
    if path not in _DP:
        _DP[path] = C.read_probs(path)
    return C.causal_P(_DP[path], hours, oblasts, C.active_at(hours, oblasts)).astype(np.float64)


def load_leaky(hours, oblasts, act_full):
    """v1 input (for comparison only, NON-causal): feature row of hour t (complete only at t+1:00),
    models k = 1..18 used for lead k, lead 0 = alert status over the WHOLE hour t, and the
    full-sample rel_rate_24h normalisation."""
    lk = C.P3 / "direct_probs_leaky.parquet"           # v4: re-trained look-ahead variant if present
    dp = C.read_probs(lk if lk.exists() else C.P3_IN / "direct_probs.parquet")
    dp = dp[dp["hour"].isin(hours) & dp["oblast"].isin(oblasts) & (dp["k"] <= C.K_MAX)]
    P = np.zeros((len(hours), C.K_MAX + 1, len(oblasts)))
    hi = pd.Series(np.arange(len(hours)), index=hours)
    oi = {o: j for j, o in enumerate(oblasts)}
    P[hi[dp["hour"]].values, dp["k"].values.astype(int), dp["oblast"].map(oi).values] = dp["p"].values
    P[:, 0, :] = act_full
    return P


def load_retro(hours, oblasts, h=6):
    if (C.P3 / "direct_probs_leaky.parquet").exists():
        # v4: the misalignment emulated with the direct 1-h model applied to the feature row of hour t+h-1
        dp = C.read_probs(C.P3 / "direct_probs.parquet")
        dp = dp[dp["k"] == 1].pivot_table(index="hour", columns="oblast", values="p")
        look = hours + pd.Timedelta(hours=h - 1)
        return dp.reindex(index=look, columns=oblasts).fillna(0.0).values
    q = pd.read_parquet(C.RES / "method_cache_probs.parquet")
    q["hour"] = pd.to_datetime(q["hour"], utc=True)
    QC = q.pivot_table(index="hour", columns="oblast", values="q_calib")
    look = hours + pd.Timedelta(hours=h - 1)
    return QC.reindex(index=look, columns=oblasts).fillna(0.0).values


def main():
    act_all = C.presence()
    hours = pd.date_range(C.TEST_START, C.TEST_END - pd.Timedelta(hours=C.K_MAX), freq="h", tz="UTC")
    pre = pd.date_range(C.DATA_START, C.TEST_START - pd.Timedelta(hours=C.K_MAX + 1), freq="h", tz="UTC")
    summary = {"n_hours": len(hours), "hours": [str(hours[0]), str(hours[-1])], "corridors": {}}
    pvals = []
    for key in G.CORRIDORS:
        c = C.Corridor(key, act_all)
        R = len(c.ids)
        # v2: the objective and the evaluation metric are ABSOLUTE km under alert per trip
        SH = c.dua(hours)                                        # share of route km (secondary)
        HU = c.hours_under_alert(hours)                          # hours driven under alert
        DUA, AES = c.metric(hours), c.aes_legacy(hours)          # DUA := km under alert
        DUA_pre, AES_pre = c.metric(pre), c.aes_legacy(pre)
        i_fast, i_short = int(c.T.argmin()), int(c.D.argmin())
        i_stat = int(DUA_pre.mean(0).argmin())
        i_stat_aes = int(AES_pre.mean(0).argmin())
        i_hind = int(DUA.mean(0).argmin())
        i_hind_aes = int(AES.mean(0).argmin())

        # signals ---------------------------------------------------------
        A = c.A
        i0 = c.pos[hours].values
        cur = C.active_at(hours, c.obl)                          # alerts active at departure h:00
        P = load_direct(hours, c.obl)                            # lead 0 = max(active, k=1 forecast)
        # climatology: pre-test hour-of-week rate
        how_pre = (c.act.index.dayofweek * 24 + c.act.index.hour).values
        pre_mask = (c.act.index >= C.DATA_START) & (c.act.index < C.TEST_START)
        rate = pd.DataFrame(A[pre_mask]).groupby(how_pre[pre_mask]).mean().reindex(range(168)).values
        how = lambda idx: ((c.act.index[idx].dayofweek * 24 + c.act.index[idx].hour).values)
        Pclim = np.stack([rate[how(i0 + j)] for j in range(C.K_MAX + 1)], axis=1)
        q_retro = load_retro(hours, c.obl)
        P_leak = load_leaky(hours, c.obl, A[i0])
        i_stat_share = int(c.dua(pre).mean(0).argmin())

        choices = {
            "fastest": np.full(len(hours), i_fast),
            "shortest": np.full(len(hours), i_short),
            "static_pretest": np.full(len(hours), i_stat),
            "static_hindsight": np.full(len(hours), i_hind),
            "reactive": select(c.current_metric(cur), i_stat),
            "climatology": select(c.expected_metric(Pclim), i_stat),
            "pred_h6_retro": select(c.current_metric(q_retro), i_stat),
            "pred_h6": select(c.current_metric(P[:, 6, :]), i_stat),
            "pred_sched": select(c.expected_metric(P), i_stat),
            # objective ablation: minimise the expected SHARE of km under alert (baseline by share)
            "static_share": np.full(len(hours), i_stat_share),
            "pred_share": select(c.expected_dua(P), i_stat_share),
            "pred_share_leaky": select(c.expected_dua(P_leak), i_stat_share),
            # leakage ablation (non-causal): v1 inputs, same km objective
            "pred_sched_leaky": select(c.expected_metric(P_leak), i_stat),
            "reactive_leaky": select(c.current_metric(A[i0]), i_stat),
            "oracle": DUA.argmin(1),
        }
        rows = np.arange(len(hours))
        base_d = DUA[:, i_stat]
        res = {}
        for name, ch in choices.items():
            d, a = DUA[rows, ch], AES[rows, ch]
            e = {"mean_KUA": float(d.mean()), "mean_share": float(SH[rows, ch].mean()),
                 "mean_hours_ua": float(HU[rows, ch].mean()),
                 "impr_hours_ua_vs_static_pretest_pct": C.impr(HU[:, i_stat], HU[rows, ch]),
                 "impr_share_vs_static_pretest_pct": C.impr(SH[:, i_stat], SH[rows, ch]),
                 "impr_share_vs_static_share_pct": C.impr(SH[:, i_stat_share], SH[rows, ch]),
                 "mean_AES8": float(a.mean()),
                 "impr_KUA_vs_static_pretest_pct": C.impr(base_d, d),
                 "impr_KUA_vs_static_hindsight_pct": C.impr(DUA[:, i_hind], d),
                 "impr_AES8_vs_static_pretest_aes_pct": C.impr(AES[:, i_stat_aes], a),
                 "mean_D_km": float(c.D[ch].mean()), "mean_T_h": float(c.T[ch].mean()),
                 "dT_vs_static_min": float(60 * (c.T[ch].mean() - c.T[i_stat])),
                 "dD_vs_static_km": float(c.D[ch].mean() - c.D[i_stat]),
                 "pct_hours_off_static": float(100 * np.mean(ch != i_stat)),
                 "churn": int(np.sum(ch[1:] != ch[:-1]))}
            if name not in ("static_pretest",):
                diff = base_d - d
                if np.any(diff != 0):
                    e["wilcoxon_p_vs_static"] = float(wilcoxon(base_d, d).pvalue)
                e["block_boot_ci_impr_pct"] = C.block_bootstrap_impr(base_d, d)
            res[name] = e
            if name in ("pred_sched", "pred_h6"):
                pvals.append((key, name, e.get("wilcoxon_p_vs_static", 1.0)))

        # (b) cost of safety: epsilon-constraint and Lagrangian front ------
        E = c.expected_metric(P)
        Tmin = c.T.min()
        eps_rows, lam_rows = [], []
        for eps in EPS:
            allowed = np.ones(R, bool) if eps is None else (c.T <= Tmin * (1 + eps) + 1e-9)
            s = np.where(allowed[None, :], E, np.inf)
            pref = i_stat if allowed[i_stat] else int(np.where(allowed)[0][np.argmin(DUA_pre.mean(0)[allowed])])
            ch = select(s, pref)
            base_eps = DUA[:, pref]
            d = DUA[rows, ch]
            eps_rows.append({"eps": "inf" if eps is None else eps, "n_allowed": int(allowed.sum()),
                             "static_in_set": c.ids[pref], "static_KUA": float(base_eps.mean()),
                             "pred_KUA": float(d.mean()), "impr_vs_same_set_static_pct": C.impr(base_eps, d),
                             "mean_T_h": float(c.T[ch].mean()), "mean_D_km": float(c.D[ch].mean())})
        for lam in LAMBDAS:
            # km normalised by the static route length keeps lambda on the scale of a share
            ch = select(E / c.D[i_stat] + lam * (c.T / Tmin - 1)[None, :], i_stat)
            d = DUA[rows, ch]
            lam_rows.append({"lambda": lam, "KUA": float(d.mean()), "share": float(SH[rows, ch].mean()),
                             "T_h": float(c.T[ch].mean()),
                             "D_km": float(c.D[ch].mean())})
        SH_pre = c.dua(pre)
        fixed = [{"route": c.ids[r], "KUA": float(DUA[:, r].mean()), "share": float(SH[:, r].mean()),
                  "AES8": float(AES[:, r].mean()),
                  "KUA_pretest": float(DUA_pre[:, r].mean()), "share_pretest": float(SH_pre[:, r].mean()), "T_h": float(c.T[r]), "D_km": float(c.D[r])}
                 for r in range(R)]
        pd.DataFrame(lam_rows).assign(kind="dynamic_lambda").to_csv(C.P3 / f"pareto_{key}.csv", index=False)
        pd.DataFrame(fixed).to_csv(C.P3 / f"fixed_routes_{key}.csv", index=False)

        # temporal stability (monthly) for pred_sched
        m = hours.to_period("M").astype(str)
        dsel = DUA[rows, choices["pred_sched"]]
        monthly = {mm: C.impr(base_d[m == mm], dsel[m == mm]) for mm in sorted(set(m))}

        summary["corridors"][key] = {
            "routes": c.ids, "duplicates_removed": c.duplicates, "oblasts": c.obl,
            "static_pretest": c.ids[i_stat], "static_hindsight": c.ids[i_hind],
            "static_pretest_aes": c.ids[i_stat_aes], "static_hindsight_aes": c.ids[i_hind_aes],
            "fastest": c.ids[i_fast], "shortest": c.ids[i_short],
            "strategies": res, "epsilon": eps_rows, "lambda": lam_rows, "fixed_routes": fixed,
            "monthly_impr_pred_sched_pct": monthly,
            "share_hours_static_zero_KUA": float(np.mean(base_d == 0)),
            "static_pretest_by_share": c.ids[int(SH_pre.mean(0).argmin())],
        }
        print(f"\n=== {key}: R={R} dup={c.duplicates} static_pre={c.ids[i_stat]} hind={c.ids[i_hind]}")
        for n, e in res.items():
            print(f"  {n:18s} KUA={e['mean_KUA']:.2f}km impr={e['impr_KUA_vs_static_pretest_pct']:+6.2f}% "
                  f"share={e['impr_share_vs_static_pretest_pct']:+6.2f}% "
                  f"(vs hind {e['impr_KUA_vs_static_hindsight_pct']:+6.2f}%) AES8={e['mean_AES8']:.4f} "
                  f"dT={e['dT_vs_static_min']:+6.1f}min dD={e['dD_vs_static_km']:+6.1f}km p={e.get('wilcoxon_p_vs_static', float('nan')):.1e} "
                  f"CI={e.get('block_boot_ci_impr_pct')}")
    # Holm across corridors x {pred_h6, pred_sched}
    order = sorted(range(len(pvals)), key=lambda i: pvals[i][2])
    m = len(pvals)
    holm, run = {}, 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * pvals[i][2]))
        holm[f"{pvals[i][0]}:{pvals[i][1]}"] = run
    summary["holm_adjusted_p"] = holm
    (C.P3 / "eval_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
