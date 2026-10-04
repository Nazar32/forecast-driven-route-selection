"""
Paper 3, supplementary tables generated from the results folder (C.P3):
supp_monthly.tex, supp_subgroups.tex, supp_ablation.tex, supp_metric.tex, supp_learners.tex,
supp_ensemble.tex, supp_sens.tex (if sens/ exists) and the numeric columns of the candidate-route table
(supp_cands.json). Written into paper/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import p3_core as C  # noqa: E402

OUT = C.PAPER
CO = ["lviv_kyiv", "odesa_kyiv", "kharkiv_lviv", "dnipro_kyiv"]
NAME = {"lviv_kyiv": "Lviv--Kyiv", "odesa_kyiv": "Odesa--Kyiv", "kharkiv_lviv": "Kharkiv--Lviv",
        "dnipro_kyiv": "Dnipro--Kyiv"}
J = lambda f: json.loads((C.P3 / f).read_text())


def num(x, d=1):
    s = f"{x:+.{d}f}"
    return "0.0" if s in ("+0.0", "-0.0") else s


def cell(v, ci=None, same=False):
    if same:
        return "= static"
    m = ""
    if ci is not None:
        if ci[0] <= 0 <= ci[1]:
            m = " n.s."
        elif ci[1] < 0:
            m = "$^{*}$"
    return f"${num(v)}${m}"


def table(lines):
    return "\n".join(lines) + "\n"


def monthly(s):
    months = sorted({m for k in CO for m in s["corridors"][k]["monthly_impr_pred_sched_pct"]})
    hrs = pd.date_range(C.TEST_START, C.TEST_END - pd.Timedelta(hours=C.K_MAX), freq="h", tz="UTC")
    cnt = hrs.tz_convert(None).to_period("M").astype(str).value_counts()
    months = [m for m in months if cnt.get(m, 0) >= 72]          # drop boundary months with < 3 days
    lab = [pd.Period(m).strftime("%b") for m in months]
    L = [r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{@{}l" + "r" * len(months) + "@{}}", r"\toprule",
         r"\textbf{Corridor} & " + " & ".join(rf"\textbf{{{x}}}" for x in lab) + r" \\", r"\midrule"]
    for k in CO:
        mm = s["corridors"][k]["monthly_impr_pred_sched_pct"]
        L.append(NAME[k] + " & " + " & ".join(f"${num(mm[m])}$" for m in months) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "supp_monthly.tex").write_text(table(L))


def subgroups(x):
    L = [r"\setlength{\tabcolsep}{2.4pt}", r"\begin{tabular}{@{}lccc|ccc@{}}", r"\toprule",
         r" & \multicolumn{3}{c|}{\textbf{Hours (\%)}} & \multicolumn{3}{c}{\textbf{Reduction (\%)}} \\",
         r"\textbf{Corridor} & \textbf{better} & \textbf{equal} & \textbf{worse} & \textbf{exposed} & \textbf{high} & \textbf{low} \\",
         r"\midrule"]
    for k in CO:
        c = x["corridors"][k]
        w, sg = c["wtl"], c["subgroup"]
        L.append(f"{NAME[k]} & {100 * w['better']:.0f} & {100 * w['equal']:.0f} & {100 * w['worse']:.0f} & "
                 + " & ".join(cell(sg[g]["impr"], sg[g]["ci"]) for g in ("static_exposed", "national_high", "national_low"))
                 + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "supp_subgroups.tex").write_text(table(L))


def ablation(s):
    st = lambda k, n: s["corridors"][k]["strategies"][n]
    rows = [("Main (causal inputs, km objective)", "pred_sched", "impr_KUA_vs_static_pretest_pct"),
            ("Look-ahead inputs, km objective", "pred_sched_leaky", "impr_KUA_vs_static_pretest_pct"),
            ("Share objective", "pred_share", "impr_KUA_vs_static_pretest_pct"),
            (r"\quad its static baseline", "static_share", "impr_KUA_vs_static_pretest_pct"),
            None,
            ("Share objective", "pred_share", "impr_share_vs_static_share_pct"),
            ("Share objective + look-ahead inputs", "pred_share_leaky", "impr_share_vs_static_share_pct")]
    L = [r"\setlength{\tabcolsep}{2.6pt}", r"\renewcommand{\arraystretch}{1.18}", r"\begin{tabular}{@{}lrrrr@{}}",
         r"\toprule", r" & \textbf{L--K} & \textbf{O--K} & \textbf{Kh--L} & \textbf{D--K} \\", r"\midrule",
         r"\multicolumn{5}{@{}l}{\emph{Scored in km under alert}} \\"]
    for r in rows:
        if r is None:
            L += [r"\midrule", r"\multicolumn{5}{@{}l}{\emph{Scored as share of the route}} \\"]
            continue
        L.append(r[0] + " & " + " & ".join(f"${num(st(k, r[1])[r[2]])}$" for k in CO) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "supp_ablation.tex").write_text(table(L))


def metric(s):
    names = [("Reactive", "reactive"), ("Calendar", "climatology"), ("Single-horizon", "pred_h6"),
             ("Schedule-aware", "pred_sched"), (r"Oracle$^\dagger$", "oracle")]
    L = [r"\setlength{\tabcolsep}{1.1pt}", r"\begin{tabular}{@{}lcccccccc@{}}", r"\toprule",
         r" & \multicolumn{2}{c}{\textbf{L--K}} & \multicolumn{2}{c}{\textbf{O--K}} &",
         r"   \multicolumn{2}{c}{\textbf{Kh--L}} & \multicolumn{2}{c}{\textbf{D--K}} \\",
         r"\textbf{Strategy} & KUA & AES$_W$ & KUA & AES$_W$ & KUA & AES$_W$ & KUA & AES$_W$ \\", r"\midrule"]
    for lab, n in names:
        cs = []
        for k in CO:
            e = s["corridors"][k]["strategies"][n]
            cs += [f"${num(e['impr_KUA_vs_static_pretest_pct'])}$", f"${num(e['impr_AES8_vs_static_pretest_aes_pct'])}$"]
        L.append(lab + " & " + " & ".join(cs) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "supp_metric.tex").write_text(table(L))


def learners(m):
    lab = {"hgb": "Gradient boosting (main)", "rf": "Random forest", "logreg": "Logistic regression"}
    L = [r"\setlength{\tabcolsep}{2.4pt}", r"\begin{tabular}{@{}lcccccccc@{}}", r"\toprule",
         r"\textbf{Learner} & \textbf{AUC $j{=}0$} & \textbf{AUC $j{=}6$} & \textbf{L--K} & \textbf{O--K} & \textbf{Kh--L} & \textbf{D--K} & \textbf{OSM} & \textbf{better} \\",
         r"\midrule"]
    for k in ("hgb", "rf", "logreg"):
        v = m[k]
        L.append(f"{lab[k]} & {v['auc_k1']:.3f} & {v['auc_k7']:.3f} & "
                 + " & ".join(cell(v["corridors"][c]["pred_impr_pct"], v["corridors"][c]["ci"]) for c in CO)
                 + f" & ${num(v['osm']['pred_pooled_pct'])}$ & {100 * v['osm']['share_od_better']:.0f}\\% " + r"\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "supp_learners.tex").write_text(table(L))


def ensemble(e):
    names = [("predictive", "Direct forecast (paper)"), ("markov", "Markov persistence"),
             ("calendar", "Calendar baseline"), ("ens_mean3", "Mean of three forecasts"),
             ("ens_mean2", "Mean of direct and Markov"), ("ens_vote", "Majority vote of choices")]
    L = [r"\renewcommand{\arraystretch}{1.12}", r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
         r"\textbf{Rule} & \textbf{L--K} & \textbf{O--K} & \textbf{Kh--L} & \textbf{D--K} & \textbf{OSM prim.} & \textbf{better} \\",
         r"\midrule"]
    for n, lab in names:
        cs = []
        for k in CO:
            x = e["corridors"][k][n]
            same = x["ci"] == [0.0, 0.0] and abs(x["impr"]) < 1e-9
            cs.append(cell(x["impr"], x["ci"], same))
        o = e["osm_primary"][n]
        L.append(lab + " & " + " & ".join(cs) + f" & ${num(o['pooled_pct'])}$ & {100 * o['share_better']:.0f}\\% " + r"\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "supp_ensemble.tex").write_text(table(L))


def sens():
    d = C.P3 / "sens"
    if not d.exists():
        return
    rows = [("Base", "base"), None, (r"$\theta=0.3$", "theta0.3"), (r"$\theta=0.5$", "theta0.5"),
            (r"$\theta=1.0$ (none)", "theta1.0"), None, ("stretch 1.1", "stretch1.1"), ("stretch 1.2", "stretch1.2"),
            ("stretch 1.3", "stretch1.3"), ("stretch 2.0", "stretch2.0"), None, ("factor 1.05", "factor1.05"),
            ("factor 1.1", "factor1.1"), ("factor 1.5", "factor1.5"), ("factor 2.0", "factor2.0")]
    L = [r"\setlength{\tabcolsep}{2.2pt}", r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
         r"\textbf{Setting} & \textbf{Cand.} & \textbf{Overl.} & \textbf{Red.} & \textbf{Better} & \textbf{Oracle} & \textbf{Time (h)} \\",
         r"\midrule"]
    out = {}
    for r in rows:
        if r is None:
            L.append(r"\midrule")
            continue
        f = d / f"roadgraph_osm_primary_{r[1]}.json"
        if not f.exists():
            continue
        s = json.loads(f.read_text())
        out[r[1]] = s
        k = s["pred_K10"]
        L.append(f"{r[0]} & {s['mean_n_cand']:.1f} & {s['overlap10']:.2f} & {k['pooled_pct']:.1f} & "
                 f"{100 * k['share_better']:.0f}\\% & {s['oracle_K10']['pooled_pct']:.1f} & {k['T_h']:.2f} / {s['base10_T']:.2f} " + r"\\")
    f = d / "roadgraph_osm_primary_base.json"
    if f.exists():
        s = json.loads(f.read_text())
        if "pred_K20" in s:
            k = s["pred_K20"]
            L += [r"\midrule", f"Base, $K=20$ & {s['mean_n_cand']:.1f} & {s['overlap10']:.2f} & {k['pooled_pct']:.1f} & "
                  f"{100 * k['share_better']:.0f}\\% & {s['oracle_K20']['pooled_pct']:.1f} & {k['T_h']:.2f} / {s['base10_T']:.2f} " + r"\\"]
    L += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "supp_sens.tex").write_text(table(L))


def cands(s):
    res = {}
    for k in CO:
        c = s["corridors"][k]
        for f in c["fixed_routes"]:
            rid = f["route"].split("_")[-1]
            res[f"{k}:{rid}"] = {"T": f["T_h"], "D": f["D_km"], "pre": f["KUA_pretest"], "test": f["KUA"],
                                 "dua": 100 * f["share"]}
        res[f"{k}:static"] = c["static_pretest"].split("_")[-1]
        res[f"{k}:fastest"] = c["fastest"].split("_")[-1]
    (OUT / "supp_cands.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    # paper tables: P3_RESULTS=p3v7 (monthly, metric, cands, subgroups, sens); the ablation table comes from
    # the unanchored pipeline: P3_RESULTS=p3v5 python p3_supp_tables.py ablation
    which = sys.argv[1:] or ["monthly", "metric", "cands", "subgroups", "sens"]
    s = J("eval_summary.json")
    for w in which:
        if w == "subgroups":
            subgroups(J("extra_summary.json"))
        elif w == "sens":
            sens()
        else:
            globals()[w](s)
    print("written to", OUT)
