"""Supplementary per-OD table for OSM primary (K = 10): writes paper/tab_perod.tex"""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
import p3_core as C  # noqa: E402

g = pd.read_csv(C.P3 / "graph_eval_per_od_osm_primary.csv")
m = pd.read_csv(C.P3 / "review_extra_per_od_osm_primary.csv")
d = g.merge(m[["od", "markov_KUA", "static_KUA"]], on="od", suffixes=("", "_m"))
assert (abs(d.static10_KUA - d.static_KUA) < 1e-6).all()
b = d.static10_KUA
red = lambda x: 100 * (b - x) / b
d["pred"], d["react"], d["mk"], d["orc"] = red(d.pred_sched_K10_KUA), red(d.reactive_K10_KUA), red(d.markov_KUA), red(d.oracle_K10_KUA)
import os
MKR = os.environ.get("P3_MKROLL")                      # v4: rolling-Markov results folder (same pipeline)
if MKR:
    r_ = pd.read_csv(C.RES / MKR / "graph_eval_per_od_osm_primary.csv")[["od", "pred_sched_K10_KUA", "static10_KUA"]]
    d = d.merge(r_, on="od", suffixes=("", "_r"))
    assert (abs(d.static10_KUA - d.static10_KUA_r) < 1e-6).all()
    d["mk"] = red(d.pred_sched_K10_KUA_r)
d["dyn"] = 100 * (d["dyn_mu64.0_static_KUA"] - d["dyn_mu64.0_KUA"]) / d["dyn_mu64.0_static_KUA"]
d = d.sort_values("od").reset_index(drop=True)

def row(r):
    s = "$^{\\ast}$" if (r.pred_K10_p < 0.05 and r.pred > 0) else ("$^{-}$" if (r.pred_K10_p < 0.05 and r.pred < 0) else "")
    od = r.od.replace("-", "--")
    return (f"{od} & {r.T_fast_h:.1f} & {r.static10_KUA:.1f} & {r.pred:+.1f}{s} & {r.mk:+.1f} & {r.react:+.1f} & "
            f"{r.dyn:+.1f} & {r.orc:.1f} \\\\")

head = ("\\begin{tabular}{@{}lrrrrrrr@{}}\\toprule\n"
        "\\textbf{OD pair} & $T$ & static & pred. & " + ("roll.\\ Mk." if MKR else "Markov") + " & react. & Dijk. & oracle \\\\\\midrule\n")
half = (len(d) + 1) // 2
parts = [d.iloc[:half], d.iloc[half:]]
out = []
for p in parts:
    out.append(head + "\n".join(row(r) for r in p.itertuples()) + "\n\\bottomrule\\end{tabular}")
dst = C.PAPER / "tab_perod.tex"
dst.write_text("{\\scriptsize\\setlength{\\tabcolsep}{2.2pt}\n" + "\\hfill".join(out) + "}\n")
print("wrote", dst, len(d), "pooled pred", 100 * (b.sum() - d.pred_sched_K10_KUA.sum()) / b.sum(),
      "pos", (d.pred > 0).mean(), "sig+", ((d.pred_K10_p < .05) & (d.pred > 0)).mean(), "sig-", ((d.pred_K10_p < .05) & (d.pred < 0)).mean())
