"""
Paper 3, supplement: forecast quality by lead for both hazards, computed identically
(decision at t:00, model k = j+1 on row t-1, lead 0 = max(active at t:00, g_1); test decision hours).
Metrics per lead: AUC (direct models, Markov persistence), Brier score, Brier skill score vs. base rate,
expected calibration error (10 bins), reliability bins for leads 0, 6, 18.
Output: results/p3v2/transfer/lead_quality.json, supplementary table tab_lead.tex, figure fig_transfer_lead.pdf
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
import p3_transfer as T  # noqa: E402
from p3_eval import load_direct  # noqa: E402

LEADS = list(range(C.K_MAX + 1))
PAPER = C.PAPER


def markov(pres_vals, prev, act0, pre_mask):
    a = pres_vals[pre_mask]
    a0, a1 = a[:-1], a[1:]
    p11 = (a0 * a1).sum(0) / np.maximum(a0.sum(0), 1); p01 = ((1 - a0) * a1).sum(0) / np.maximum((1 - a0).sum(0), 1)
    pi = p01 / np.maximum(1 - p11 + p01, 1e-9); lam = p11 - p01
    j = np.arange(C.K_MAX + 1)
    P = pi[None, None] + (prev[:, None, :] - pi[None, None]) * lam[None, None] ** (j[None, :, None] + 1)
    P[:, 0, :] = np.maximum(P[:, 0, :], act0)
    return P


def metrics(P, PM, Y):
    out = {}
    for j in LEADS:
        p, y, pm = P[:, j].ravel(), Y[:, j].ravel(), PM[:, j].ravel()
        base = y.mean(); br = float(np.mean((p - y) ** 2))
        bins = np.minimum((p * 10).astype(int), 9)
        rel = [{"bin": int(b), "p": float(p[bins == b].mean()), "y": float(y[bins == b].mean()), "n": int((bins == b).sum())}
               for b in range(10) if (bins == b).sum() > 0]
        out[j] = {"auc": float(roc_auc_score(y, p)), "auc_markov": float(roc_auc_score(y, pm)), "brier": br,
                  "brier_markov": float(np.mean((pm - y) ** 2)), "bss": float(1 - br / (base * (1 - base))),
                  "ece": float(sum(r["n"] * abs(r["p"] - r["y"]) for r in rel) / len(p)), "base": float(base),
                  "rel": rel}
    return out


def ukraine():
    act = C.presence()
    hours = pd.date_range(C.TEST_START, C.TEST_END - pd.Timedelta(hours=C.K_MAX), freq="h", tz="UTC")
    obl = list(act.columns)
    P = load_direct(hours, obl)
    pos = pd.Series(np.arange(len(act)), index=act.index); ii = pos[hours].values
    Y = np.stack([act.values[ii + j] for j in LEADS], axis=1)
    prev = act.reindex(hours - pd.Timedelta(hours=1)).values
    pre = (act.index >= C.DATA_START) & (act.index < C.TEST_START)
    PM = markov(act.values.astype(float), prev, C.active_at(hours, obl), pre)
    return metrics(P, PM, Y)


def usa():
    pres = T.presence(); reg = list(pres.columns)
    t_test = pd.Timestamp(json.loads((T.OUT / "split.json").read_text())["t_test"])
    hours = pd.date_range(t_test, T.END - pd.Timedelta(hours=C.K_MAX), freq="h", tz="UTC")
    act0 = T.active_at(hours, reg)
    P = C.causal_P(C.read_probs(T.OUT / "direct_probs.parquet"), hours, reg, act0)
    pos = pd.Series(np.arange(len(pres)), index=pres.index); ii = pos[hours].values
    Y = np.stack([pres.values[ii + j] for j in LEADS], axis=1)
    prev = pres.reindex(hours - pd.Timedelta(hours=1)).values
    pre = (pres.index >= T.START) & (pres.index < t_test)
    PM = markov(pres.values.astype(float), prev, act0, pre)
    return metrics(P, PM, Y)


def main():
    res = {"ukraine": ukraine(), "usa": usa()}
    (T.OUT / "lead_quality.json").write_text(json.dumps(res, indent=1))
    rows = []
    for j in (0, 1, 2, 3, 6, 9, 12, 15, 18):
        u, s = res["ukraine"][j], res["usa"][j]
        rows.append(f"{j} & {u['auc']:.3f} & {u['auc_markov']:.3f} & {u['bss']:.2f} & {u['ece']:.3f} & "
                    f"{s['auc']:.3f} & {s['auc_markov']:.3f} & {s['bss']:.2f} & {s['ece']:.3f} \\\\")
    tex = ("\\begin{tabular}{@{}rcccccccc@{}}\\toprule\n"
           " & \\multicolumn{4}{c}{\\textbf{Air alerts, Ukraine}} & \\multicolumn{4}{c}{\\textbf{Severe weather, US}} \\\\\n"
           "\\cmidrule(lr){2-5}\\cmidrule(l){6-9}\n"
           "$j$ (h) & AUC & AUC$_{\\mathrm{M}}$ & BSS & ECE & AUC & AUC$_{\\mathrm{M}}$ & BSS & ECE \\\\\\midrule\n"
           + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\n")
    (PAPER / "tab_lead.tex").write_text(tex)
    # figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import p3_paper_figs as PF
    fig, ax = plt.subplots(1, 3, figsize=(PF.TXT, 2.0))
    for k, (name, col) in enumerate((("ukraine", PF.BLUE), ("usa", PF.ORANGE))):
        r = res[name]
        lab = "air alerts, UA" if name == "ukraine" else "severe weather, US"
        ax[0].plot(LEADS, [r[j]["auc"] for j in LEADS], color=col, lw=1.4, label=f"{lab}, direct")
        ax[0].plot(LEADS, [r[j]["auc_markov"] for j in LEADS], color=col, lw=1.0, ls="--", label=f"{lab}, Markov")
        ax[1].plot(LEADS, [r[j]["bss"] for j in LEADS], color=col, lw=1.4, label=lab)
    ax[0].set_xlabel("lead $j$ (h)"); ax[0].set_ylabel("AUC"); ax[0].set_title("(a) discrimination", pad=3)
    ax[1].set_xlabel("lead $j$ (h)"); ax[1].set_ylabel("Brier skill score"); ax[1].set_title("(b) skill vs. base rate", pad=3)
    ax[0].legend(fontsize=5.5, frameon=False); ax[1].legend(fontsize=5.5, frameon=False)
    for j, c in zip((0, 6, 18), (PF.INK2, PF.AQUA, PF.ORANGE)):
        rel = [b for b in res["usa"][j]["rel"] if b["n"] >= 200]
        ax[2].plot([b["p"] for b in rel], [b["y"] for b in rel], marker="o", ms=2.5, color=c, lw=1, label=f"$j={j}$")
    ax[2].plot([0, 1], [0, 1], color=PF.MUTED, lw=0.7, ls=":")
    ax[2].set_xlabel("forecast probability"); ax[2].set_ylabel("observed frequency")
    ax[2].set_title("(c) reliability, US", pad=3); ax[2].legend(fontsize=5.5, frameon=False)
    fig.tight_layout(w_pad=0.6)
    fig.savefig(PAPER / "images" / "fig_transfer_lead.pdf"); fig.savefig(T.OUT / "fig_transfer_lead.png", dpi=200)
    for j in (0, 6, 12, 18):
        print(j, {n: {k: round(res[n][j][k], 3) for k in ("auc", "auc_markov", "bss", "ece", "base")} for n in res})


if __name__ == "__main__":
    main()
