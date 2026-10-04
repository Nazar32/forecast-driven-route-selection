"""Publication figures for Paper 3 (IEEE Access). Vector PDF, Times-like font, 8 pt.
Writes paper/images/fig_*.pdf from results/p3/*.json|csv."""
import json, os, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon as MplPoly

sys.path.insert(0, str(Path(__file__).parent))
import p3_core as C  # noqa

OUT = C.PAPER / "images"
P3 = C.P3
for f in ["texgyretermes-regular.otf", "texgyretermes-bold.otf", "texgyretermes-italic.otf"]:
    for d in ["/usr/share/texmf/fonts/opentype/public/tex-gyre/"]:
        if Path(d + f).exists():
            fm.fontManager.addfont(d + f)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
plt.rcParams.update({
    "font.family": "TeX Gyre Termes", "mathtext.fontset": "stix", "font.size": 8,
    "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.05})
COL, TXT = 3.45, 7.1
NAMES = {"lviv_kyiv": "Lviv–Kyiv", "odesa_kyiv": "Odesa–Kyiv", "kharkiv_lviv": "Kharkiv–Lviv",
         "dnipro_kyiv": "Dnipro–Kyiv"}
J = lambda f: json.loads((P3 / f).read_text())


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(P3 / f"{name}.png", dpi=220)
    plt.close(fig)
    print("wrote", name)


# ------------------------------------------------------------------ Fig. map
def fig_map():
    import p3_geo as G
    from shapely.geometry import MultiPolygon
    act = C.presence()
    test = act[(act.index >= C.TEST_START) & (act.index <= C.TEST_END)]
    share = test.mean()
    meta = json.loads((C.P3_IN / "roadgraph_osm_primary" / "meta.json").read_text())
    excl = set(meta["excluded_oblasts"])
    nodes = pd.read_csv(C.P3_IN / "roadgraph_osm_primary" / "nodes.csv").set_index("node_id")
    edges = pd.read_csv(C.P3_IN / "roadgraph_osm_primary" / "edges.csv", usecols=["u", "v"])
    fig, ax = plt.subplots(figsize=(COL, 2.45))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "b", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"])
    vmax = 0.8
    for name, geom, _ in G.oblast_polygons():
        polys = geom.geoms if isinstance(geom, MultiPolygon) else [geom]
        for p in polys:
            xy = np.asarray(p.exterior.coords)
            if name in excl or name not in share.index:
                ax.add_patch(MplPoly(xy, closed=True, facecolor="#f0efec", edgecolor="white", lw=0.4,
                                     hatch="////", zorder=1))
            else:
                ax.add_patch(MplPoly(xy, closed=True, facecolor=cmap(min(float(share[name]) / vmax, 1.0)),
                                     edgecolor="white", lw=0.4, zorder=1))
    xy = nodes[["lon", "lat"]]
    seg = np.stack([xy.loc[edges.u].values, xy.loc[edges.v].values], axis=1)
    ax.add_collection(LineCollection(seg, colors=INK, linewidths=0.18, alpha=0.55, zorder=2))
    cities = meta["cities"]
    from p3_graph_eval import CENTERS
    cx = nodes.loc[[cities[c] for c in CENTERS]]
    ax.scatter(cx.lon, cx.lat, s=7, color=ORANGE, edgecolor="white", linewidth=0.4, zorder=3)
    for n in ["Lviv", "Kyiv", "Odesa", "Kharkiv", "Dnipro"]:
        r = nodes.loc[cities[n]]
        ax.annotate(n, (r.lon, r.lat), xytext=(3, 2), textcoords="offset points", fontsize=6.5, color=INK,
                    zorder=4)
    ax.set_xlim(22.0, 40.4); ax.set_ylim(44.3, 52.5); ax.set_aspect(1 / np.cos(np.radians(48.5)))
    ax.axis("off")
    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.Normalize(0, 100 * vmax))
    cb = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.01)
    cb.set_label("test hours under alert, %", fontsize=7); cb.ax.tick_params(labelsize=6.5)
    cb.outline.set_linewidth(0.4)
    save(fig, "fig_map")
    print("alert share test range:", float(share[[o for o in share.index if o not in excl]].min()),
          float(share[[o for o in share.index if o not in excl]].max()))


# ------------------------------------------------------------------ Fig. Pareto (corridors)
def fig_pareto():
    s = J("eval_summary.json")["corridors"]
    t = J("tradeoff_summary.json")
    fig, axs = plt.subplots(1, 4, figsize=(TXT, 1.95))
    for i, (ax, k) in enumerate(zip(axs, NAMES)):
        c = s[k]
        fx = pd.DataFrame(c["fixed_routes"])
        ax.scatter(fx.T_h, fx.KUA, s=14, color=MUTED, edgecolor="white", lw=0.4, zorder=3,
                   label="fixed candidate routes")
        st = fx[fx.route == c["static_pretest"]].iloc[0]
        ax.scatter([st.T_h], [st.KUA], s=34, facecolor="none", edgecolor=INK, lw=0.9, marker="s",
                   zorder=4, label="static baseline (pre-test best)")
        lam = pd.DataFrame(c["lambda"]).sort_values("T_h")
        ax.plot(lam.T_h, lam.KUA, color=BLUE, lw=1.3, zorder=5, label="predictive, $\\lambda$-penalised front")
        ps = c["strategies"]["pred_sched"]
        ax.scatter([ps["mean_T_h"]], [ps["mean_KUA"]], s=22, color=BLUE, edgecolor="white", lw=0.5,
                   zorder=6, label="predictive, min $E$[km under alert]")
        g = [t[k][f"nl_gE{x}"] for x in [0.5, 0.6, 0.7, 0.8, 0.9]]
        ax.plot([x["T_h"] for x in g], [x["KUA"] for x in g], color=AQUA, lw=0.8, ls=(0, (2, 1.5)), zorder=6)
        ax.scatter([x["T_h"] for x in g], [x["KUA"] for x in g], s=13, marker="D", color=AQUA,
                   edgecolor="white", lw=0.4, zorder=7, label="nonlinear trade-off, $\\gamma_E=0.5$–$0.9$")
        g8 = t[k]["nl_gE0.8"]
        ax.scatter([g8["T_h"]], [g8["KUA"]], s=30, marker="D", facecolor="none", edgecolor=INK, lw=0.8,
                   zorder=8, label="recommended point, $\\gamma_E=0.8$")
        r = c["strategies"]["reactive"]
        ax.scatter([r["mean_T_h"]], [r["mean_KUA"]], s=16, marker="^", color=ORANGE, edgecolor="white",
                   lw=0.4, zorder=6, label="reactive (current alerts)")
        ax.set_title(NAMES[k], color=INK, pad=3)
        ax.set_xlabel("mean travel time (h)")
        if i == 0:
            ax.set_ylabel("km under alert per trip")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.2), columnspacing=1.2, handletextpad=0.4)
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig_pareto")


# ------------------------------------------------------------------ Fig. graph candidates
def fig_graph():
    Ks = [1, 2, 3, 5, 10]
    s = J("graph_eval_summary_osm_primary.json")
    fig, axs = plt.subplots(1, 2, figsize=(COL, 1.8), gridspec_kw={"width_ratios": [1.05, 1]})
    ax = axs[0]
    for key, col, mk, lab, ls in [("oracle", MUTED, "s", "oracle", "--"), ("pred_sched", BLUE, "o", "predictive", "-"),
                                  ("reactive", ORANGE, "^", "reactive", "-"), ("static", INK2, "D", "static", "-")]:
        ys = [s[f"{key}_K{K}"]["pooled_pct"] for K in Ks]
        ax.plot(Ks, ys, color=col, marker=mk, ms=3, lw=1.1, ls=ls, label=lab)
        print("K-curve", key, [round(y, 1) for y in ys])
    ax.axhline(0, color=MUTED, lw=0.5)
    ax.set_xscale("log"); ax.set_xticks(Ks); ax.set_xticklabels(Ks); ax.minorticks_off()
    ax.set_xlabel("candidate paths $K$"); ax.set_ylabel("reduction of km under alert (%)")
    ax.set_title("(a) OSM primary, 78 OD pairs", pad=3)
    ax.legend(loc="upper left", handlelength=1.6, frameon=True, facecolor="white", edgecolor="none", framealpha=0.9)
    ax = axs[1]
    for f, col, ls, lab in [("graph_eval_per_od.csv", MUTED, ":", "corridor"),
                            ("graph_eval_per_od_osm_primary.csv", BLUE, "-", "OSM primary"),
                            ("graph_eval_per_od_osm_secondary.csv", AQUA, "--", "OSM secondary")]:
        d = pd.read_csv(P3 / f)
        g = np.sort(100 * (d.static10_KUA - d.pred_sched_K10_KUA) / d.static10_KUA)
        ax.plot(g, np.arange(1, len(g) + 1) / len(g), color=col, lw=1.1, ls=ls, label=lab, drawstyle="steps-post")
        print("ECDF", f, "median", round(float(np.median(g)), 1), "share>0", round(float(np.mean(g > 0)), 3))
    ax.axvline(0, color=MUTED, lw=0.5)
    ax.set_xlabel("per-OD reduction (%)"); ax.set_ylabel("share of OD pairs")
    ax.set_title("(b) $K=10$, by network", pad=3)
    ax.set_xlim(-12, 20); ax.set_xticks([-10, -5, 0, 5, 10, 15, 20])
    ax.legend(loc="lower right", handlelength=1.4, fontsize=6.5, borderaxespad=0.1, frameon=True, facecolor="white",
              edgecolor="none", framealpha=0.9)
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig_graph")


# ------------------------------------------------------------------ Fig. backends
def _scale(graph, backend):
    f = C.P3_IN / (f"scale_{graph}_{backend}.json")
    return json.loads(f.read_text())


def fig_backends():
    graphs = [("roadgraph", "corridor"), ("roadgraph_osm_primary", "OSM primary"),
              ("roadgraph_osm_secondary", "OSM secondary")]
    backs = [("memory", "in-memory", AQUA, "o", "--"), ("postgres", "PostgreSQL + pgRouting", ORANGE, "^", "-"),
             ("neo4j", "Neo4j + GDS", BLUE, "s", "-")]
    rows = []
    for g, gl in graphs:
        for b, *_ in backs:
            d = _scale(g, b)
            s1 = [v["median_ms"] for kk, v in d["S1"].items() if kk.startswith("Q1000")]
            s2 = d["S2"]["Q100"]["per_query_ms"] if "Q100" in d["S2"] else list(d["S2"].values())[-1]["per_query_ms"]
            s3 = d["S3"]["k6"]["per_query_ms"]
            rows.append({"graph": gl, "edges": d["edges"], "backend": b, "S1": min(s1), "S2": s2, "S3": s3})
    R = pd.DataFrame(rows)
    print(R.round(3).to_string())
    R.to_csv(P3 / "fig_backends_data.csv", index=False)
    fig, axs = plt.subplots(1, 3, figsize=(TXT, 1.85))
    titles = {"S1": "(a) fleet cycle: update + 1000 shortest paths", "S2": "(b) candidate generation: Yen, $K=10$",
              "S3": "(c) 6-hop neighbourhood"}
    for ax, sc in zip(axs, ["S1", "S2", "S3"]):
        for b, lab, col, mk, ls in backs:
            q = R[R.backend == b].sort_values("edges")
            ax.plot(range(3), q[sc].values, color=col, marker=mk, ms=3.5, lw=1.1, ls=ls, label=lab)
        ax.set_yscale("log")
        ax.set_xticks(range(3)); ax.set_xticklabels(["corridor\n126 edges", "OSM primary\n17 100", "OSM secondary\n29 114"])
        ax.set_xlim(-0.25, 2.25); ax.grid(axis="x", visible=False)
        ax.set_title(titles[sc], pad=3)
    axs[0].set_ylabel("median latency per call (ms)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig_backends")


# ------------------------------------------------------------------ Fig. rolling origin
def fig_rolling():
    s = J("rolling_summary.json")
    W = [w for w in ["W0", "W1", "W2", "W3c"] if w in s] or ["W1", "W2", "W3", "W4"]
    LAB = {"W0": "W0\n04–09.2023", "W1": "W1\n10.2023–03.2024", "W2": "W2\n04–09.2024",
           "W3c": "W3\n10–12.2024", "W3": "W3\n10.2024–03.2025", "W4": "W4\n04–10.2025"}
    lab = [LAB[w] for w in W]
    fig, axs = plt.subplots(2, 1, figsize=(COL, 3.2), sharex=True)
    ax = axs[0]
    x = np.arange(4)
    p = [s[w]["osm"]["pred_pooled_pct"] for w in W]
    r = [s[w]["osm"]["reactive_pooled_pct"] for w in W]
    ax.axhline(0, color=MUTED, lw=0.5)
    ax.bar(x - 0.19, p, 0.36, color=BLUE, label="predictive")
    ax.bar(x + 0.19, r, 0.36, color=ORANGE, label="reactive")
    for xi, v in zip(x, p):
        ax.annotate(f"{v:.1f}", (xi - 0.19, v), xytext=(0, 1.5), textcoords="offset points", ha="center",
                    fontsize=6, color=INK)
    ax.set_ylabel("reduction, %"); ax.set_title("(a) national OSM network, 78 OD pairs (pooled)", pad=3)
    ax.set_ylim(-1.8, 6.2)
    ax.legend(loc="upper center", ncol=2); ax.grid(axis="x", visible=False)
    ax = axs[1]
    mk = {"lviv_kyiv": "o", "odesa_kyiv": "s", "kharkiv_lviv": "^", "dnipro_kyiv": "D"}
    for j, k in enumerate(NAMES):
        pv = [s[w]["corridors"][k]["pred_sched_impr_pct"] for w in W]
        rv = [s[w]["corridors"][k]["reactive_impr_pct"] for w in W]
        off = (j - 1.5) * 0.13
        ax.scatter(x + off, pv, marker=mk[k], s=12, color=BLUE, edgecolor="white", lw=0.3, zorder=3,
                   label=NAMES[k])
        ax.scatter(x + off, rv, marker=mk[k], s=10, facecolor="none", edgecolor=ORANGE, lw=0.6, zorder=2)
    ax.axhline(0, color=MUTED, lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=6.5)
    ax.set_ylabel("reduction, %")
    ax.set_title("(b) four corridors: filled = predictive, open = reactive", pad=3)
    ax.legend(loc="lower center", ncol=4, handletextpad=0.1, columnspacing=0.8, fontsize=6.5,
              bbox_to_anchor=(0.5, 1.08))
    ax.set_ylim(-10, 15)
    fig.tight_layout(h_pad=1.8)
    save(fig, "fig_rolling")


def fig_rolling4():
    """v4: rolling windows from the C4 runs (results/p3v6/raion_exp/obl_W*), with the rolling Markov chain
    and the unanchored forecast of v3 (results/p3v5/rolling_summary.json)."""
    R = C.RES / "p3v6" / "raion_exp"
    W = ["W0", "W1", "W2", "W3c"]
    E = {w: json.loads((R / f"obl_{w}" / "eval.json").read_text()) for w in W}
    old = json.loads((C.RES / "p3v5" / "rolling_summary.json").read_text())
    lab = ["W0\n04–09.2023", "W1\n10.2023–03.2024", "W2\n04–09.2024", "W3\n10–12.2024"]
    fig, axs = plt.subplots(2, 1, figsize=(COL, 3.2), sharex=True)
    ax = axs[0]
    x = np.arange(4)
    ser = [("predictive (anchored + decision layer)", [E[w]["osm"]["pooled"]["blend50"] for w in W], BLUE),
           ("rolling Markov", [E[w]["osm"]["pooled"]["markov_roll"] for w in W], "#7a7871"),
           ("unanchored forecast", [old[w]["osm"]["pred_pooled_pct"] for w in W], "#9ec5f4"),
           ("reactive", [E[w]["osm"]["pooled"]["reactive"] for w in W], ORANGE)]
    wd = 0.2
    ax.axhline(0, color=MUTED, lw=0.5)
    for i, (n, v, c) in enumerate(ser):
        ax.bar(x + (i - 1.5) * wd, v, wd * 0.95, color=c, label=n)
    for xi, v in zip(x, ser[0][1]):
        ax.annotate(f"{v:.1f}", (xi - 1.5 * wd, v), xytext=(0, 1.5), textcoords="offset points", ha="center",
                    fontsize=6, color=INK)
    ax.set_ylabel("reduction, %"); ax.set_title("(a) national OSM network, 78 OD pairs (pooled)", pad=3)
    ax.set_ylim(-2, 13.5)
    ax.legend(loc="upper left", ncol=2, fontsize=6); ax.grid(axis="x", visible=False)
    ax = axs[1]
    mk = {"lviv_kyiv": "o", "odesa_kyiv": "s", "kharkiv_lviv": "^", "dnipro_kyiv": "D"}
    for j, k in enumerate(NAMES):
        pv = [E[w]["corridors"][k]["blend50"] for w in W]
        rv = [E[w]["corridors"][k]["markov_roll"] for w in W]
        off = (j - 1.5) * 0.13
        ax.scatter(x + off, pv, marker=mk[k], s=12, color=BLUE, edgecolor="white", lw=0.3, zorder=3,
                   label=NAMES[k])
        ax.scatter(x + off, rv, marker=mk[k], s=10, facecolor="none", edgecolor="#7a7871", lw=0.6, zorder=2)
    ax.axhline(0, color=MUTED, lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=6.5)
    ax.set_ylabel("reduction, %")
    ax.set_title("(b) four corridors: filled = predictive, open = rolling Markov", pad=3)
    ax.legend(loc="lower center", ncol=4, handletextpad=0.1, columnspacing=0.8, fontsize=6.5,
              bbox_to_anchor=(0.5, 1.08))
    ax.set_ylim(-2, 11)
    fig.tight_layout(h_pad=1.8)
    save(fig, "fig_rolling")


RED = "#e34948"
SH = {"Дніпропетровська область": "Dnipr.", "Кіровоградська область": "Kirov.",
      "Черкаська область": "Cherkasy", "Вінницька область": "Vinnytsia", "Київська область": "Kyiv obl.",
      "м. Київ": "city", "Полтавська область": "Poltava", "Харківська область": "Khark.",
      "Хмельницька область": "Khmeln.", "Львівська область": "Lviv", "Житомирська область": "Zhyt.",
      "Тернопільська область": "Tern.", "Рівненська область": "Rivne"}
EN = {"Дніпропетровська область": "Dnipropetrovsk", "Кіровоградська область": "Kirovohrad",
      "Черкаська область": "Cherkasy", "Вінницька область": "Vinnytsia", "Київська область": "Kyiv obl.",
      "м. Київ": "Kyiv city", "Полтавська область": "Poltava", "Запорізька область": "Zaporizhzhia",
      "Житомирська область": "Zhytomyr", "Хмельницька область": "Khmelnytskyi",
      "Харківська область": "Kharkiv obl.", "Рівненська область": "Rivne", "Львівська область": "Lviv obl.",
      "Тернопільська область": "Ternopil", "Волинська область": "Volyn", "Сумська область": "Sumy",
      "Чернігівська область": "Chernihiv", "Одеська область": "Odesa", "Миколаївська область": "Mykolaiv",
      "Херсонська область": "Kherson", "Івано-Франківська область": "Ivano-Frankivsk",
      "Чернівецька область": "Chernivtsi", "Закарпатська область": "Zakarpattia", "Донецька область": "Donetsk"}


def fig_lead():
    d = J("extra_summary.json")["lead"]
    ks = sorted(int(k) for k in d)
    fig, axs = plt.subplots(1, 2, figsize=(COL, 1.75), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axs[0]
    ax.plot(ks, [d[str(k)]["auc"] for k in ks], color=BLUE, marker="o", ms=2.5, lw=1.1, label="AUC")
    ax.plot(ks, [d[str(k)]["bss"] for k in ks], color=ORANGE, marker="^", ms=2.5, lw=1.1, label="Brier skill score")
    ax.set_ylim(0, 1); ax.set_xticks([0, 3, 6, 9, 12, 15, 18]); ax.set_xlabel("lead $j$ after departure (h)")
    ax.set_title("(a) quality by lead", pad=3); ax.legend(loc="center right")
    ax = axs[1]
    ax.plot([0, 1], [0, 1], color=MUTED, lw=0.6, ls="--")
    for k, col, mk in [(0, BLUE, "o"), (6, ORANGE, "^"), (18, AQUA, "s")]:
        r = [x for x in d[str(k)]["reliability"] if x["n"] >= 200]
        ax.plot([x["p_mean"] for x in r], [x["y_mean"] for x in r], color=col, marker=mk, ms=2.5, lw=1.0, label=f"$j={k}$")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xlabel("forecast probability"); ax.set_ylabel("observed frequency")
    ax.set_title("(b) reliability", pad=3); ax.legend(loc="upper left")
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig_lead")


def fig_depart():
    d = J("extra_summary.json")["corridors"]
    fig, axs = plt.subplots(1, 4, figsize=(TXT, 1.75))
    for i, (ax, k) in enumerate(zip(axs, NAMES)):
        r = d[k]["depart"]; h = [x["h"] for x in r]
        ax.plot(h, [x["static"] for x in r], color=INK2, lw=1.1, label="static baseline")
        ax.plot(h, [x["pred"] for x in r], color=BLUE, lw=1.3, label="schedule-aware predictive")
        ax.plot(h, [x["oracle"] for x in r], color=MUTED, lw=0.9, ls="--", label="oracle")
        ax.set_xticks([0, 6, 12, 18, 23]); ax.set_xlim(0, 23); ax.set_ylim(bottom=0)
        ax.set_title(NAMES[k], pad=3); ax.set_xlabel("departure hour (local)")
        if i == 0: ax.set_ylabel("km under alert per trip")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig_depart")


def fig_netfront():
    fig, axs = plt.subplots(1, 2, figsize=(COL, 1.9), sharey=True)
    for ax, (f, t) in zip(axs, [("graph_eval_summary_osm_primary.json", "(a) OSM primary"),
                                 ("graph_eval_summary_osm_secondary.json", "(b) OSM secondary")]):
        s = J(f)
        lam = sorted([(s[k]["mean_T_h"], s[k]["pooled_pct"]) for k in s if k.startswith("cand_lam")])
        dyn = sorted([(s[k]["mean_T_h"], s[k]["pooled_pct"]) for k in s if k.startswith("dyn_mu")])
        ax.plot(*zip(*lam), color=BLUE, marker="o", ms=2.5, lw=1.2, label="candidates, $\\lambda$ sweep")
        ax.plot(*zip(*dyn), color=ORANGE, marker="^", ms=2.5, lw=1.2, label="risk-weighted Dijkstra, $\\mu$ sweep")
        ax.scatter([s["static_K10"]["mean_T_h"]], [0], marker="s", s=22, facecolor="none", edgecolor=INK, lw=0.8,
                   zorder=5, label="static baseline")
        ax.scatter([s["static_K1"]["mean_T_h"]], [s["static_K1"]["pooled_pct"]], marker="v", s=18, color=INK2, zorder=5,
                   label="fastest path")
        ax.axhline(0, color=MUTED, lw=0.5)
        ax.set_title(t, pad=3); ax.set_xlabel("mean trip time (h)")
        print(f, "lam", [(round(a, 2), round(b, 1)) for a, b in lam], "dyn", [(round(a, 2), round(b, 1)) for a, b in dyn])
    axs[0].set_ylabel("reduction of km under alert (%)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.2), columnspacing=0.8)
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig_netfront")


def fig_perod():
    from scipy.stats import spearmanr
    d = pd.read_csv(P3 / "graph_eval_per_od_osm_primary.csv")
    g = 100 * (d.static10_KUA - d.pred_sched_K10_KUA) / d.static10_KUA
    o = 100 * (d.static10_KUA - d.oracle_K10_KUA) / d.static10_KUA
    fig, ax = plt.subplots(figsize=(COL, 2.1))
    sc = ax.scatter(o, g, c=d.T_fast_h, cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
        "b", ["#9ec5f4", "#3987e5", "#184f95", "#0d366b"]), s=14, edgecolor="white", lw=0.4, zorder=3)
    ax.axhline(0, color=MUTED, lw=0.5)
    cb = fig.colorbar(sc, ax=ax, fraction=0.05, pad=0.02); cb.set_label("fastest trip (h)", fontsize=7)
    cb.ax.tick_params(labelsize=6.5); cb.outline.set_linewidth(0.4)
    ax.set_xlabel("oracle ceiling for the OD pair (%)"); ax.set_ylabel("predictive reduction (%)")
    r1 = spearmanr(o, g); r2 = spearmanr(d.T_fast_h, g)
    ax.set_title(f"OSM primary, 78 OD pairs, $K=10$ (Spearman $\\rho={r1.correlation:.2f}$)", pad=3)
    fig.tight_layout()
    save(fig, "fig_perod")
    print("perod spearman oracle", r1, "Tfast", r2, "capture median", float(np.median(g / o)))
    J2 = {"rho_oracle": float(r1.correlation), "p_oracle": float(r1.pvalue), "rho_T": float(r2.correlation),
          "p_T": float(r2.pvalue), "capture_median": float(np.median(g / o)),
          "gain_short": float(g[d.T_fast_h < 8].mean()), "gain_long": float(g[d.T_fast_h >= 8].mean()),
          "n_short": int((d.T_fast_h < 8).sum())}
    (P3 / "perod_stats.json").write_text(json.dumps(J2, indent=1)); print(J2)


def fig_hyst():
    d = J("extra_summary.json")["corridors"]
    fig, ax = plt.subplots(figsize=(COL, 1.95))
    mk = {"lviv_kyiv": "o", "odesa_kyiv": "s", "kharkiv_lviv": "^", "dnipro_kyiv": "D"}
    cols = {"lviv_kyiv": BLUE, "odesa_kyiv": ORANGE, "kharkiv_lviv": AQUA, "dnipro_kyiv": INK2}
    for k in NAMES:
        h = d[k]["hyst"]
        ax.plot([x["churn"] for x in h], [x["impr"] for x in h], color=cols[k], marker=mk[k], ms=3, lw=1.0, label=NAMES[k])
        x = [e for e in h if e["theta"] == 0.0025][0]
        ax.scatter([x["churn"]], [x["impr"]], s=36, facecolor="none", edgecolor=INK, lw=0.7, zorder=5)
    ax.set_xlabel("route changes over the test period"); ax.set_ylabel("reduction of km under alert (%)")
    ax.set_title("threshold $\\theta_s$ from 0.02 (left) to 0 (right); circles: $\\theta_s=0.0025$", pad=3)
    ax.legend(loc="lower right", ncol=2, columnspacing=0.8)
    fig.tight_layout()
    save(fig, "fig_hyst")


def fig_case():
    c = J("extra_summary.json")["corridors"][os.environ.get("P3_CASE", "dnipro_kyiv")]["case"]
    nj = min(19, int(np.ceil(max(c["static_T_h"], c["chosen_T_h"]))) + 1)
    fig, axs = plt.subplots(1, 2, figsize=(TXT, 1.9), gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axs[0]
    rid = lambda x: x.split("_")[-1]
    rows = [(f"static baseline ({rid(c['static'])})", c["static_trav"], c["static_km_ua"], c["static_T_h"]),
            (f"predictive choice ({rid(c['chosen'])})", c["chosen_trav"], c["chosen_km_ua"], c["chosen_T_h"])]
    for y, (lab, tr, kua, T) in zip([1, 0], rows):
        nl = 0
        for n, s in enumerate(tr):
            ax.barh(y, s["t1"] - s["t0"], left=s["t0"], height=0.5, color=["#e6e5e0", "#cfcdc6"][n % 2],
                    edgecolor="white", lw=0.6, zorder=2)
            al = c["alert"][s["oblast"]]
            for j in range(int(np.floor(s["t0"])), int(np.ceil(s["t1"]))):
                a0, a1 = max(s["t0"], j), min(s["t1"], j + 1)
                if a1 > a0 and al[j]:
                    ax.barh(y, a1 - a0, left=a0, height=0.5, color=RED, alpha=0.85, lw=0, zorder=3)
            if s["t1"] - s["t0"] > 0.6:
                up = nl % 2 == 0; nl += 1
                ax.text((s["t0"] + s["t1"]) / 2, y + (0.3 if up else -0.3), SH.get(s["oblast"], EN.get(s["oblast"], s["oblast"])),
                        ha="center", va="bottom" if up else "top", fontsize=5.8, color=INK2)
        ax.text(T + 0.15, y, f"{kua:.0f} km under alert, {T:.1f} h", va="center", fontsize=6.5, color=INK)
    ax.set_yticks([1, 0]); ax.set_yticklabels([r[0] for r in rows]); ax.set_ylim(-0.75, 1.75)
    ax.set_xlim(0, 1.55 * max(c["static_T_h"], c["chosen_T_h"])); ax.set_xlabel("hours after departure"); ax.grid(axis="y", visible=False)
    ax.set_title("(a) realised alerts along the two routes (red)", pad=3)
    ax = axs[1]
    obl = []
    for tr in (c["static_trav"], c["chosen_trav"]):
        for s_ in tr:
            if s_["oblast"] not in obl and s_["km"] >= 5:
                obl.append(s_["oblast"])
    M = np.array([c["prob"][o][:nj] for o in obl])
    im = ax.imshow(M, aspect="auto", cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
        "b", ["#f0efec", "#9ec5f4", "#3987e5", "#184f95"]), vmin=0, vmax=1)
    for i, o in enumerate(obl):
        for j in range(nj):
            if c["alert"][o][j]:
                ax.plot(j, i, marker="s", ms=2.2, color=RED, lw=0)
    ax.set_yticks(range(len(obl))); ax.set_yticklabels([EN[o] for o in obl], fontsize=6.5)
    ax.set_xticks(range(0, nj, 2 if nj <= 13 else 3)); ax.set_xlabel("lead (h)"); ax.grid(False)
    ax.set_title("(b) forecast at departure; red: realised alert", pad=3)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.ax.tick_params(labelsize=6); cb.outline.set_linewidth(0.4)
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig_case")


def fig_sens():
    S = lambda g, t: json.loads((P3 / "sens" / f"roadgraph_osm_{g}_{t}.json").read_text())
    fig, axs = plt.subplots(1, 3, figsize=(TXT, 1.95))
    ax = axs[0]
    for g, ls, lab in [("primary", "-", "OSM primary"), ("secondary", "--", "OSM secondary")]:
        xs = [1.1, 1.2, 1.3, 1.5, 2.0] if g == "primary" else [1.2, 1.5, 2.0]
        d = [S(g, "base" if x == 1.5 else f"stretch{x}") for x in xs]
        ax.plot(xs, [x["pred_K10"]["pooled_pct"] for x in d], color=BLUE, ls=ls, marker="o", ms=3, lw=1.1,
                label=f"predictive, {lab}")
        ax.plot(xs, [x["oracle_K10"]["pooled_pct"] for x in d], color=MUTED, ls=ls, marker="s", ms=3, lw=1.0,
                label=f"oracle, {lab}")
    ax.set_xlabel("stretch bound"); ax.set_ylabel("reduction of km under alert (%)"); ax.set_title("(a) stretch, $\\theta=0.7$, $K=10$", pad=3)
    ax = axs[1]
    for g, ls in [("primary", "-"), ("secondary", "--")]:
        xs = [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] if g == "primary" else [0.3, 0.5, 0.6, 0.7, 0.8, 1.0]
        d = [S(g, "base" if x == 0.7 else f"theta{x}") for x in xs]
        ax.plot(xs, [x["pred_K10"]["pooled_pct"] for x in d], color=BLUE, ls=ls, marker="o", ms=3, lw=1.1)
        ax.plot(xs, [x["oracle_K10"]["pooled_pct"] for x in d], color=MUTED, ls=ls, marker="s", ms=3, lw=1.0)
    ax.set_xlabel("overlap limit $\\theta$"); ax.set_title("(b) overlap limit, stretch 1.5, $K=10$", pad=3)
    ax = axs[2]
    Ks = [1, 2, 3, 5, 10, 15, 20]
    for g, ls in [("primary", "-"), ("secondary", "--")]:
        d = S(g, "base")
        ax.plot(Ks, [d[f"pred_K{k}"]["pooled_pct"] for k in Ks], color=BLUE, ls=ls, marker="o", ms=3, lw=1.1)
        ax.plot(Ks, [d[f"oracle_K{k}"]["pooled_pct"] for k in Ks], color=MUTED, ls=ls, marker="s", ms=3, lw=1.0)
        ax.plot(Ks, [d[f"static_K{k}"]["pooled_pct"] for k in Ks], color=INK2, ls=ls, marker="D", ms=2.5, lw=0.9)
    ax.axhline(0, color=MUTED, lw=0.5)
    ax.set_xscale("log"); ax.set_xticks(Ks); ax.set_xticklabels(Ks); ax.minorticks_off()
    ax.set_xlabel("candidate paths $K$"); ax.set_title("(c) up to $K=20$ (dark: static)", pad=3)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig_sens")


if __name__ == "__main__":
    which = sys.argv[1:] or ["pareto", "graph", "backends", "rolling", "map"]
    for w in which:
        globals()[f"fig_{w}"]()
