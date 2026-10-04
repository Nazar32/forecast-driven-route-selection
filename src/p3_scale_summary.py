"""Summarise scale_<graph>_<backend>.json into a markdown table + figure (results/p3/)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = Path(__file__).resolve().parent.parent / "results" / "p3"
GRAPHS = [("roadgraph", "corridor"), ("roadgraph_osm_primary", "OSM primary"), ("roadgraph_osm_secondary", "OSM secondary")]
BACK = ["memory", "postgres", "neo4j"]
COL = {"memory": "#8a8984", "postgres": "#2a78d6", "neo4j": "#1baf7a"}
MK = {"memory": "s", "postgres": "o", "neo4j": "^"}


def get(g, b):
    p = RES / f"scale_{g}_{b}.json"
    return json.loads(p.read_text()) if p.exists() else None


def best_s1(d, q):
    vals = [d["S1"][k]["median_ms"] for k in (f"Q{q}_single", f"Q{q}_batch") if k in d["S1"]]
    return min(vals) if vals else None


rows = ["| Graph | Nodes / edges | Backend | S1 Q=1 | S1 Q=100 | S1 Q=1000 | S2 Yen per query | S3 k=6 per query | Mismatches |",
        "|---|---|---|---|---|---|---|---|---|"]
for g, name in GRAPHS:
    for b in BACK:
        d = get(g, b)
        if not d:
            continue
        f = lambda x: "n/a" if x is None else (f"{x:.3f} ms" if x < 1 else f"{x:.1f} ms" if x < 1e4 else f"{x / 1000:.1f} s")
        rows.append(f"| {name} | {d['nodes']} / {d['edges']} | {b} | {f(best_s1(d, 1))} | {f(best_s1(d, 100))} | "
                    f"{f(best_s1(d, 1000))} | {f(d['S2']['Q100']['per_query_ms'])} | {f(d['S3']['k6']['per_query_ms'])} | "
                    f"{'ref' if d['mismatch_groups_vs_reference'] is None else d['mismatch_groups_vs_reference']} |")
table = "\n".join(rows)
(RES / "scale_summary.md").write_text(table + "\n")
print(table)

fig, axs = plt.subplots(1, 3, figsize=(12, 3.4))
x = [get(g, "memory")["edges"] for g, _ in GRAPHS if get(g, "memory")]
titles = ["S1 fleet cycle, Q = 1000 (best variant)", "S2 Yen K = 10, per OD pair", "S3 6-hop neighbourhood, per start"]
for ax, t, fn in zip(axs, titles, [lambda d: best_s1(d, 1000), lambda d: d["S2"]["Q100"]["per_query_ms"],
                                  lambda d: d["S3"]["k6"]["per_query_ms"]]):
    for b in BACK:
        pts = [(get(g, b)["edges"], fn(get(g, b))) for g, _ in GRAPHS if get(g, b)]
        ax.plot(*zip(*pts), color=COL[b], marker=MK[b], lw=2, ms=6, label=b)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(t, fontsize=9.5)
    ax.set_xlabel("road edges in graph"); ax.grid(True, color="#e6e5e0", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
axs[0].set_ylabel("latency, ms (log)")
axs[0].legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(RES / "fig_graphdb_scale.png", dpi=200)
