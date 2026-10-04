"""
OSM road graph with raion-level region profiles (same edges, nodes, times and oblast split as
p3_osm_graph.py; every profile piece additionally lies in one raion and is labelled "<oblast>|<raion>").

Output: results/p3/roadgraph_osm_<level>_raion/ (nodes.csv, edges.csv, meta.json), same format.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import p3_osm_graph as OG  # noqa: E402
import p3_raion as RA  # noqa: E402

_orig_assign = OG.assign_oblast


def assign_unit(x, y):
    idx, names = _orig_assign(x, y)
    obl = np.array([names[i] for i in idx], dtype=object)
    un = RA.raion_of_vec(obl, np.asarray(x), np.asarray(y))
    labels = sorted(set(un))
    pos = {u: k for k, u in enumerate(labels)}
    OG.EXCLUDE = {u for u in labels if RA.oblast_of(u) in ORIG_EXCLUDE}
    return np.array([pos[u] for u in un], dtype=np.int32), labels


ORIG_EXCLUDE = set(OG.EXCLUDE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", choices=list(OG.LEVELS), default="primary")
    a = ap.parse_args()
    OG.assign_oblast = assign_unit
    g, cities, stats = OG.build(a.level, OG.PBF)
    k, ratios = OG.calibrate(g, cities)
    stats.update({"level": a.level, "time_factor": k, "ors_over_graph_before_calibration": ratios,
                  "region_units": "oblast|raion (OSM admin_level 6, 2026-10-01)"})
    OG.EXCLUDE = ORIG_EXCLUDE
    OG.write(g, cities, OG.OUTBASE / f"roadgraph_osm_{a.level}_raion", stats)


if __name__ == "__main__":
    main()
