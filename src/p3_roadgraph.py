"""
Paper 3 / experiment (d), step 1: build a real road graph from ORS route geometries.

The 34 ORS (driving-car) routes of the four corridors cover the main national highway
network of central/western Ukraine (M03, M05, M06, M12, M30, H-roads ...). Where two
routes run on the same road their OSM-derived vertices coincide; where they cross or
diverge, a junction appears. We:

  1. snap every geometry vertex to a 150 m grid cell (merges both carriageways of a
     dual carriageway and vertices of the same road from different ORS versions),
  2. connect consecutive cells along each route (undirected road segments),
  3. contract degree-2 chains into edges with attributes
        length_m, time_s (median ORS speed over all traversals, ORS-version harmonised),
        profile: ordered [(oblast, distance_m, time_s), ...] along the edge,
  4. keep the largest connected component.

The result is a junction-level road graph whose paths are concatenations of real road
pieces; combining pieces yields candidate routes that no single ORS query produced.
It is a *corridor network* (roads used by at least one ORS route), not the full OSM
graph: the build step accepts any edge list, so a full OSM extract can replace it.

Outputs (results/p3/roadgraph/):
  nodes.csv  (node_id, lon, lat, city)       edges.csv (u, v, length_m, time_s, profile_json)
Run: python src/p3_roadgraph.py
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import p3_geo as G  # noqa: E402

OUT = G.ROOT / "results" / "p3" / "roadgraph"
CELL_M = 150.0
PIECE_S = 300.0     # max duration of one profile piece (time resolution of exposure)
LAT0 = math.radians(49.0)

CITIES = {  # waypoint coordinates as used by the corridor generators (ORS geocoded)
    "Lviv": (24.031111, 49.842957), "Kyiv": (30.523333, 50.450001), "Odesa": (30.7233, 46.4825),
    "Kharkiv": (36.2304, 49.9935), "Dnipro": (35.0462, 48.4647), "Ternopil": (25.5948, 49.5535),
    "Rivne": (26.2516, 50.6199), "Zhytomyr": (28.6587, 50.2547), "Vinnytsia": (28.4682, 49.2331),
    "Khmelnytskyi": (26.9871, 49.4229), "Uman": (30.2219, 48.7484), "Bila Tserkva": (30.117, 49.8066),
    "Cherkasy": (32.0598, 49.4444), "Kropyvnytskyi": (32.2623, 48.5079), "Kremenchuk": (33.4211, 49.0659),
    "Poltava": (34.5514, 49.5883), "Lubny": (32.9906, 50.0186), "Boryspil": (30.955, 50.3527),
    "Mykolaiv": (31.9946, 46.975), "Voznesensk": (31.3346, 47.5670), "Lutsk": (25.3254, 50.7472),
    "Berdychiv": (28.5853, 49.8994),
}


def xy(lon, lat):
    return (math.radians(lon) * 6371008.8 * math.cos(LAT0), math.radians(lat) * 6371008.8)


def cell_of(lon, lat):
    x, y = xy(lon, lat)
    return (int(math.floor(x / CELL_M)), int(math.floor(y / CELL_M)))


def build():
    g = nx.Graph()
    cell_pts = defaultdict(list)
    trav_speed = defaultdict(list)      # (a,b) -> [(dist, time, ors_version)]
    seg_obl = {}                        # (a,b) -> oblast of the sub-segment midpoint
    for key, path in G.GEOJSON.items():
        gj = json.loads(Path(path).read_text())
        ver = gj["metadata"].get("engine", {}).get("version", "?")
        for f in gj["features"]:
            coords, seg_d, dur, _, _ = G.point_times(f)
            prev = None
            acc_d = acc_t = 0.0
            for i, (lon, lat) in enumerate(coords):
                c = cell_of(lon, lat)
                cell_pts[c].append((lon, lat))
                if i > 0:
                    acc_d += seg_d[i - 1]; acc_t += dur[i - 1]
                if prev is not None and c != prev:
                    e = (min(prev, c), max(prev, c))
                    trav_speed[e].append((acc_d, acc_t, ver))
                    if e not in seg_obl:
                        mid = ((coords[i - 1][0] + lon) / 2, (coords[i - 1][1] + lat) / 2)
                        seg_obl[e] = G.locate(*mid)
                    g.add_edge(prev, c)
                    acc_d = acc_t = 0.0
                prev = c
    # harmonise ORS versions: speed ratio on cell-edges traversed by both versions
    ratios = []
    for e, lst in trav_speed.items():
        v90 = [t / d for d, t, v in lst if v.startswith("9.0") and d > 0]
        v99 = [t / d for d, t, v in lst if not v.startswith("9.0") and d > 0]
        if v90 and v99:
            ratios.append(np.median(v99) / np.median(v90))
    k_ver = float(np.median(ratios)) if ratios else 1.0
    for (a, b) in g.edges:
        lst = trav_speed[(min(a, b), max(a, b))]
        d = float(np.median([x[0] for x in lst]))
        pace = np.median([(t / dd) * (k_ver if v.startswith("9.0") else 1.0) for dd, t, v in lst if dd > 0])
        g.edges[a, b].update(length_m=d, time_s=float(d * pace), oblast=seg_obl[(min(a, b), max(a, b))])
    for c, pts in cell_pts.items():
        arr = np.array(pts)
        g.nodes[c]["lon"], g.nodes[c]["lat"] = float(arr[:, 0].mean()), float(arr[:, 1].mean())
    g = g.subgraph(max(nx.connected_components(g), key=len)).copy()
    return g, k_ver


def attach_cities(g):
    ids = list(g.nodes)
    P = np.array([xy(g.nodes[n]["lon"], g.nodes[n]["lat"]) for n in ids])
    city_node = {}
    for name, (lon, lat) in CITIES.items():
        q = np.array(xy(lon, lat))
        dist = np.hypot(*(P - q).T)
        j = int(dist.argmin())
        if dist[j] < 8000:                       # route passes the city
            city_node[name] = ids[j]
    return city_node


def contract(g, keep):
    """Contract degree-2 chains; keep junctions, dead ends and city nodes."""
    keep_cities = set(keep)
    keep = set(keep) | {n for n in g.nodes if g.degree(n) != 2}
    H = nx.Graph()
    seen = set()
    for s in keep:
        for nb in g.neighbors(s):
            if (s, nb) in seen:
                continue
            chain = [s, nb]
            while chain[-1] not in keep:
                a, b = chain[-2], chain[-1]
                nxt = [x for x in g.neighbors(b) if x != a]
                chain.append(nxt[0])
            for a, b in zip(chain, chain[1:]):
                seen.add((a, b)); seen.add((b, a))
            u, v = chain[0], chain[-1]
            if u == v:
                continue
            prof, L, T = [], 0.0, 0.0
            for a, b in zip(chain, chain[1:]):
                e = g.edges[a, b]
                L += e["length_m"]; T += e["time_s"]
                if prof and prof[-1][0] == e["oblast"] and prof[-1][2] < PIECE_S:
                    prof[-1][1] += e["length_m"]; prof[-1][2] += e["time_s"]
                else:
                    prof.append([e["oblast"], e["length_m"], e["time_s"]])
            if H.has_edge(u, v):
                # genuine parallel road between the same junctions: keep it by routing it
                # through a virtual mid node (the chain's middle vertex)
                old = H.edges[u, v]["time_s"]
                if abs(old - T) < max(120.0, 0.05 * min(old, T)):   # same road (carriageway / snapping variant)
                    if T < old:
                        H.edges[u, v].update(length_m=L, time_s=T, profile=prof, fwd=(u, v))
                    continue
                m = chain[len(chain) // 2]
                if m in (u, v):
                    continue
                k = sum(1 for _ in zip(chain, chain[1:]))
                half = _split_profile(prof, 0.5)
                H.add_edge(u, m, length_m=L / 2, time_s=T / 2, profile=half[0], fwd=(u, m))
                H.add_edge(m, v, length_m=L / 2, time_s=T / 2, profile=half[1], fwd=(m, v))
                continue
            H.add_edge(u, v, length_m=L, time_s=T, profile=prof, fwd=(u, v))
    for n in H.nodes:
        H.nodes[n].update(g.nodes[n])
    return simplify(H, keep_extra=set(keep_cities))


def _split_profile(prof, frac):
    L = sum(p[1] for p in prof)
    cut = L * frac
    a, b, acc = [], [], 0.0
    for o, d, t in prof:
        if acc + d <= cut:
            a.append([o, d, t])
        elif acc >= cut:
            b.append([o, d, t])
        else:
            x = (cut - acc) / d
            a.append([o, d * x, t * x]); b.append([o, d * (1 - x), t * (1 - x)])
        acc += d
    return a, b


def _oriented(H, a, b):
    e = H.edges[a, b]
    return e["profile"] if e["fwd"] == (a, b) else [list(p) for p in e["profile"][::-1]]


def simplify(H, keep_extra):
    """Iteratively merge degree-2 nodes (not cities) unless that would create a parallel edge."""
    changed = True
    while changed:
        changed = False
        for n in list(H.nodes):
            if n in keep_extra or n not in H or H.degree(n) != 2:
                continue
            a, b = list(H.neighbors(n))
            if a == b or H.has_edge(a, b):
                continue
            p1, p2 = _oriented(H, a, n), _oriented(H, n, b)
            prof = [list(x) for x in p1]
            for o, d, t in p2:
                if prof and prof[-1][0] == o and prof[-1][2] + t <= PIECE_S:
                    prof[-1][1] += d; prof[-1][2] += t
                else:
                    prof.append([o, d, t])
            L = H.edges[a, n]["length_m"] + H.edges[n, b]["length_m"]
            T = H.edges[a, n]["time_s"] + H.edges[n, b]["time_s"]
            H.remove_node(n)
            H.add_edge(a, b, length_m=L, time_s=T, profile=prof, fwd=(a, b))
            changed = True
    return H


def merge_micro(H, cities, min_len=1000.0):
    """Cluster interchange micro-structure: contract edges shorter than min_len.
    The (<1 km) contracted piece is dropped from distance/time; parallel edges created by
    the merge follow the same same-road rule as in contract()."""
    city_nodes = set(cities.values())
    while True:
        short = [(u, v) for u, v, e in H.edges(data=True) if e["length_m"] < min_len]
        if not short:
            break
        u, v = short[0]
        if v in city_nodes and u not in city_nodes:
            u, v = v, u                               # keep the city node
        for w in list(H.neighbors(v)):
            if w == u:
                continue
            e = dict(H.edges[v, w])
            prof = _oriented(H, v, w)
            new = dict(length_m=e["length_m"], time_s=e["time_s"], profile=prof, fwd=(u, w))
            if H.has_edge(u, w):
                old = H.edges[u, w]["time_s"]
                if abs(old - new["time_s"]) < max(120.0, 0.05 * min(old, new["time_s"])):
                    if new["time_s"] < old:
                        H.edges[u, w].update(new)
                    continue
                # genuine parallel: keep via a virtual node
                m = ("virt", v, w)
                a, b = _split_profile(prof, 0.5)
                H.add_node(m, lon=(H.nodes[u]["lon"] + H.nodes[w]["lon"]) / 2, lat=(H.nodes[u]["lat"] + H.nodes[w]["lat"]) / 2)
                H.add_edge(u, m, length_m=new["length_m"] / 2, time_s=new["time_s"] / 2, profile=a, fwd=(u, m))
                H.add_edge(m, w, length_m=new["length_m"] / 2, time_s=new["time_s"] / 2, profile=b, fwd=(m, w))
                continue
            H.add_edge(u, w, **new)
        H.remove_node(v)
        for k, n in list(cities.items()):
            if n == v:
                cities[k] = u
    return simplify(H, keep_extra=set(cities.values()))


def main():
    g, k_ver = build()
    cities = attach_cities(g)
    H = contract(g, cities.values())
    H = merge_micro(H, cities)
    OUT.mkdir(parents=True, exist_ok=True)
    idmap = {n: i for i, n in enumerate(H.nodes)}
    inv_city = {v: k for k, v in cities.items()}
    pd.DataFrame([{"node_id": idmap[n], "lon": H.nodes[n]["lon"], "lat": H.nodes[n]["lat"],
                   "city": inv_city.get(n, "")} for n in H.nodes]).to_csv(OUT / "nodes.csv", index=False)
    rows = []
    for u, v, e in H.edges(data=True):
        prof = e["profile"] if e["fwd"] == (u, v) else e["profile"][::-1]
        rows.append({"u": idmap[u], "v": idmap[v], "length_m": e["length_m"], "time_s": e["time_s"],
                     "profile_json": json.dumps(prof, ensure_ascii=False)})
    pd.DataFrame(rows).to_csv(OUT / "edges.csv", index=False)
    meta = {"cell_m": CELL_M, "ors_version_pace_ratio_9.9_over_9.0": k_ver,
            "fine_nodes": g.number_of_nodes(), "fine_edges": g.number_of_edges(),
            "nodes": H.number_of_nodes(), "edges": H.number_of_edges(),
            "total_km": sum(e["length_m"] for *_, e in H.edges(data=True)) / 1000,
            "cities": {k: idmap[v] for k, v in cities.items()},
            "missing_cities": [c for c in CITIES if c not in cities]}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
