"""
Paper 3 / experiment (d) on the FULL OSM road network of Ukraine.

Input : data/osm/ukraine-roads.osm.pbf
        (Geofabrik extract filtered with osmium to highway=motorway..secondary incl. links)
Output: results/p3/roadgraph_osm_<level>/{nodes.csv, edges.csv, meta.json}
        in the same format as the corridor graph (p3_roadgraph.py), so p3_graph_eval.py,
        p3_graphdb_bench.py and p3_graphdb_scale.py run on it unchanged (P3_GRAPH_DIR).

Levels: primary   = motorway, trunk, primary (+ links)
        secondary = primary + secondary (+ links)

Construction
  1. ways split at every node shared by >1 way (true junctions) and at way ends;
  2. sub-segment length = haversine; free-flow speed by road class (maxspeed tag if numeric
     and lower), then ONE global factor calibrated so that the graph's fastest times match the
     ORS fastest route of the four ORS corridors (median ratio = 1);
  3. every sub-segment is assigned to an oblast (geoBoundaries ADM1, city of Kyiv first);
  4. edges lying mostly in occupied / inaccessible territory are removed (assumption stated in
     the paper): AR Crimea, Sevastopol, Luhansk, Donetsk oblasts;
  5. degree-2 chains are contracted, profile pieces kept at <= 5 min resolution;
  6. largest connected component; oblast centres attached to the nearest node (<= 5 km).
Roads are treated as undirected (dual carriageways give parallel alternatives of equal cost).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import shapely

sys.path.insert(0, str(Path(__file__).parent))
import p3_geo as G  # noqa: E402
from p3_roadgraph import CITIES, PIECE_S, simplify  # noqa: E402

ROOT = G.ROOT
PBF = ROOT / "data" / "osm" / "ukraine-roads.osm.pbf"
OUTBASE = ROOT / "results" / "p3"

SPEED = {"motorway": 110, "motorway_link": 60, "trunk": 90, "trunk_link": 50,
         "primary": 75, "primary_link": 40, "secondary": 60, "secondary_link": 35}
LEVELS = {"primary": {"motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link"}}
LEVELS["secondary"] = LEVELS["primary"] | {"secondary", "secondary_link"}
EXCLUDE = {"Автономна Республіка Крим", "Севастополь", "Луганська область", "Донецька область"}
ORS_OD = {"lviv_kyiv": ("Lviv", "Kyiv"), "odesa_kyiv": ("Odesa", "Kyiv"),
          "kharkiv_lviv": ("Kharkiv", "Lviv"), "dnipro_kyiv": ("Dnipro", "Kyiv")}


def read_ways(pbf, classes):
    import osmium

    ways = []

    class H(osmium.SimpleHandler):
        def way(self, w):
            hw = w.tags.get("highway")
            if hw not in classes:
                return
            pts = [(n.ref, n.lon, n.lat) for n in w.nodes if n.location.valid()]
            if len(pts) < 2:
                return
            ms = w.tags.get("maxspeed", "")
            try:
                ms = float(ms.split()[0])
            except (ValueError, IndexError):
                ms = None
            ways.append((hw, ms, pts))

    H().apply_file(str(pbf), locations=True, idx="flex_mem")
    return ways


def haversine_vec(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    d = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371008.8 * np.arcsin(np.sqrt(d))


def assign_oblast(x, y):
    polys = G.oblast_polygons()                       # sorted small -> large
    names = [p[0] for p in polys]
    idx = np.full(len(x), -1, dtype=np.int16)
    for k, (_, geom, _) in enumerate(polys):
        m = idx == -1
        if not m.any():
            break
        hit = shapely.contains_xy(geom, x[m], y[m])
        sub = np.where(m)[0][hit]
        idx[sub] = k
    rest = np.where(idx == -1)[0]
    if len(rest):                                     # border simplification gaps -> nearest
        pts = shapely.points(x[rest], y[rest])
        dist = np.stack([shapely.distance(p[1], pts) for p in polys])
        idx[rest] = dist.argmin(0)
    return idx, names


def build(level, pbf=PBF):
    ways = read_ways(pbf, LEVELS[level])
    use = defaultdict(int)
    for _, _, pts in ways:
        for i, (ref, _, _) in enumerate(pts):
            use[ref] += 2 if i in (0, len(pts) - 1) else 1
    # sub-segments
    rows = []                                          # (u_ref, v_ref, lon1, lat1, lon2, lat2, speed, way_idx, seq)
    coord = {}
    for wi, (hw, ms, pts) in enumerate(ways):
        sp = SPEED[hw] if ms is None else min(SPEED[hw], ms)
        for i in range(len(pts) - 1):
            (a, x1, y1), (b, x2, y2) = pts[i], pts[i + 1]
            coord[a] = (x1, y1); coord[b] = (x2, y2)
            rows.append((a, b, x1, y1, x2, y2, sp, wi))
    R = np.array([r[2:] for r in rows], dtype=np.float64)
    L = haversine_vec(R[:, 0], R[:, 1], R[:, 2], R[:, 3])
    T = L / (R[:, 4] / 3.6)
    oi, names = assign_oblast((R[:, 0] + R[:, 2]) / 2, (R[:, 1] + R[:, 3]) / 2)
    # split ways into junction-to-junction edges
    g = nx.Graph()
    cur, prof, Lsum, Tsum, start = None, [], 0.0, 0.0, None
    for k, r in enumerate(rows):
        a, b, wi = r[0], r[1], r[7]
        if cur != wi:
            cur, prof, Lsum, Tsum, start = wi, [], 0.0, 0.0, a
        o = names[oi[k]]
        if prof and prof[-1][0] == o and prof[-1][2] + T[k] <= PIECE_S:
            prof[-1][1] += L[k]; prof[-1][2] += T[k]
        else:
            prof.append([o, float(L[k]), float(T[k])])
        Lsum += L[k]; Tsum += T[k]
        last_of_way = (k + 1 == len(rows)) or rows[k + 1][7] != wi
        if use[b] > 1 or last_of_way:
            if start != b:
                excl = sum(p[1] for p in prof if p[0] in EXCLUDE)
                if excl < 0.5 * Lsum and (not g.has_edge(start, b) or g.edges[start, b]["time_s"] > Tsum):
                    g.add_edge(start, b, length_m=float(Lsum), time_s=float(Tsum), profile=prof, fwd=(start, b))
            prof, Lsum, Tsum, start = [], 0.0, 0.0, b
    g = g.subgraph(max(nx.connected_components(g), key=len)).copy()
    for n in g.nodes:
        g.nodes[n]["lon"], g.nodes[n]["lat"] = coord[n]
    # attach cities, contract
    ids = np.array(list(g.nodes))
    XY = np.array([coord[n] for n in ids])
    cities = {}
    for name, (lon, lat) in CITIES.items():
        d = haversine_vec(XY[:, 0], XY[:, 1], np.full(len(XY), lon), np.full(len(XY), lat))
        j = int(d.argmin())
        if d[j] < 5000:
            cities[name] = ids[j]
    n_before = g.number_of_nodes()
    g = simplify(g, keep_extra=set(cities.values()))
    return g, cities, {"ways": len(ways), "subsegments": len(rows), "nodes_before_contraction": n_before}


def calibrate(g, cities):
    """Global time factor so that graph fastest times match the ORS fastest corridor routes."""
    ratios = {}
    for key, (a, b) in ORS_OD.items():
        if a not in cities or b not in cities:
            continue
        routes = G.load_corridor_geo(key)
        ors_T = min(r["T"] for r in routes)
        t = nx.shortest_path_length(g, cities[a], cities[b], weight="time_s")
        ratios[key] = ors_T / t
    k = float(np.median(list(ratios.values())))
    for _, _, e in g.edges(data=True):
        e["time_s"] *= k
        e["profile"] = [[o, d, dt * k] for o, d, dt in e["profile"]]
    return k, ratios


def write(g, cities, out, meta):
    out.mkdir(parents=True, exist_ok=True)
    idmap = {n: i for i, n in enumerate(g.nodes)}
    inv = {v: k for k, v in cities.items()}
    pd.DataFrame([{"node_id": idmap[n], "lon": g.nodes[n]["lon"], "lat": g.nodes[n]["lat"],
                   "city": inv.get(n, "")} for n in g.nodes]).to_csv(out / "nodes.csv", index=False)
    rows = []
    for u, v, e in g.edges(data=True):
        prof = e["profile"] if e["fwd"] == (u, v) else e["profile"][::-1]
        rows.append({"u": idmap[u], "v": idmap[v], "length_m": e["length_m"], "time_s": e["time_s"],
                     "profile_json": json.dumps(prof, ensure_ascii=False)})
    pd.DataFrame(rows).to_csv(out / "edges.csv", index=False)
    meta.update({"nodes": g.number_of_nodes(), "edges": g.number_of_edges(),
                 "total_km": sum(e["length_m"] for *_, e in g.edges(data=True)) / 1000,
                 "cities": {k: idmap[v] for k, v in cities.items()},
                 "missing_cities": [c for c in CITIES if c not in cities],
                 "excluded_oblasts": sorted(EXCLUDE), "speeds_kmh": SPEED})
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in meta.items() if k != "speeds_kmh"}, indent=2, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", choices=list(LEVELS), default="primary")
    ap.add_argument("--pbf", default=str(PBF))
    a = ap.parse_args()
    g, cities, stats = build(a.level, Path(a.pbf))
    k, ratios = calibrate(g, cities)
    stats.update({"level": a.level, "time_factor": k, "ors_over_graph_before_calibration": ratios})
    write(g, cities, OUTBASE / f"roadgraph_osm_{a.level}", stats)


if __name__ == "__main__":
    main()
