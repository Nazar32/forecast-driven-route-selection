"""
Paper 3: static criteria of the infological model for every corridor candidate route.

Criteria (all "less is better"), computed offline and identically for the four ORS corridors:
  rail      railway level crossings on the route (OSM railway=level_crossing nodes within 8 m of the
            route geometry; nodes closer than 100 m along the route are one crossing)
  junction  slowing points: pedestrian crossings, traffic signals, stop and give-way nodes
            (merged within 30 m)
  settle_km km on roads with a posted limit of 60 km/h or less (settlement passages)
  bridges   bridge passages (route runs on OSM bridge=* ways; runs < 500 m apart are one passage)
  poor_km   km on a surface other than asphalt, concrete or generic paved (ORS surface extra)
Inputs : data/osm/ukraine-roads.osm.pbf (Geofabrik extract, roads + their nodes),
         the ORS corridor GeoJSONs used everywhere in the paper (p3_geo.GEOJSON).
Output : results/p3v2/infocriteria.csv  (corridor, route id, T, D, criteria)
Validation against the independent Overpass-based extraction of the infological-model paper
(routes_efficiency_calc/main.py, Lviv-Kyiv): results/p3v2/infocriteria_validation.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
import p3_core as C  # noqa: E402
import p3_geo as G  # noqa: E402

PBF = C.DATA / "osm" / "ukraine-roads.osm.pbf"
LAT0 = np.radians(48.5)
R_E = 6371000.0
STEP = 5.0                                  # densification step of the route geometry (m)
GOOD_SURFACE = {1, 3, 4}                    # ORS: paved, asphalt, concrete
STREET = 3                                  # ORS waytype: street


def xy(lonlat):
    a = np.radians(np.asarray(lonlat, dtype=float))
    return np.c_[R_E * a[:, 0] * np.cos(LAT0), R_E * a[:, 1]]


def read_osm():
    import osmium
    rail, junc, bridge_pts = [], [], []

    class H(osmium.SimpleHandler):
        def node(self, n):
            t = n.tags
            if t.get("railway") == "level_crossing":
                rail.append((n.location.lon, n.location.lat))
            elif t.get("highway") in ("crossing", "traffic_signals", "stop", "give_way"):
                junc.append((n.location.lon, n.location.lat))

        def way(self, w):
            b = w.tags.get("bridge")
            ms = w.tags.get("maxspeed", "")
            is_b = bool(b and b != "no")
            is_s = ms.isdigit() and int(ms) <= 60
            if is_b or is_s:
                pts = [(nd.lon, nd.lat) for nd in w.nodes if nd.location.valid()]
                if len(pts) >= 2:
                    d = dens_pts(pts)
                    if is_b:
                        bridge_pts.extend(d)
                    if is_s:
                        slow_pts.extend(d)

    slow_pts = []
    H().apply_file(str(PBF), locations=True)
    return xy(rail), xy(junc), np.asarray(bridge_pts), np.asarray(slow_pts)


def dens_pts(pts):
    p = xy(pts)
    out = []
    for a, c in zip(p[:-1], p[1:]):
        k = max(int(np.hypot(*(c - a)) // STEP), 1)
        out.extend(a + (c - a) * s for s in np.linspace(0, 1, k + 1))
    return out


def densify(coords):
    p = xy(coords)
    seg = np.hypot(*(p[1:] - p[:-1]).T)
    cum = np.r_[0, np.cumsum(seg)]
    s = np.arange(0, cum[-1], STEP)
    return np.c_[np.interp(s, cum, p[:, 0]), np.interp(s, cum, p[:, 1])], s, cum


def count_events(pts, tree, s, tol, merge):
    """Points (OSM nodes) within tol of the route, merged when closer than `merge` along it."""
    if len(pts) == 0:
        return 0
    d, i = tree.query(pts, distance_upper_bound=tol)
    along = np.sort(s[i[np.isfinite(d)]])
    if len(along) == 0:
        return 0
    return int(1 + np.sum(np.diff(along) > merge))


def extra_km(feature, cum, codes, key):
    ex = feature["properties"]["extras"].get(key, {}).get("values", [])
    km = 0.0
    for a, b, v in ex:
        if (v in codes) if key == "waytypes" else (v not in codes):
            km += (cum[min(b, len(cum) - 1)] - cum[a]) / 1000
    return km


def main():
    rail, junc, bpts, spts = read_osm()
    btree, stree = cKDTree(bpts), cKDTree(spts)
    rows = []
    for key in G.CORRIDORS:
        g = json.loads(G.GEOJSON[key].read_text())
        for i, f in enumerate(g["features"]):
            rid = f"R{i + 1}" if key == "lviv_kyiv" else f"{key}_R{i + 1}"
            coords = f["geometry"]["coordinates"]
            P, s, cum = densify(coords)
            tree = cKDTree(P)
            on_bridge = btree.query(P, distance_upper_bound=4.0)[0] < np.inf
            runs = np.flatnonzero(on_bridge)
            n_br = 0 if len(runs) == 0 else int(1 + np.sum(np.diff(s[runs]) > 500))
            in_settle = stree.query(P, distance_upper_bound=4.0)[0] < np.inf
            summ = f["properties"]["summary"]
            rows.append({"corridor": key, "route": rid, "T_h": summ["duration"] / 3600,
                         "D_km": summ["distance"] / 1000,
                         "rail": count_events(rail, tree, s, 8.0, 100.0),
                         "junction": count_events(junc, tree, s, 8.0, 30.0),
                         "settle_km": float(in_settle.sum() * STEP / 1000),
                         "bridges": n_br,
                         "bridge_km": float(on_bridge.sum() * STEP / 1000),
                         "poor_km": extra_km(f, cum, GOOD_SURFACE, "surface")})
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    C.P3.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.P3 / "infocriteria.csv", index=False)
    # validation: routes_efficiency_calc/main.py (Overpass-based, 10 Lviv-Kyiv routes, same order)
    ref = np.array([
        [572821, 30856, 6, 250, 46, 67], [540141, 29209, 1, 323, 63, 66], [586978, 31440, 2, 362, 48, 65],
        [561111, 32317, 2, 301, 57, 64], [625541, 33927, 2, 419, 59, 67], [593905, 33108, 7, 281, 36, 64],
        [572822, 30856, 6, 250, 46, 67], [562036, 31503, 5, 220, 58, 51], [551221, 30407, 5, 419, 64, 56],
        [572192, 33514, 6, 397, 58, 54]])
    quality = np.array([2.63, 3.07, 2.45, 2.98, 2.60, 2.36, 2.63, 2.59, 3.05, 2.96])
    lk = df[df.corridor == "lviv_kyiv"].reset_index(drop=True)
    from scipy.stats import spearmanr
    val = {name: {"ours": lk[col].tolist(), "ref": ref[:, j].tolist(),
                  "spearman": float(spearmanr(lk[col], ref[:, j]).correlation)}
           for name, col, j in [("rail", "rail", 2), ("junction_vs_crossroads", "junction", 3),
                                ("settle_km_vs_settlements", "settle_km", 4), ("bridges", "bridges", 5)]}
    val["poor_km_vs_quality_index"] = {"ours": lk["poor_km"].tolist(), "ref": quality.tolist(),
                                       "spearman": float(-spearmanr(lk["poor_km"], quality).correlation)}
    (C.P3 / "infocriteria_validation.json").write_text(json.dumps(val, indent=1))
    print(json.dumps({k: round(v["spearman"], 2) for k, v in val.items()}))


if __name__ == "__main__":
    main()
