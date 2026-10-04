"""
Paper 3, transfer experiment, step 1: a CONUS road graph with state profiles in the format of
p3_graph_candidates (nodes.csv, edges.csv with profile_json [[region, length_m, time_s], ...], meta.json).

Input : Natural Earth 1:10m roads, admin-1 states, populated places (data/transfer/ne_10m_*; .shp/.shx/.dbf/.prj
        from https://github.com/nvkelso/natural-earth-vector, folder 10m_cultural)
Output: results/p3v2/transfer/roadgraph_us/
Speeds: expressway 100 km/h, other highways 80 km/h (heavy-vehicle planning speeds).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point
from shapely.ops import unary_union

sys.path.insert(0, str(Path(__file__).parent))
import p3_core as C  # noqa: E402

DATA = C.DATA / "transfer"
OUT = C.P3 / "transfer" / "roadgraph_us"
CRS = "EPSG:5070"            # CONUS Albers equal area (metres)
SNAP_M = 500.0               # merge line ends closer than this (generalised map data)
STEP_M = 1000.0              # densification step for region assignment
CONUS = {"AK", "HI"}
CITY_POP = 600_000


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    st = gpd.read_file(DATA / "ne_10m_admin_1_states_provinces.shp")
    st = st[(st.adm0_a3 == "USA") & ~st.postal.isin(CONUS)][["postal", "name", "geometry"]].to_crs(CRS)
    rd = gpd.read_file(DATA / "ne_10m_roads.shp")
    rd = rd[rd.sov_a3 == "USA"].to_crs(CRS)
    rd = rd[rd.intersects(st.union_all().buffer(20_000))]
    rd = rd.explode(index_parts=False)
    speed = np.where(rd.expressway.values == 1, 100.0, 80.0)
    # node the network: split every line at intersections, then give each piece the speed of the
    # original line it lies on
    src = gpd.GeoDataFrame({"spd": speed}, geometry=rd.geometry.values, crs=CRS)
    allu = unary_union(list(src.geometry.values))
    segs = list(getattr(allu, "geoms", [allu]))
    mid = gpd.GeoDataFrame(geometry=[s.interpolate(0.5, normalized=True) for s in segs], crs=CRS)
    near = gpd.sjoin_nearest(mid, src, how="left", max_distance=50)
    near = near.groupby(level=0)["spd"].max()
    seg_spd = near.reindex(range(len(segs))).fillna(80.0).values
    # snap end points
    ends = np.array([[s.coords[0], s.coords[-1]] for s in segs]).reshape(-1, 2)
    tree = cKDTree(ends)
    lab = np.arange(len(ends))
    for i, j in tree.query_pairs(SNAP_M):
        a, b = lab[i], lab[j]
        while lab[a] != a: a = lab[a]
        while lab[b] != b: b = lab[b]
        if a != b: lab[max(a, b)] = min(a, b)
    for i in range(len(lab)):
        r = i
        while lab[r] != r: r = lab[r]
        lab[i] = r
    uniq, node_of = np.unique(lab, return_inverse=True)
    node_xy = ends[uniq]
    node_of = node_of.reshape(-1, 2)
    # region profile of each segment (all densified midpoints joined to states at once)
    keep, mids, segid = [], [], []
    for k, sg in enumerate(segs):
        if node_of[k, 0] == node_of[k, 1]:
            continue
        L = sg.length
        n = max(2, int(np.ceil(L / STEP_M)) + 1)
        xs = np.linspace(0, L, n)
        mids += [sg.interpolate((a + b) / 2) for a, b in zip(xs, xs[1:])]
        segid += [k] * (n - 1)
        keep.append((k, L, n))
    j = gpd.sjoin_nearest(gpd.GeoDataFrame({"seg": segid}, geometry=mids, crs=CRS), st[["postal", "geometry"]],
                          how="left")
    j = j[~j.index.duplicated()]
    reg_by_seg = j.groupby("seg")["postal"].apply(list)
    rows = []
    for k, L, n in keep:
        dl = L / (n - 1)
        prof = []
        for r_ in reg_by_seg[k]:
            if prof and prof[-1][0] == r_:
                prof[-1][1] += dl
            else:
                prof.append([r_, dl])
        v_ms = seg_spd[k] / 3.6
        rows.append((int(node_of[k, 0]), int(node_of[k, 1]), L, L / v_ms, [[r_, d, d / v_ms] for r_, d in prof]))
    g = nx.Graph()
    for u, v, L, t, prof in rows:
        if g.has_edge(u, v) and g.edges[u, v]["time_s"] <= t:
            continue
        g.add_edge(u, v, length_m=L, time_s=t, profile=prof)
    comp = max(nx.connected_components(g), key=len)
    g = g.subgraph(comp).copy()
    xy = gpd.GeoSeries([Point(*node_xy[n]) for n in g.nodes], crs=CRS).to_crs(4326)
    N = pd.DataFrame({"node_id": list(g.nodes), "lon": xy.x.values, "lat": xy.y.values, "city": ""})
    # cities
    pp = gpd.read_file(DATA / "ne_10m_populated_places_simple.shp")
    pp = pp[(pp.adm0_a3 == "USA") & (pp.pop_max >= CITY_POP)].to_crs(CRS)
    pp = pp[pp.within(st.union_all())].sort_values("pop_max", ascending=False).drop_duplicates("name")
    nodes = np.array(list(g.nodes)); nt = cKDTree(node_xy[nodes])
    cities = {}
    for r in pp.itertuples():
        d, i = nt.query([r.geometry.x, r.geometry.y])
        if d < 15_000:
            cities[r.name] = int(nodes[i])
    N.loc[N.node_id.isin(cities.values()), "city"] = N.node_id.map({v: k for k, v in cities.items()})
    N.to_csv(OUT / "nodes.csv", index=False)
    E = pd.DataFrame([{"u": u, "v": v, "length_m": d["length_m"], "time_s": d["time_s"],
                       "profile_json": json.dumps(d["profile"])} for u, v, d in g.edges(data=True)])
    E.to_csv(OUT / "edges.csv", index=False)
    meta = {"source": "Natural Earth 1:10m roads", "nodes": g.number_of_nodes(), "edges": g.number_of_edges(),
            "total_km": float(E.length_m.sum() / 1000), "speeds_kmh": {"expressway": 100, "other": 80},
            "snap_m": SNAP_M, "regions": sorted(st.postal.tolist()), "cities": cities}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1))
    print({k: meta[k] for k in ("nodes", "edges", "total_km")}, len(cities), "cities")


if __name__ == "__main__":
    main()
