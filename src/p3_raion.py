"""
Paper 3, raion-level geography (post-2020 raions, 126 in the 24 oblasts outside Crimea).

Boundaries: data/geo/ukraine_raions_osm_20261001.geojson, extracted from the Geofabrik
Ukraine extract of 2026-10-01 (OSM boundary=administrative, admin_level=6), simplified to 0.0003 deg.
Region units are "<oblast>|<raion>" (the oblast part comes from the same geoBoundaries ADM1 polygons as in
the paper, so every oblast-level result is unchanged); the city of Kyiv is the unit "м. Київ|".

Presence of a unit in hour u: an oblast-level alert of its oblast OR a raion-level alert of that raion
(optionally also hromada-level alerts, which are counted for their raion).
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from shapely.geometry import Point, shape
from shapely.prepared import prep

import p3_core as C
import p3_geo as G

RAIONS = G.ROOT / "data" / "geo" / "ukraine_raions_osm_20261001.geojson"
OSM2LOG_OBL = {"Київ": "м. Київ"}
RENAMED = {  # names in the alert log -> current OSM names (2024 renaming)
    "Красноградський район": "Берестинський район",
    "Червоноградський район": "Шептицький район",
    "Володимир-Волинський район": "Володимирський район",
}
SEP = "|"


def norm(s):
    s = re.sub(r"[’'ʼ`]", "'", str(s)).strip()
    return RENAMED.get(s, s)


def unit(oblast, raion=""):
    return f"{oblast}{SEP}{raion}"


def oblast_of(u):
    return u.split(SEP)[0]


@lru_cache(maxsize=1)
def raion_polygons():
    """{oblast: [(unit, geom, prepared), ...]} for the raions of every oblast."""
    g = json.loads(RAIONS.read_text())
    out = {}
    for f in g["features"]:
        p = f["properties"]
        if not p.get("raion"):
            continue
        obl = OSM2LOG_OBL.get(p["oblast"], p["oblast"])
        geom = shape(f["geometry"]).buffer(0)
        out.setdefault(obl, []).append((unit(obl, norm(p["raion"])), geom, prep(geom)))
    return out


def raion_of(oblast, lon, lat):
    """Raion unit of a point already assigned to `oblast` (nearest raion of that oblast if the point
    falls into a gap between the two boundary sources)."""
    polys = raion_polygons().get(oblast)
    if not polys:                       # city of Kyiv, Sevastopol, ...
        return unit(oblast)
    pt = Point(lon, lat)
    for u, g, pg in polys:
        if pg.contains(pt):
            return u
    return min(polys, key=lambda x: x[1].distance(pt))[0]


def raion_of_vec(oblasts, x, y):
    """Vectorised raion assignment: oblasts (array of names), x, y arrays."""
    out = np.empty(len(x), dtype=object)
    for obl in np.unique(oblasts):
        m = np.where(oblasts == obl)[0]
        polys = raion_polygons().get(obl)
        if not polys:
            out[m] = unit(obl)
            continue
        lab = np.full(len(m), -1)
        for k, (_, geom, _) in enumerate(polys):
            free = lab == -1
            if not free.any():
                break
            hit = shapely.contains_xy(geom, x[m][free], y[m][free])
            lab[np.where(free)[0][hit]] = k
        rest = np.where(lab == -1)[0]
        if len(rest):
            pts = shapely.points(x[m][rest], y[m][rest])
            d = np.stack([shapely.distance(p[1], pts) for p in polys])
            lab[rest] = d.argmin(0)
        out[m] = [polys[k][0] for k in lab]
    return out


def all_units(oblasts):
    us = []
    for o in oblasts:
        polys = raion_polygons().get(o)
        us += [p[0] for p in polys] if polys else [unit(o)]
    return us


def read_log(path=None):
    log = pd.read_csv(path or C.LOG).drop_duplicates()
    sw = log["oblast"].str.endswith("район") & log["raion"].fillna("").str.endswith("область")
    log.loc[sw, ["oblast", "raion"]] = log.loc[sw, ["raion", "oblast"]].values
    log["raion"] = log["raion"].map(lambda s: norm(s) if isinstance(s, str) else s)
    return log


def presence_units(units, levels=("oblast", "raion"), log=None, start=None, end=None):
    """H x U presence for raion units: oblast-level alert of the oblast OR alert of the raion at `levels`."""
    start = start or C.DATA_START
    end = end or C.TEST_END + pd.Timedelta(hours=C.K_MAX + C.W_LEGACY + 2)
    full = pd.date_range(start, end, freq="h", tz="UTC")
    log = read_log() if log is None else log
    log = log[log["level"].isin(levels)].copy()
    log["s"] = pd.to_datetime(log["started_at"], utc=True, errors="coerce", format="ISO8601").dt.floor("h")
    log["f"] = (pd.to_datetime(log["finished_at"], utc=True, errors="coerce", format="ISO8601")
                - pd.Timedelta(seconds=1)).dt.floor("h")
    log = log.dropna(subset=["s", "f"])
    log = log[(log["f"] >= full[0]) & (log["s"] <= full[-1])]
    col = {u: j for j, u in enumerate(units)}
    by_obl = {}
    for u in units:
        by_obl.setdefault(oblast_of(u), []).append(col[u])
    A = np.zeros((len(full), len(units)), dtype=np.int8)
    pos = pd.Series(np.arange(len(full)), index=full)
    a = pos[log["s"].clip(lower=full[0])].values
    b = pos[log["f"].clip(upper=full[-1])].values
    for k, (lev, o, r) in enumerate(zip(log["level"], log["oblast"], log["raion"])):
        if lev == "oblast":
            cols = by_obl.get(o, [])
        else:
            j = col.get(unit(o, r))
            cols = [j] if j is not None else []
        for j in cols:
            A[a[k]:b[k] + 1, j] = 1
    return pd.DataFrame(A, index=full, columns=units)


# ---------------------------------------------------------------- corridor routes (ORS geometries)
def raion_hour_bins(feature):
    """(hour, unit, metres) bins of an ORS route, with the oblast part as in G.hour_bins."""
    coords, seg_d, dur, cum_d, cum_t = G.point_times(feature)
    mid_t = (cum_t[:-1] + cum_t[1:]) / 2
    xy = np.array([((coords[i][0] + coords[i + 1][0]) / 2, (coords[i][1] + coords[i + 1][1]) / 2)
                   for i in range(len(coords) - 1)])
    obl = np.array([G.locate(x, y) for x, y in xy], dtype=object)
    un = raion_of_vec(obl, xy[:, 0], xy[:, 1])
    out = {}
    for i in range(len(seg_d)):
        key = (int(mid_t[i] // 3600), un[i])
        out[key] = out.get(key, 0.0) + float(seg_d[i])
    return [(j, u, d) for (j, u), d in out.items()]


def corridor_routes_raion(key):
    """Routes of a corridor with raion-level hour bins (same order and ids as G.load_corridor_geo)."""
    g = json.loads(G.GEOJSON[key].read_text())
    base = G.load_corridor_geo(key)
    out = []
    for r, f in zip(base, g["features"]):
        hb = raion_hour_bins(f)
        rr = dict(r)
        rr["hour_bins"] = hb
        agg = {}
        for _, u, d in hb:
            agg[u] = agg.get(u, 0.0) + d
        rr["oblast_segments"] = [{"oblast": u, "distance_m": d} for u, d in agg.items()]
        rr["traversal"] = [{"oblast": u, "distance_m": d, "t_enter_s": 0.0, "t_exit_s": 0.0} for u, d in agg.items()]
        out.append(rr)
    return out
