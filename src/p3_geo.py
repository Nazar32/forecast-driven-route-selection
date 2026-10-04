"""
Paper 3 geometry utilities: route polylines -> ordered, time-stamped oblast traversals.

For every ORS route geometry we compute, point by point:
  * cumulative distance (haversine),
  * cumulative travel time (ORS step durations spread over the step's points
    proportionally to distance),
  * the oblast containing each sub-segment midpoint (geoBoundaries ADM1, which,
    unlike OSM admin_level=4, contains the city of Kyiv as its own unit).

Output per route: ordered list of dicts
  {oblast, distance_m, t_enter_s, t_exit_s}
merged over consecutive sub-segments in the same oblast. Distances are used for
the distance-weighted scores; entry/exit times for the time-resolved exposure metric.

Boundary source: geoBoundaries gbOpen UKR ADM1 (CC BY 4.0), vendored in
data/geo/.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
from shapely.geometry import shape, Point
from shapely.prepared import prep

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                             # repository root
REC = ROOT / "data" / "corridors"              # OpenRouteService corridor routes
GEO = ROOT / "data" / "geo" / "geoBoundaries-UKR-ADM1_simplified.geojson"

EN2UA = {
    "Kherson Oblast": "Херсонська область", "Volyn Oblast": "Волинська область",
    "Rivne Oblast": "Рівненська область", "Zhytomyr Oblast": "Житомирська область",
    "Kyiv Oblast": "Київська область", "Chernihiv Oblast": "Чернігівська область",
    "Sumy Oblast": "Сумська область", "Kharkiv Oblast": "Харківська область",
    "Luhansk Oblast": "Луганська область", "Donetsk Oblast": "Донецька область",
    "Zaporizhia Oblast": "Запорізька область", "Lviv Oblast": "Львівська область",
    "Ivano-Frankivsk Oblast": "Івано-Франківська область",
    "Zakarpattia Oblast": "Закарпатська область", "Ternopil Oblast": "Тернопільська область",
    "Chernivtsi Oblast": "Чернівецька область", "Odessa Oblast": "Одеська область",
    "Mykolaiv Oblast": "Миколаївська область",
    "Autonomous Republic of Crimea": "Автономна Республіка Крим",
    "Vinnytsia Oblast": "Вінницька область", "Khmelnytskyi Oblast": "Хмельницька область",
    "Cherkasy Oblast": "Черкаська область", "Poltava Oblast": "Полтавська область",
    "Dnipropetrovsk Oblast": "Дніпропетровська область",
    "Kirovohrad Oblast": "Кіровоградська область", "Kyiv": "м. Київ", "Sevastopol": "Севастополь",
}

GEOJSON = {
    "lviv_kyiv": REC / "lviv_kyiv.json",
    "odesa_kyiv": REC / "odesa_kyiv.json",
    "kharkiv_lviv": REC / "kharkiv_lviv.json",
    "dnipro_kyiv": REC / "dnipro_kyiv.json",
}
CORRIDORS = list(GEOJSON)


@lru_cache(maxsize=1)
def oblast_polygons():
    g = json.loads(GEO.read_text())
    out = []
    for f in g["features"]:
        name = EN2UA[f["properties"]["shapeName"]]
        geom = shape(f["geometry"]).buffer(0)
        out.append((name, geom, prep(geom)))
    # small units first so the city of Kyiv wins over the surrounding oblast
    out.sort(key=lambda x: x[1].area)
    return out


def locate(lon, lat, last=None):
    p = Point(lon, lat)
    polys = oblast_polygons()
    if last is not None:
        for name, g, pg in polys:
            if name == last and pg.contains(p):
                # keep last unless a smaller unit (Kyiv city) contains the point
                break
    for name, g, pg in polys:
        if pg.contains(p):
            return name
    # outside every polygon (simplification gaps at borders): nearest
    return min(polys, key=lambda x: x[1].distance(p))[0]


def haversine(a, b):
    lon1, lat1, lon2, lat2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    d = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371008.8 * math.asin(math.sqrt(d))


def point_times(feature):
    """Cumulative distance and time at each coordinate of an ORS route feature."""
    coords = feature["geometry"]["coordinates"]
    n = len(coords)
    seg_d = np.array([haversine(coords[i], coords[i + 1]) for i in range(n - 1)])
    dur = np.zeros(n - 1)
    for seg in feature["properties"]["segments"]:
        for st in seg["steps"]:
            i, j = st["way_points"]
            if j <= i:
                continue
            w = seg_d[i:j]
            tot = w.sum()
            dur[i:j] += st["duration"] * (w / tot if tot > 0 else np.full(j - i, 1.0 / (j - i)))
    cum_d = np.concatenate([[0.0], np.cumsum(seg_d)])
    cum_t = np.concatenate([[0.0], np.cumsum(dur)])
    return coords, seg_d, dur, cum_d, cum_t


def traversal(feature, scale_to_summary=True):
    coords, seg_d, dur, cum_d, cum_t = point_times(feature)
    summ = feature["properties"].get("summary", {})
    ks = summ.get("distance", cum_d[-1]) / cum_d[-1] if (scale_to_summary and cum_d[-1] > 0) else 1.0
    out = []
    cur = None
    for i in range(len(coords) - 1):
        mid = ((coords[i][0] + coords[i + 1][0]) / 2, (coords[i][1] + coords[i + 1][1]) / 2)
        o = locate(*mid)
        if cur is None or o != cur["oblast"]:
            cur = {"oblast": o, "distance_m": 0.0, "t_enter_s": float(cum_t[i]), "t_exit_s": float(cum_t[i])}
            out.append(cur)
        cur["distance_m"] += seg_d[i] * ks
        cur["t_exit_s"] = float(cum_t[i + 1])
    # merge tiny border flickers (< 2 km) into the neighbour by re-merging equal names
    merged = []
    for s in out:
        if merged and s["distance_m"] < 2000 and len(merged) >= 1:
            merged[-1]["distance_m"] += s["distance_m"]
            merged[-1]["t_exit_s"] = s["t_exit_s"]
            continue
        if merged and merged[-1]["oblast"] == s["oblast"]:
            merged[-1]["distance_m"] += s["distance_m"]
            merged[-1]["t_exit_s"] = s["t_exit_s"]
        else:
            merged.append(dict(s))
    return merged


def load_corridor_geo(key):
    """Routes of a corridor with D, T and ordered time-stamped traversal."""
    g = json.loads(GEOJSON[key].read_text())
    routes = []
    for i, f in enumerate(g["features"]):
        rid = f"R{i + 1}" if key == "lviv_kyiv" else f"{key}_R{i + 1}"
        tr = traversal(f)
        s = f["properties"]["summary"]
        routes.append({"id": rid, "D": float(s["distance"]), "T": float(s["duration"]),
                       "traversal": tr, "hour_bins": hour_bins(f),
                       "oblast_segments": aggregate(tr)})
    return routes


def hour_bins(feature):
    """Exact per-(hour-after-departure, oblast) distance, from every geometry sub-segment
    (sub-segments are assigned to the hour that contains their mid time)."""
    coords, seg_d, dur, cum_d, cum_t = point_times(feature)
    mid_t = (cum_t[:-1] + cum_t[1:]) / 2
    out = {}
    last = None
    for i in range(len(coords) - 1):
        mid = ((coords[i][0] + coords[i + 1][0]) / 2, (coords[i][1] + coords[i + 1][1]) / 2)
        o = locate(*mid)
        key = (int(mid_t[i] // 3600), o)
        out[key] = out.get(key, 0.0) + float(seg_d[i])
    return [(j, o, d) for (j, o), d in out.items()]


def aggregate(tr):
    agg = {}
    for s in tr:
        agg[s["oblast"]] = agg.get(s["oblast"], 0.0) + s["distance_m"]
    return [{"oblast": k, "distance_m": v} for k, v in agg.items()]


if __name__ == "__main__":
    import sys
    for key in CORRIDORS:
        for r in load_corridor_geo(key):
            print(r["id"], round(r["D"] / 1000), "km", round(r["T"] / 3600, 2), "h",
                  [(s["oblast"][:6], round(s["distance_m"] / 1000), round(s["t_enter_s"] / 3600, 1)) for s in r["traversal"]])
