"""
Paper 3 / experiment (d), step 2: graph-native candidate generation on the road graph.

Methods (all on the junction-level road graph from p3_roadgraph.py):
  yen_raw(K)          K shortest loopless paths by travel time (Yen 1971)
  yen_lo(K, theta)    Yen with LIMITED OVERLAP: a path is kept only if its shared length
                      with every already-kept path is < theta * its own length, and its
                      time is within a stretch bound (<= STRETCH * fastest).  This is the
                      standard remedy for Yen's near-duplicate paths (k-shortest paths with
                      limited overlap, Chondrogiannis et al., VLDB J. 2020).
Also provides path -> time-stamped oblast traversal so any generated path can be scored
with exactly the same exposure metrics (p3_core) as the ORS corridor routes.
"""
from __future__ import annotations

import itertools
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

import networkx as nx
import pandas as pd

HERE = Path(__file__).resolve().parent
# graph directory: corridor graph by default; set P3_GRAPH_DIR (e.g. results/p3/roadgraph_osm_primary)
GDIR = Path(os.environ.get("P3_GRAPH_DIR", HERE.parent / "results" / "p3" / "roadgraph"))
if not GDIR.is_absolute():
    GDIR = HERE.parent / GDIR
STRETCH = 1.5
THETA = 0.7
MAX_ENUM = 400          # Yen paths examined per OD before giving up


@lru_cache(maxsize=1)
def load_graph():
    N = pd.read_csv(GDIR / "nodes.csv").fillna({"city": ""})
    E = pd.read_csv(GDIR / "edges.csv")
    g = nx.Graph()
    for r in N.itertuples():
        g.add_node(int(r.node_id), lon=r.lon, lat=r.lat, city=r.city)
    for r in E.itertuples():
        g.add_edge(int(r.u), int(r.v), length_m=r.length_m, time_s=r.time_s,
                   profile=json.loads(r.profile_json), fwd=(int(r.u), int(r.v)))
    meta = json.loads((GDIR / "meta.json").read_text())
    return g, meta["cities"]


def oriented_profile(g, a, b):
    e = g.edges[a, b]
    return e["profile"] if e["fwd"] == (a, b) else e["profile"][::-1]


def path_time(g, p):
    return sum(g.edges[a, b]["time_s"] for a, b in zip(p, p[1:]))


def path_len(g, p):
    return sum(g.edges[a, b]["length_m"] for a, b in zip(p, p[1:]))


def path_route(g, p, rid):
    """Path -> route dict compatible with p3_core (traversal with entry/exit times)."""
    trav, t = [], 0.0
    bins = {}
    for a, b in zip(p, p[1:]):
        for o, d, dt in oriented_profile(g, a, b):
            key = (int((t + dt / 2) // 3600), o)
            bins[key] = bins.get(key, 0.0) + d
            if trav and trav[-1]["oblast"] == o:
                trav[-1]["distance_m"] += d; trav[-1]["t_exit_s"] = t + dt
            else:
                trav.append({"oblast": o, "distance_m": d, "t_enter_s": t, "t_exit_s": t + dt})
            t += dt
    agg = {}
    for s in trav:
        agg[s["oblast"]] = agg.get(s["oblast"], 0.0) + s["distance_m"]
    return {"id": rid, "D": path_len(g, p), "T": t, "traversal": trav, "path": list(p),
            "hour_bins": [(j, o, d) for (j, o), d in bins.items()],
            "oblast_segments": [{"oblast": k, "distance_m": v} for k, v in agg.items()]}


def _edge_set(p):
    return {frozenset(e) for e in zip(p, p[1:])}


def shared_len(g, p, q):
    common = _edge_set(p) & _edge_set(q)
    return sum(g.edges[tuple(e)]["length_m"] for e in common)


def yen_raw(g, s, t, K):
    return list(itertools.islice(nx.shortest_simple_paths(g, s, t, weight="time_s"), K))


def yen_lo(g, s, t, K, theta=THETA, stretch=STRETCH):
    kept = []
    tmin = None
    for i, p in enumerate(nx.shortest_simple_paths(g, s, t, weight="time_s")):
        if i >= MAX_ENUM or len(kept) >= K:
            break
        tp = path_time(g, p)
        if tmin is None:
            tmin = tp
        if tp > stretch * tmin:
            break
        L = path_len(g, p)
        if all(shared_len(g, p, q) < theta * L for q in kept):
            kept.append(p)
    return kept


# ---------------------------------------------------------------- penalty method (large graphs)
class Csr:
    """Undirected graph as a scipy CSR with edge-id bookkeeping (fast repeated Dijkstra)."""

    def __init__(self, g):
        from scipy.sparse import csr_matrix
        self.nodes = list(g.nodes)
        self.ni = {n: i for i, n in enumerate(self.nodes)}
        self.E = list(g.edges)
        self.t = np.array([g.edges[e]["time_s"] for e in self.E])
        self.L = np.array([g.edges[e]["length_m"] for e in self.E])
        u = np.array([self.ni[a] for a, b in self.E]); v = np.array([self.ni[b] for a, b in self.E])
        self.rows, self.cols = np.r_[u, v], np.r_[v, u]
        self.eid = {}
        for k, (a, b) in enumerate(self.E):
            self.eid[(self.ni[a], self.ni[b])] = k; self.eid[(self.ni[b], self.ni[a])] = k
        self.n = len(self.nodes)
        # permutation so that data can be refreshed without rebuilding the structure
        m = csr_matrix((np.r_[np.arange(len(self.E)), np.arange(len(self.E))] + 1.0,
                        (self.rows, self.cols)), shape=(self.n, self.n))
        m.sum_duplicates()
        self.struct = m
        self.perm = (m.data - 1).astype(int)

    def matrix(self, w):
        m = self.struct.copy()
        m.data = w[self.perm]
        return m

    def path(self, w, s, t):
        from scipy.sparse.csgraph import dijkstra
        _, pred = dijkstra(self.matrix(w), directed=True, indices=self.ni[s], return_predecessors=True)
        x, out = self.ni[t], []
        if pred[x] < 0 and x != self.ni[s]:
            return None
        while x != self.ni[s]:
            out.append(x); x = pred[x]
        out.append(self.ni[s])
        return out[::-1]                                  # node indices

    def edges_of(self, p):
        return [self.eid[(a, b)] for a, b in zip(p, p[1:])]


def penalty_lo(g, s, t, K, theta=THETA, stretch=STRETCH, factor=1.2, max_iter=120, csr=None):
    """Iterative penalty method (de la Barra et al. 1993; Bader et al. 2011) with the same
    limited-overlap and stretch filters as yen_lo.  Returns node-id paths."""
    c = csr or Csr(g)
    pen = np.ones(len(c.E))
    kept, kept_e = [], []
    tmin = None
    for _ in range(max_iter):
        p = c.path(c.t * pen, s, t)
        if p is None:
            break
        ep = c.edges_of(p)
        tp = c.t[ep].sum()
        if tmin is None:
            tmin = tp
        pen[ep] *= factor
        if tp > stretch * tmin:
            continue
        Lp = c.L[ep].sum()
        es = set(ep)
        if all(c.L[list(es & q)].sum() < theta * Lp for q in kept_e):
            kept.append([c.nodes[i] for i in p]); kept_e.append(es)
            if len(kept) >= K:
                break
    return kept
