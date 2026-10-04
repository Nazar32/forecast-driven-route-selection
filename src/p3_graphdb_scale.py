"""
Paper 3 / experiment (d), step 5: WHEN does the graph database pay off?  Scaling benchmark.

Pre-registered hypothesis (written before any run, 2026-09-26):
  H1  On a small graph and for a single query per update, the in-memory reference and
      PostgreSQL+pgRouting are faster than Neo4j+GDS (already observed in p3_graphdb_bench).
  H2  On a large graph and for many queries per hourly update (fleet scenario), Neo4j+GDS
      becomes competitive or faster than pgRouting, because GDS answers every query from one
      in-memory projection while pgRouting rebuilds its graph from SQL on every call.
  H3  For local neighbourhood queries (k-hop), native traversal (GDS BFS) beats the
      relational recursive CTE on a large graph.
  Whatever the outcome, all rows are reported; PostgreSQL is given its best form
  (batched many-to-many pgr_dijkstra, indexed arc table), and the in-memory row is kept.

Scenarios (per graph size; graphs: corridor, OSM primary, OSM secondary):
  S1 fleet cycle   one hourly update (write 25 alert states, re-weight every edge in the DB:
                   w_e = time_e[h] + MU * sum_pieces km * p8(oblast) / 100, p8 = mean 1..8 h
                   lead forecast) + Q shortest-path queries, Q in {1, 10, 100, 1000}.
                   Timed: update + all queries.  Variants: per-call and batched.
  S2 static Yen    K = 10 shortest loopless paths for Q OD pairs (Q in {1, 10, 100}); the graph /
                   projection is built once (setup time reported separately).
  S3 k-hop         number of junctions within k hops (k = 3, 6) of Q start nodes (Q = 100).
Correctness: every backend's answers are compared with the in-memory reference written by the
memory run (scale_ref_<graph>.json); mismatches are counted and reported.

Run (from project root; same flags as p3_graphdb_bench.py):
  python src/p3_graphdb_scale.py --graph roadgraph --backend memory
  python src/p3_graphdb_scale.py --graph roadgraph_osm_primary --backend postgres --pg "..."
  python src/p3_graphdb_scale.py --graph roadgraph_osm_primary --backend neo4j --neo4j bolt://127.0.0.1:7688
Outputs: results/p3/scale_<graph>_<backend>.json
"""
from __future__ import annotations

import argparse
import io
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

RES = Path(__file__).resolve().parent.parent / "results" / "p3"
MU = 16.0
K_YEN = 10
SEED = 7
QS_S1 = [1, 10, 100, 1000]
QS_S2 = [1, 10, 100]
Q_S3, KS_S3 = 100, [3, 6]
REL_TOL = 1e-6


# ------------------------------------------------------------------ data
def load(graph):
    gdir = RES / graph
    N = pd.read_csv(gdir / "nodes.csv")
    E = pd.read_csv(gdir / "edges.csv")
    oblasts = sorted({p[0] for pj in E["profile_json"] for p in json.loads(pj)})
    oi = {o: i for i, o in enumerate(oblasts)}
    pe, po, pk = [], [], []
    for eid, pj in enumerate(E["profile_json"]):
        for o, d, _ in json.loads(pj):
            pe.append(eid); po.append(oi[o]); pk.append(d / 1000.0)
    pieces = pd.DataFrame({"edge_id": pe, "oidx": po, "km": pk})
    dp = pd.read_parquet(RES / "direct_probs.parquet")
    dp["hour"] = pd.to_datetime(dp["hour"], utc=True)
    dp = dp[(dp["hour"] >= pd.Timestamp("2025-03-31 18:00", tz="UTC")) & dp["k"].between(1, 8)]
    hrs = sorted(dp["hour"].unique())[::97][:5]
    dp = dp[dp["hour"].isin(hrs)]
    p8 = dp.groupby(["hour", "oblast"])["p"].mean().unstack().reindex(columns=oblasts).fillna(0.0)
    return N, E, oblasts, pieces, p8.values                                  # alerts: H x O


def sample_pairs(n_nodes, q, salt=0):
    rng = np.random.default_rng(SEED + 7919 * salt + q)
    s = rng.integers(0, n_nodes, q); t = rng.integers(0, n_nodes, q)
    t = np.where(t == s, (t + 1) % n_nodes, t)
    return [(int(a), int(b)) for a, b in zip(s, t)]


def timed(fn, *a, **k):
    t0 = time.perf_counter(); out = fn(*a, **k)
    return out, (time.perf_counter() - t0) * 1000.0


def stats(ms):
    return {"n": len(ms), "median_ms": statistics.median(ms), "min_ms": min(ms), "max_ms": max(ms)}


# ------------------------------------------------------------------ backends
class Memory:
    name = "memory"

    def __init__(self, N, E, oblasts, pieces):
        from scipy.sparse import csr_matrix
        self.u, self.v, self.t = E["u"].values, E["v"].values, E["time_s"].values / 3600.0
        self.n = len(N)
        self.pe, self.po, self.pk = pieces.edge_id.values, pieces.oidx.values, pieces.km.values
        self.w = self.t.copy()
        self.csr = csr_matrix

    def mat(self, w):
        return self.csr((np.r_[w, w], (np.r_[self.u, self.v], np.r_[self.v, self.u])), shape=(self.n, self.n))

    def update(self, alert):
        risk = np.bincount(self.pe, weights=self.pk * alert[self.po], minlength=len(self.t))
        self.w = self.t + MU * risk / 100.0

    def sp_batch(self, pairs):
        from scipy.sparse.csgraph import dijkstra
        src = sorted({s for s, _ in pairs}); si = {s: i for i, s in enumerate(src)}
        D = dijkstra(self.mat(self.w), directed=True, indices=src)
        return [float(D[si[s], t]) for s, t in pairs]

    sp_single = lambda self, pairs: [self.sp_batch([p])[0] for p in pairs]

    def yen(self, pairs, k):
        import itertools
        import networkx as nx
        g = nx.Graph()
        for a, b, t in zip(self.u, self.v, self.t):
            g.add_edge(int(a), int(b), w=float(t))
        out = []
        for s, t in pairs:
            ps = itertools.islice(nx.shortest_simple_paths(g, s, t, weight="w"), k)
            out.append([round(sum(g[a][b]["w"] for a, b in zip(p, p[1:])), 6) for p in ps])
        return out

    def khop(self, starts, k):
        import networkx as nx
        if not hasattr(self, "_g"):
            self._g = nx.Graph(); self._g.add_edges_from(zip(self.u.tolist(), self.v.tolist()))
        return [len(nx.single_source_shortest_path_length(self._g, s, cutoff=k)) for s in starts]


class Postgres:
    name = "postgres"

    def __init__(self, dsn, N, E, oblasts, pieces):
        import psycopg2
        self.c = psycopg2.connect(dsn); self.c.autocommit = True
        cur = self.cur = self.c.cursor()
        cur.execute("CREATE EXTENSION IF NOT EXISTS pgrouting CASCADE")
        cur.execute("DROP TABLE IF EXISTS p3s_road, p3s_piece, p3s_alert, p3s_arc, p3s_q")
        cur.execute("CREATE TABLE p3s_road (id int PRIMARY KEY, source bigint, target bigint, t float8, w float8)")
        cur.execute("CREATE TABLE p3s_piece (edge_id int, oidx int, km float8)")
        cur.execute("CREATE TABLE p3s_alert (oidx int PRIMARY KEY, p float8)")
        cur.execute("CREATE TABLE p3s_arc (src bigint, dst bigint)")
        cur.execute("CREATE TABLE p3s_q (source bigint, target bigint)")
        self._copy("p3s_road", pd.DataFrame({"id": np.arange(len(E)), "source": E.u, "target": E.v,
                                             "t": E.time_s / 3600.0, "w": E.time_s / 3600.0}))
        self._copy("p3s_piece", pieces[["edge_id", "oidx", "km"]])
        self._copy("p3s_alert", pd.DataFrame({"oidx": range(len(oblasts)), "p": 0.0}))
        self._copy("p3s_arc", pd.DataFrame({"src": np.r_[E.u, E.v], "dst": np.r_[E.v, E.u]}))
        cur.execute("CREATE INDEX ON p3s_piece(edge_id)"); cur.execute("CREATE INDEX ON p3s_arc(src)")
        cur.execute("ANALYZE")

    def _copy(self, table, df):
        buf = io.StringIO(); df.to_csv(buf, index=False, header=False); buf.seek(0)
        self.cur.copy_expert(f"COPY {table} FROM STDIN WITH CSV", buf)

    def update(self, alert):
        cur = self.cur
        cur.execute("UPDATE p3s_alert a SET p = v.p FROM (SELECT unnest(%s::int[]) oidx, unnest(%s::float8[]) p) v "
                    "WHERE a.oidx = v.oidx", (list(range(len(alert))), [float(x) for x in alert]))
        cur.execute("""UPDATE p3s_road r SET w = r.t + %s * x.risk / 100.0 FROM (
                         SELECT pc.edge_id, sum(pc.km * a.p) risk FROM p3s_piece pc JOIN p3s_alert a USING (oidx)
                         GROUP BY pc.edge_id) x WHERE x.edge_id = r.id""", (MU,))

    EDGES = "SELECT id, source, target, w AS cost, w AS reverse_cost FROM p3s_road"

    def sp_single(self, pairs):
        out = []
        for s, t in pairs:
            self.cur.execute("SELECT coalesce(max(agg_cost), 0) FROM pgr_dijkstra(%s, %s, %s, directed := false) "
                             "WHERE node = %s", (self.EDGES, s, t, t))
            out.append(float(self.cur.fetchone()[0]))
        return out

    def sp_batch(self, pairs):
        self.cur.execute("TRUNCATE p3s_q")
        self._copy("p3s_q", pd.DataFrame(pairs, columns=["source", "target"]))
        self.cur.execute("SELECT start_vid, end_vid, max(agg_cost) FROM pgr_dijkstra(%s, "
                         "'SELECT source, target FROM p3s_q', directed := false) GROUP BY 1, 2", (self.EDGES,))
        res = {(int(a), int(b)): float(c) for a, b, c in self.cur.fetchall()}
        return [res.get((s, t), 0.0) for s, t in pairs]

    def yen(self, pairs, k):
        out = []
        for s, t in pairs:
            self.cur.execute("SELECT path_id, max(agg_cost) FROM pgr_KSP('SELECT id, source, target, t AS cost, "
                             "t AS reverse_cost FROM p3s_road', %s, %s, %s, directed := false) GROUP BY 1 ORDER BY 1",
                             (s, t, k))
            out.append([round(float(r[1]), 6) for r in self.cur.fetchall()])
        return out

    def khop(self, starts, k):
        out = []
        for s in starts:
            self.cur.execute("""WITH RECURSIVE r(node, d) AS (SELECT %s::bigint, 0
                                  UNION SELECT a.dst, r.d + 1 FROM r JOIN p3s_arc a ON a.src = r.node WHERE r.d < %s)
                                SELECT count(DISTINCT node) FROM r""", (s, k))
            out.append(int(self.cur.fetchone()[0]))
        return out


class Neo4j:
    name = "neo4j"

    def __init__(self, uri, user, pw, N, E, oblasts, pieces):
        from neo4j import GraphDatabase
        self.d = GraphDatabase.driver(uri, auth=(user, pw), connection_timeout=10)
        self.d.verify_connectivity()
        print("Neo4j", self.run("CALL dbms.components() YIELD versions RETURN versions[0] AS v")[0]["v"],
              "| GDS", self.run("RETURN gds.version() AS v")[0]["v"], flush=True)
        for gname in ("p3s_time", "p3s_w"):
            self.run("CALL gds.graph.drop($g, false) YIELD graphName RETURN graphName", g=gname)
        while self.run("MATCH (n:SJunction) WITH n LIMIT 20000 DETACH DELETE n RETURN count(*) AS c")[0]["c"]:
            pass
        self.run("MATCH (n:SAlertState) DELETE n")
        self.run("CREATE CONSTRAINT sj IF NOT EXISTS FOR (n:SJunction) REQUIRE n.nid IS UNIQUE")
        for i in range(0, len(N), 20000):
            self.run("UNWIND $ids AS i CREATE (:SJunction {nid: i})", ids=N.node_id.iloc[i:i + 20000].astype(int).tolist())
        grp = pieces.groupby("edge_id")
        po = grp["oidx"].apply(list).reindex(range(len(E))).tolist()
        pk = grp["km"].apply(list).reindex(range(len(E))).tolist()
        rels = [{"u": int(u), "v": int(v), "t": float(t) / 3600.0, "po": [int(x) for x in a], "pk": [float(x) for x in b]}
                for u, v, t, a, b in zip(E.u, E.v, E.time_s, po, pk)]
        for i in range(0, len(rels), 20000):
            self.run("""UNWIND $rels AS r MATCH (a:SJunction {nid: r.u}), (b:SJunction {nid: r.v})
                        CREATE (a)-[:SROAD {t: r.t, w: r.t, po: r.po, pk: r.pk}]->(b)""", rels=rels[i:i + 20000])
        self.run("UNWIND range(0, $n - 1) AS i CREATE (:SAlertState {idx: i, p: 0.0})", n=len(oblasts))
        _, self.setup_ms = timed(self.run, """MATCH (a:SJunction)-[r:SROAD]->(b:SJunction)
            WITH gds.graph.project('p3s_time', a, b, {relationshipProperties: r {.t}},
                                   {undirectedRelationshipTypes: ['*']}) AS g RETURN g.graphName""")
        self.nid_ok = True

    def run(self, cypher, **kw):
        with self.d.session() as s:
            return list(s.run(cypher, **kw))

    def update(self, alert):
        self.run("UNWIND $rows AS r MATCH (x:SAlertState {idx: r.i}) SET x.p = r.p",
                 rows=[{"i": i, "p": float(p)} for i, p in enumerate(alert)])
        self.run("""MATCH (x:SAlertState) WITH x ORDER BY x.idx WITH collect(x.p) AS P
                    MATCH ()-[r:SROAD]->()
                    SET r.w = r.t + $mu * reduce(a = 0.0, k IN range(0, size(r.pk) - 1) | a + r.pk[k] * P[r.po[k]]) / 100.0""",
                 mu=MU)
        self.run("CALL gds.graph.drop('p3s_w', false) YIELD graphName RETURN graphName")
        self.run("""MATCH (a:SJunction)-[r:SROAD]->(b:SJunction)
                    WITH gds.graph.project('p3s_w', a, b, {relationshipProperties: r {.w}},
                                           {undirectedRelationshipTypes: ['*']}) AS g RETURN g.graphName""")

    def sp_single(self, pairs):
        out = []
        for s, t in pairs:
            r = self.run("""MATCH (a:SJunction {nid: $s}), (b:SJunction {nid: $t})
                CALL gds.shortestPath.dijkstra.stream('p3s_w', {sourceNode: a, targetNode: b, relationshipWeightProperty: 'w'})
                YIELD totalCost RETURN totalCost""", s=s, t=t)
            out.append(float(r[0]["totalCost"]) if r else 0.0)
        return out

    def sp_batch(self, pairs):
        rows = self.run("""UNWIND $q AS q MATCH (a:SJunction {nid: q[0]}), (b:SJunction {nid: q[1]})
            CALL gds.shortestPath.dijkstra.stream('p3s_w', {sourceNode: a, targetNode: b, relationshipWeightProperty: 'w'})
            YIELD totalCost RETURN q[0] AS s, q[1] AS t, totalCost""", q=[list(p) for p in pairs])  # noqa
        res = {(r["s"], r["t"]): float(r["totalCost"]) for r in rows}
        return [res.get((s, t), 0.0) for s, t in pairs]

    def yen(self, pairs, k):
        out = []
        for s, t in pairs:
            rows = self.run("""MATCH (a:SJunction {nid: $s}), (b:SJunction {nid: $t})
                CALL gds.shortestPath.yens.stream('p3s_time', {sourceNode: a, targetNode: b, k: $k, relationshipWeightProperty: 't'})
                YIELD index, totalCost RETURN totalCost ORDER BY index""", s=s, t=t, k=k)
            out.append([round(float(r["totalCost"]), 6) for r in rows])
        return out

    def khop(self, starts, k):
        out = []
        for s in starts:
            r = self.run("""MATCH (a:SJunction {nid: $s})
                CALL gds.bfs.stream('p3s_time', {sourceNode: a, maxDepth: $k}) YIELD nodeIds
                RETURN size(nodeIds) AS c""", s=s, k=k)
            out.append(int(r[0]["c"]))
        return out


# ------------------------------------------------------------------ driver
def close(a, b):
    return len(a) == len(b) and all(abs(x - y) <= REL_TOL * max(1.0, abs(y)) for x, y in zip(a, b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="roadgraph")
    ap.add_argument("--backend", choices=["memory", "postgres", "neo4j"], required=True)
    ap.add_argument("--pg", default="dbname=p3 user=postgres password=postgres host=127.0.0.1 port=5433")
    ap.add_argument("--neo4j", default="bolt://127.0.0.1:7688")
    ap.add_argument("--user", default="neo4j")
    ap.add_argument("--password", default="password")
    ap.add_argument("--max-q", type=int, default=1000)
    ap.add_argument("--skip-yen", action="store_true")
    a = ap.parse_args()
    N, E, oblasts, pieces, alerts = load(a.graph)
    print(f"graph {a.graph}: {len(N)} nodes, {len(E)} edges, {len(pieces)} pieces", flush=True)
    t0 = time.perf_counter()
    be = {"memory": lambda: Memory(N, E, oblasts, pieces),
          "postgres": lambda: Postgres(a.pg, N, E, oblasts, pieces),
          "neo4j": lambda: Neo4j(a.neo4j, a.user, a.password, N, E, oblasts, pieces)}[a.backend]()
    load_s = time.perf_counter() - t0
    ref_path = RES / f"scale_ref_{a.graph}.json"
    ref = json.loads(ref_path.read_text()) if (ref_path.exists() and a.backend != "memory") else None
    newref = {}
    out = {"backend": a.backend, "graph": a.graph, "nodes": len(N), "edges": len(E), "mu": MU,
           "load_s": load_s, "S1": {}, "S2": {}, "S3": {}}
    if hasattr(be, "setup_ms"):
        out["projection_setup_ms"] = be.setup_ms
    mism = 0

    # S1 fleet cycle
    be.update(alerts[0]); be.sp_batch(sample_pairs(len(N), 2))                     # warm-up
    for q in [x for x in QS_S1 if x <= a.max_q]:
        pairs = sample_pairs(len(N), q)
        for variant in ("single", "batch"):
            if variant == "single" and q > 100 and a.backend != "memory":
                continue                                                           # per-call x1000: see batch
            cyc, upd = [], []
            for h in range(min(3, len(alerts))):
                (_, mu_ms) = timed(be.update, alerts[h])
                res, q_ms = timed(getattr(be, f"sp_{variant}"), pairs)
                cyc.append(mu_ms + q_ms); upd.append(mu_ms)
                key = f"S1_{q}_{h}"
                if ref is not None and key in ref and not close(res, ref[key]):
                    mism += 1
                newref[key] = res
            out["S1"][f"Q{q}_{variant}"] = stats(cyc) | {"update_median_ms": statistics.median(upd)}
            print(f"S1 Q={q:4d} {variant:6s} cycle median {statistics.median(cyc):9.1f} ms "
                  f"(update {statistics.median(upd):7.1f} ms)", flush=True)
    # S2 static Yen
    if not a.skip_yen:
        for q in [x for x in QS_S2 if x <= a.max_q]:
            pairs = sample_pairs(len(N), q, salt=2)
            res, ms = timed(be.yen, pairs, K_YEN)
            key = f"S2_{q}"
            if ref is not None and key in ref and not all(close(x, y) for x, y in zip(res, ref[key])):
                mism += 1
            newref[key] = res
            out["S2"][f"Q{q}"] = {"total_ms": ms, "per_query_ms": ms / q}
            print(f"S2 Q={q:4d} Yen K={K_YEN} total {ms:9.1f} ms", flush=True)
    # S3 k-hop
    starts = [s for s, _ in sample_pairs(len(N), Q_S3, salt=3)]
    for k in KS_S3:
        res, ms = timed(be.khop, starts, k)
        key = f"S3_{k}"
        if ref is not None and key in ref and res != ref[key]:
            mism += 1
        newref[key] = res
        out["S3"][f"k{k}"] = {"total_ms": ms, "per_query_ms": ms / Q_S3, "mean_reached": float(np.mean(res))}
        print(f"S3 k={k} {Q_S3} starts total {ms:9.1f} ms (mean reached {np.mean(res):.0f})", flush=True)
    out["mismatch_groups_vs_reference"] = mism if ref is not None else None
    if a.backend == "memory":
        ref_path.write_text(json.dumps(newref))
    (RES / f"scale_{a.graph}_{a.backend}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
