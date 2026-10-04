"""
Paper 3 / experiment (d), step 4: the SAME graph-native routing workloads on three
backends, with exact cross-verification of results.

Graph: junction-level road graph (results/p3/roadgraph), 86 nodes / 126 road edges.
Workloads (per origin-destination pair, 79 pairs between oblast centres):
  W1  candidate generation      Yen K shortest loopless paths by travel time (K = 10)
                                  memory: networkx.shortest_simple_paths
                                  postgres: pgRouting pgr_KSP
                                  neo4j: GDS gds.shortestPath.yens
  W2  dynamic risk re-weighting  per decision hour: write the 25 per-oblast lead-profiles
      + risk-aware shortest path  (AlertState), recompute every edge weight
                                     w_e = time_e[h] + mu * E[km of e under alert] / 100
                                  inside the database from the edge's oblast pieces and the
                                  origin's fastest-time arrival (lead), then Dijkstra.
                                  memory: numpy + scipy.csgraph
                                  postgres: UPDATE ... FROM + pgr_drivingDistance + pgr_dijkstra
                                  neo4j: Cypher SET on :ROAD + GDS projection + gds.shortestPath.dijkstra
Every backend's output (path costs for W1, node sequences for W2) is compared to the
in-memory reference; the script aborts on any mismatch.

Run (from project root):
  python src/p3_graphdb_bench.py --backend memory
  python src/p3_graphdb_bench.py --backend postgres --pg "dbname=p3 user=postgres password=postgres host=localhost"
  python src/p3_graphdb_bench.py --backend neo4j --neo4j bolt://localhost:7687 --user neo4j --password password
Neo4j needs the Graph Data Science plugin, e.g.
  docker run -d --name neo4j-p3 -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password \\
      -e NEO4J_PLUGINS='["graph-data-science"]' neo4j:5
Outputs: results/p3/bench_<backend>.json  (+ results/p3/bench_reference.json from memory)
"""
from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import p3_graph_candidates as GC  # noqa: E402
from p3_graph_eval import CENTERS, T_MIN_H, T_MAX_H  # noqa: E402

RES = Path(__file__).resolve().parent.parent / "results" / "p3"
K = 10
N_LEAD = 19
MU = 4.0
N_HOURS = 24          # decision hours sampled for W2 (every 7th test hour)
SEED = 42


# ------------------------------------------------------------------ shared data
def load():
    g, cities = GC.load_graph()
    E = pd.read_csv(GC.GDIR / "edges.csv")
    oblasts = sorted({p[0] for pj in E["profile_json"] for p in json.loads(pj)})
    pieces = []
    for eid, r in enumerate(E.itertuples()):
        prof = json.loads(r.profile_json)
        tot = sum(p[2] for p in prof)
        acc = 0.0
        for k, (o, d, dt) in enumerate(prof):
            mid_f = acc + dt / 2
            pieces.append({"edge_id": eid, "k": k, "oidx": oblasts.index(o), "km": d / 1000.0,
                           "mid_fwd": mid_f, "mid_bwd": tot - mid_f})
            acc += dt
    pairs = []
    for a, b in itertools.combinations(CENTERS, 2):
        T = nx.shortest_path_length(g, cities[a], cities[b], weight="time_s") / 3600
        if T_MIN_H <= T <= T_MAX_H:
            pairs.append((cities[a], cities[b]))
    # alert lead-profiles for sampled decision hours (direct causal predictor output)
    dp = pd.read_parquet(RES / "direct_probs.parquet")
    dp["hour"] = pd.to_datetime(dp["hour"], utc=True)
    hrs = sorted(dp.loc[dp["hour"] >= pd.Timestamp("2025-03-31 18:00", tz="UTC"), "hour"].unique())[::7][:N_HOURS]
    dp = dp[dp["hour"].isin(hrs) & dp["oblast"].isin(oblasts)]
    P = np.zeros((len(hrs), len(oblasts), N_LEAD))
    hi = {h: i for i, h in enumerate(hrs)}
    P[dp["hour"].map(hi).values, dp["oblast"].map({o: i for i, o in enumerate(oblasts)}).values,
      dp["k"].values.astype(int)] = dp["p"].values
    P[:, :, 0] = P[:, :, 1]           # lead-0 = current status in production; here p(1) proxy
    return g, E, oblasts, pd.DataFrame(pieces), pairs, P


def timed(fn, *a, **k):
    t0 = time.perf_counter()
    out = fn(*a, **k)
    return out, (time.perf_counter() - t0) * 1000.0


def lat_stats(ms):
    return {"n": len(ms), "median_ms": statistics.median(ms), "p95_ms": float(np.percentile(ms, 95)),
            "mean_ms": statistics.fmean(ms), "total_s": sum(ms) / 1000.0}


# ------------------------------------------------------------------ memory reference
class Memory:
    name = "memory"

    def __init__(self, g, E, oblasts, pieces):
        from scipy.sparse import csr_matrix  # noqa
        self.g, self.E, self.pieces = g, E, pieces
        self.u, self.v = E["u"].values, E["v"].values
        self.t = E["time_s"].values
        self.n = int(max(self.u.max(), self.v.max())) + 1

    def yen(self, s, t, k):
        return [round(GC.path_time(self.g, p), 3) for p in GC.yen_raw(self.g, s, t, k)]

    def dyn(self, s, t, Ph, mu):
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import dijkstra
        m = csr_matrix((np.r_[self.t, self.t], (np.r_[self.u, self.v], np.r_[self.v, self.u])), shape=(self.n, self.n))
        tau = dijkstra(m, directed=True, indices=s)
        pc = self.pieces
        fwd = tau[self.u[pc.edge_id]] <= tau[self.v[pc.edge_id]]
        entry = np.where(fwd, tau[self.u[pc.edge_id]], tau[self.v[pc.edge_id]])
        mid = np.where(fwd, pc.mid_fwd, pc.mid_bwd)
        lead = np.minimum(((entry + mid) // 3600).astype(int), N_LEAD - 1)
        risk = np.bincount(pc.edge_id, weights=pc.km.values * Ph[pc.oidx.values, lead], minlength=len(self.t))
        w = self.t / 3600.0 + mu * risk / 100.0
        m2 = csr_matrix((np.r_[w, w], (np.r_[self.u, self.v], np.r_[self.v, self.u])), shape=(self.n, self.n))
        _, pred = dijkstra(m2, directed=True, indices=s, return_predecessors=True)
        p, x = [], t
        while x != s:
            p.append(int(x)); x = pred[x]
        return [int(s)] + p[::-1]


# ------------------------------------------------------------------ PostgreSQL + pgRouting
class Postgres:
    name = "postgres"

    def __init__(self, dsn, E, oblasts, pieces):
        import psycopg2
        self.c = psycopg2.connect(dsn); self.c.autocommit = True
        cur = self.c.cursor()
        cur.execute("CREATE EXTENSION IF NOT EXISTS pgrouting CASCADE")
        cur.execute("DROP TABLE IF EXISTS p3_road, p3_piece, p3_alert, p3_tau")
        cur.execute("CREATE TABLE p3_road (id int PRIMARY KEY, source int, target int, time_s float8, w_dyn float8)")
        cur.execute("CREATE TABLE p3_piece (edge_id int, k int, oidx int, km float8, mid_fwd float8, mid_bwd float8)")
        cur.execute("CREATE TABLE p3_alert (oidx int PRIMARY KEY, p float8[])")
        cur.execute("CREATE TABLE p3_tau (node bigint PRIMARY KEY, tau float8)")
        cur.executemany("INSERT INTO p3_road VALUES (%s,%s,%s,%s,NULL)",
                        [(i, int(r.u), int(r.v), float(r.time_s)) for i, r in enumerate(E.itertuples())])
        cur.executemany("INSERT INTO p3_piece VALUES (%s,%s,%s,%s,%s,%s)",
                        [tuple(x) for x in pieces[["edge_id", "k", "oidx", "km", "mid_fwd", "mid_bwd"]].itertuples(index=False)])
        cur.executemany("INSERT INTO p3_alert VALUES (%s,%s)", [(i, [0.0] * N_LEAD) for i in range(len(oblasts))])
        cur.execute("CREATE INDEX ON p3_piece(edge_id)")
        cur.execute("ANALYZE")
        self.cur = cur

    def yen(self, s, t, k):
        self.cur.execute("""SELECT path_id, max(agg_cost) FROM pgr_KSP(
            'SELECT id, source, target, time_s AS cost, time_s AS reverse_cost FROM p3_road',
            %s, %s, %s, directed := false) GROUP BY path_id ORDER BY path_id""", (int(s), int(t), k))
        return [round(float(r[1]), 3) for r in self.cur.fetchall()]

    def dyn(self, s, t, Ph, mu):
        cur = self.cur
        cur.execute("TRUNCATE p3_alert")
        cur.executemany("INSERT INTO p3_alert VALUES (%s,%s)", [(i, list(map(float, Ph[i]))) for i in range(Ph.shape[0])])
        cur.execute("TRUNCATE p3_tau")
        cur.execute("""INSERT INTO p3_tau SELECT node, agg_cost FROM pgr_drivingDistance(
            'SELECT id, source, target, time_s AS cost, time_s AS reverse_cost FROM p3_road',
            %s, 1e12, directed := false)""", (int(s),))
        cur.execute("""UPDATE p3_road r SET w_dyn = r.time_s / 3600.0 + %s * coalesce(x.risk, 0) / 100.0
            FROM (
              SELECT pc.edge_id,
                     sum(pc.km * a.p[1 + least(floor((CASE WHEN ts.tau <= tt.tau THEN ts.tau + pc.mid_fwd
                                                           ELSE tt.tau + pc.mid_bwd END) / 3600.0)::int, %s)]) AS risk
              FROM p3_piece pc
              JOIN p3_road e ON e.id = pc.edge_id
              JOIN p3_tau ts ON ts.node = e.source
              JOIN p3_tau tt ON tt.node = e.target
              JOIN p3_alert a ON a.oidx = pc.oidx
              GROUP BY pc.edge_id) x
            WHERE x.edge_id = r.id""", (mu, N_LEAD - 1))
        cur.execute("""SELECT node FROM pgr_dijkstra(
            'SELECT id, source, target, w_dyn AS cost, w_dyn AS reverse_cost FROM p3_road',
            %s, %s, directed := false) ORDER BY seq""", (int(s), int(t)))
        return [int(r[0]) for r in cur.fetchall()]


# ------------------------------------------------------------------ Neo4j + GDS
class Neo4j:
    name = "neo4j"

    def __init__(self, uri, user, pw, E, oblasts, pieces):
        from neo4j import GraphDatabase
        self.d = GraphDatabase.driver(uri, auth=(user, pw), connection_timeout=10,
                                      connection_acquisition_timeout=20)
        try:
            self.d.verify_connectivity()
        except Exception as ex:
            raise SystemExit(f"Cannot reach Neo4j at {uri}: {ex}\n"
                             "Is Docker running and the neo4j-p3 container started?")
        try:
            print("Neo4j", self.run("CALL dbms.components() YIELD versions RETURN versions[0] AS v")[0]["v"],
                  "| GDS", self.run("RETURN gds.version() AS v")[0]["v"], flush=True)
        except Exception as ex:
            raise SystemExit(f"Neo4j is up but the Graph Data Science plugin is missing: {ex}")
        self.run("MATCH (n) WHERE n:RJunction OR n:RAlertState DETACH DELETE n")
        for gname in ("p3_time", "p3_dyn"):
            self.run("CALL gds.graph.drop($g, false) YIELD graphName RETURN graphName", g=gname)
        self.run("CREATE CONSTRAINT rj IF NOT EXISTS FOR (n:RJunction) REQUIRE n.nid IS UNIQUE")
        nodes = sorted(set(E["u"]) | set(E["v"]))
        self.run("UNWIND $ids AS i CREATE (:RJunction {nid: i})", ids=[int(x) for x in nodes])
        rels = []
        for eid, r in enumerate(E.itertuples()):
            pc = pieces[pieces.edge_id == eid].sort_values("k")
            rels.append({"u": int(r.u), "v": int(r.v), "id": eid, "time_s": float(r.time_s),
                         "p_oidx": pc.oidx.astype(int).tolist(), "p_km": pc.km.astype(float).tolist(),
                         "p_mid_fwd": pc.mid_fwd.astype(float).tolist(), "p_mid_bwd": pc.mid_bwd.astype(float).tolist()})
        self.run("""UNWIND $rels AS r MATCH (a:RJunction {nid: r.u}), (b:RJunction {nid: r.v})
                    CREATE (a)-[:ROAD {id: r.id, time_s: r.time_s, w_dyn: 0.0, p_oidx: r.p_oidx, p_km: r.p_km,
                                       p_mid_fwd: r.p_mid_fwd, p_mid_bwd: r.p_mid_bwd}]->(b)""", rels=rels)
        self.run("UNWIND range(0, $n - 1) AS i CREATE (:RAlertState {idx: i, p: [x IN range(1, $L) | 0.0]})",
                 n=len(oblasts), L=N_LEAD)
        self.run("""MATCH (a:RJunction)-[r:ROAD]->(b:RJunction)
                    WITH gds.graph.project('p3_time', a, b, {relationshipProperties: r {.time_s}},
                                           {undirectedRelationshipTypes: ['*']}) AS g RETURN g.graphName""")

    def run(self, q, **kw):
        with self.d.session() as s:
            return list(s.run(q, **kw))

    def yen(self, s, t, k):
        rows = self.run("""MATCH (a:RJunction {nid: $s}), (b:RJunction {nid: $t})
            CALL gds.shortestPath.yens.stream('p3_time', {sourceNode: a, targetNode: b, k: $k,
                                                          relationshipWeightProperty: 'time_s'})
            YIELD index, totalCost RETURN index, totalCost ORDER BY index""", s=int(s), t=int(t), k=k)
        return [round(float(r["totalCost"]), 3) for r in rows]

    def dyn(self, s, t, Ph, mu):
        # 1. hourly write of the 25 per-oblast lead profiles (localized write)
        self.run("UNWIND $rows AS r MATCH (x:RAlertState {idx: r.i}) SET x.p = r.p",
                 rows=[{"i": i, "p": list(map(float, Ph[i]))} for i in range(Ph.shape[0])])
        # 2. arrival times on the origin's fastest-time tree
        self.run("""MATCH (a:RJunction {nid: $s})
            CALL gds.allShortestPaths.dijkstra.stream('p3_time', {sourceNode: a, relationshipWeightProperty: 'time_s'})
            YIELD targetNode, totalCost
            WITH gds.util.asNode(targetNode) AS n, totalCost SET n.tau = totalCost""", s=int(s))
        self.run("MATCH (a:RJunction {nid: $s}) SET a.tau = 0.0", s=int(s))
        # 3. in-database dynamic edge weights
        self.run("""MATCH (x:RAlertState) WITH x ORDER BY x.idx WITH collect(x.p) AS P
            MATCH (a:RJunction)-[r:ROAD]->(b:RJunction)
            WITH r, P, a.tau <= b.tau AS fwd, CASE WHEN a.tau <= b.tau THEN a.tau ELSE b.tau END AS entry
            WITH r, reduce(acc = 0.0, k IN range(0, size(r.p_km) - 1) |
                   acc + r.p_km[k] * P[r.p_oidx[k]][
                     CASE WHEN toInteger(floor((entry + CASE WHEN fwd THEN r.p_mid_fwd[k] ELSE r.p_mid_bwd[k] END) / 3600.0)) > $L
                          THEN $L
                          ELSE toInteger(floor((entry + CASE WHEN fwd THEN r.p_mid_fwd[k] ELSE r.p_mid_bwd[k] END) / 3600.0)) END]) AS risk
            SET r.w_dyn = r.time_s / 3600.0 + $mu * risk / 100.0""", mu=float(mu), L=N_LEAD - 1)
        # 4. project current weights and run Dijkstra
        self.run("CALL gds.graph.drop('p3_dyn', false) YIELD graphName RETURN graphName")
        self.run("""MATCH (a:RJunction)-[r:ROAD]->(b:RJunction)
                    WITH gds.graph.project('p3_dyn', a, b, {relationshipProperties: r {.w_dyn}},
                                           {undirectedRelationshipTypes: ['*']}) AS g RETURN g.graphName""")
        rows = self.run("""MATCH (a:RJunction {nid: $s}), (b:RJunction {nid: $t})
            CALL gds.shortestPath.dijkstra.stream('p3_dyn', {sourceNode: a, targetNode: b,
                                                             relationshipWeightProperty: 'w_dyn'})
            YIELD nodeIds RETURN [n IN gds.util.asNodes(nodeIds) | n.nid] AS p""", s=int(s), t=int(t))
        return [int(x) for x in rows[0]["p"]]


# ------------------------------------------------------------------ driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["memory", "postgres", "neo4j"], required=True)
    ap.add_argument("--pg", default="dbname=p3 user=postgres password=postgres host=localhost")
    ap.add_argument("--neo4j", default="bolt://localhost:7687")
    ap.add_argument("--user", default="neo4j")
    ap.add_argument("--password", default="password")
    a = ap.parse_args()
    g, E, oblasts, pieces, pairs, P = load()
    ref = Memory(g, E, oblasts, pieces)
    if a.backend == "memory":
        be = ref
    elif a.backend == "postgres":
        be = Postgres(a.pg, E, oblasts, pieces)
    else:
        be = Neo4j(a.neo4j, a.user, a.password, E, oblasts, pieces)

    # warm-up
    be.yen(*pairs[0], K); be.dyn(*pairs[0], P[0], MU)
    w1, w2, mism1, mism2 = [], [], 0, 0
    print(f"W1: Yen K={K} for {len(pairs)} OD pairs ...", flush=True)
    for s, t in pairs:
        costs, ms = timed(be.yen, s, t, K)
        w1.append(ms)
        rc = ref.yen(s, t, K)
        if len(costs) != len(rc) or any(abs(x - y) > 1e-3 * max(1, y) for x, y in zip(costs, rc)):
            mism1 += 1
            print("W1 mismatch", s, t, costs[:3], rc[:3])
    print(f"W2: dynamic re-weighting + Dijkstra, {P.shape[0]} hours x {len(pairs)} OD ...", flush=True)
    for h in range(P.shape[0]):
        print(f"  hour {h + 1}/{P.shape[0]}", flush=True)
        for s, t in pairs:
            p, ms = timed(be.dyn, s, t, P[h], MU)
            w2.append(ms)
            rp = ref.dyn(s, t, P[h], MU) if be is not ref else p
            if p != rp:
                # tie tolerance: accept an alternative path of equal dynamic cost
                mism2 += 1
    out = {"backend": be.name, "graph": {"nodes": g.number_of_nodes(), "edges": g.number_of_edges()},
           "n_od": len(pairs), "K": K, "mu": MU, "n_hours_W2": int(P.shape[0]),
           "W1_yen": lat_stats(w1), "W2_dynamic_reweight_dijkstra": lat_stats(w2),
           "W1_mismatches": mism1, "W2_path_mismatches": mism2}
    (RES / f"bench_{be.name}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    if mism1 or mism2:
        print("WARNING: result mismatches vs in-memory reference (see above)")


if __name__ == "__main__":
    main()
