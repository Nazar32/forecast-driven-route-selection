"""
Paper 3 follow-up: does spatial propagation carry information beyond the anchored forecast?

Cheap proxy for a spatio-temporal GNN: the C4 configuration (base+inst+anchor+no_region, oblast units) plus a
"graph" feature group, evaluated with the unchanged harness p3_raion_exp.py:
  2-hop neighbours   share under alert of regions at graph distance 2 (now, 6 h, at the departure instant)
  onset waves        onsets in the last 1 and 3 h of the region, its neighbours, 2-hop neighbours, the country
  learned upstream   weights w_ij = max(lift - 1, 0) of "onset of j at t -> onset of i in (t, t+3]", estimated
                     only on hours before T_CAL (all region pairs, not just neighbours), applied to onsets of the
                     last 1 and 3 h and to the state at the departure instant
Development data only (c2024 and rolling windows); compared with the existing runs obl_<win> of p3_raion_exp.

  python p3_graph_diag.py NAME '{"win": "c2024"}'
Outputs: results/p3v8/graph_diag/NAME/... (same files as p3_raion_exp).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import p3_raion_exp as X  # noqa: E402
import p3_raion_model as RM  # noqa: E402

X.OUTROOT = X.C.RES / "p3v8" / "graph_diag"
_build = X.build
T_CAL = None


def upstream_weights(on, T, min_on=20, horizon=3):
    """W[i, j]: lift of an onset of i within (t, t+horizon] after an onset of j at t, hours < T only."""
    O = on.loc[:T - X.H1].values.astype(np.float64)
    n, m = O.shape
    nxt = np.zeros_like(O)
    for h in range(1, horizon + 1):
        nxt[:-h] = np.maximum(nxt[:-h], O[h:])
    base = nxt.mean(0)
    cnt = O.sum(0)
    W = np.zeros((m, m))
    for j in range(m):
        if cnt[j] < min_on:
            continue
        cond = nxt[O[:, j] > 0].mean(0)
        W[:, j] = np.maximum(cond / np.maximum(base, 1e-9) - 1, 0)
    np.fill_diagonal(W, 0)
    s = W.sum(1, keepdims=True)
    return np.where(s > 0, W / np.where(s > 0, s, 1), 0).astype(np.float32)


def build(pres, adj, feats, truth, end=None):
    Xf = _build(pres, adj, feats, truth, end)
    if "graph" not in feats:
        return Xf
    A = pres.loc[RM.START:(end or RM.END)].astype(np.float32)
    us = list(A.columns)
    m = len(us)
    D = np.full((m, m), 99)
    for i, u in enumerate(us):
        D[i, i] = 0
        for q in adj[u]:
            if q in us:
                D[i, us.index(q)] = 1
    for _ in range(3):
        D = np.minimum(D, (D[:, :, None] + D[None, :, :]).min(1))
    def norm(B):
        B = B.astype(np.float32)
        s = B.sum(1, keepdims=True)
        return np.where(s > 0, B / np.where(s > 0, s, 1), 0).astype(np.float32)
    M1, M2 = norm(D == 1), norm(D == 2)
    on1 = ((A == 1) & (A.shift(1) == 0)).astype(np.float32)
    on3 = on1.rolling(3, min_periods=1).sum()
    act = X.instant_state(A.index + X.H1, us, truth)["act"]
    W = upstream_weights(on1, T_CAL)
    mix = lambda V, B: np.asarray(V, np.float32) @ B.T
    g = {"nb2_0": mix(A, M2), "nb2_6": mix(A.rolling(6, min_periods=1).mean(), M2), "i_nb2": mix(act, M2),
         "on1": on1.values, "on3": on3.values, "nbon1": mix(on1, M1), "nbon3": mix(on3, M1),
         "nb2on3": mix(on3, M2), "naton3": np.repeat(on3.values.mean(1, keepdims=True), m, 1),
         "up_on1": mix(on1, W), "up_on3": mix(on3, W), "up_inst": mix(act, W),
         "up_new": mix(act * (1 - A.values), W)}
    for k, v in g.items():
        Xf[k] = np.asarray(v, np.float32).T.reshape(-1)
    nz = (W > 0).sum(1)
    print("graph feats", len(g), "upstream sources per region: median", int(np.median(nz)), "max", int(nz.max()),
          flush=True)
    top = {us[i]: [us[j] for j in np.argsort(-W[i])[:3] if W[i, j] > 0] for i in range(m)}
    (X.OUTROOT / X.NAME).mkdir(parents=True, exist_ok=True)
    (X.OUTROOT / X.NAME / "upstream_top3.json").write_text(json.dumps(top, ensure_ascii=False, indent=1))
    return Xf


if __name__ == "__main__":
    cfg = json.loads(sys.argv[2])
    cfg = {"feats": ["base", "inst", "anchor", "no_region", "graph"], "truth": "oblast_any", "units": "oblast", **cfg}
    sys.argv[2] = json.dumps(cfg)
    T_CAL = X.UTC(X.WIN[cfg["win"]]["T_CAL"])
    X.NAME = sys.argv[1]
    X.build = build
    X.main()
