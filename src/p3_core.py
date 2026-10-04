"""
Paper 3 core evaluation primitives (experiments a, b, c).

Two realized exposure metrics, both computed only from the ground-truth alert log
and the chosen route (prediction-independent):

  AES_W   (legacy, as in the current manuscript): distance share of the route whose
          oblast had ANY alert during [t, t+W], W = 8 h, regardless of when the
          vehicle is actually there.
  DUA     (new, time-resolved): "distance driven under alert" -- share of route
          kilometres driven while the oblast the vehicle is in at that hour has an
          active alert.  Uses the ORS travel-time profile of the route, binned by
          hour after departure.   DUA(r,t) = sum_j sum_i W_r[j,i] * a(t+j, i).

Alert presence a(u,i): oblast-level rows only, hour-overlap floor(start)..floor(finish-1s)
(identical to routing_analysis.py, which reproduces the manuscript numbers exactly).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p3_geo as G  # noqa: E402

ROOT = HERE.parent                                   # repository root
DATA = ROOT / "data"
RES = ROOT / "results"
PAPER = ROOT / "paper"                               # generated tables and figures
P3_IN = RES / "p3"                                   # shared inputs (road graphs, geometry caches)
P3 = RES / os.environ.get("P3_RESULTS", "p3v2")      # outputs of this revision
LOG = DATA / "alerts" / "official_data_uk_2026-07.csv.gz"

# v2 (paper v2.x): data to 13.10.2025, test block from 31.03.2025.  v4 (two declaration regimes): the main
# evaluation is restricted to the oblast-declaration period, P3_END="2025-04-30 23:00" and
# P3_TEST_START="2024-11-10 15:00" (the 70/15/15 split of the truncated frame).
TEST_START = pd.Timestamp(os.environ.get("P3_TEST_START", "2025-03-31 18:00"), tz="UTC")
TEST_END = pd.Timestamp(os.environ.get("P3_END", "2025-10-13 23:00"), tz="UTC")
DATA_START = pd.Timestamp("2022-03-15 16:00", tz="UTC")
W_LEGACY = 8
K_MAX = 18


# ---------------------------------------------------------------- alert presence
def presence(oblasts=None):
    full = pd.date_range(DATA_START, TEST_END + pd.Timedelta(hours=K_MAX + W_LEGACY + 2), freq="h", tz="UTC")
    log = pd.read_csv(LOG)
    log = log[log["level"] == "oblast"].copy()
    if oblasts is not None:
        log = log[log["oblast"].isin(oblasts)]
    obl = sorted(log["oblast"].unique()) if oblasts is None else list(oblasts)
    log["s"] = pd.to_datetime(log["started_at"], utc=True, errors="coerce").dt.floor("h")
    log["f"] = (pd.to_datetime(log["finished_at"], utc=True, errors="coerce") - pd.Timedelta(seconds=1)).dt.floor("h")
    log = log.dropna(subset=["s", "f"])
    A = np.zeros((len(full), len(obl)), dtype=np.int8)
    pos = pd.Series(np.arange(len(full)), index=full)
    col = {o: j for j, o in enumerate(obl)}
    for o, gg in log.groupby("oblast"):
        j = col[o]
        for s, f in zip(gg["s"], gg["f"]):
            if f < full[0] or s > full[-1]:
                continue
            a = pos[max(s, full[0])]
            b = pos[min(f, full[-1])]
            A[a:b + 1, j] = 1
    return pd.DataFrame(A, index=full, columns=obl)


def active_at(hours, oblasts):
    """1 if an oblast-level alert is ACTIVE at the instant h:00 (started <= h < finished).
    This is what a dispatcher observes at departure; a(u,i) above (any alert during hour u)
    is only known at the end of hour u."""
    log = pd.read_csv(LOG)
    log = log[(log["level"] == "oblast") & log["oblast"].isin(oblasts)].copy()
    log["s"] = pd.to_datetime(log["started_at"], utc=True, errors="coerce")
    log["f"] = pd.to_datetime(log["finished_at"], utc=True, errors="coerce")
    log = log.dropna(subset=["s", "f"])
    hv = hours.values.astype("datetime64[ns]")
    out = np.zeros((len(hours), len(oblasts)), dtype=np.float32)
    for j, o in enumerate(oblasts):
        g = log[log["oblast"] == o]
        s = g["s"].values.astype("datetime64[ns]"); f = g["f"].values.astype("datetime64[ns]")
        order = np.argsort(s); s, f = s[order], f[order]
        fmax = np.maximum.accumulate(f)                      # latest finish among alerts started so far
        k = np.searchsorted(s, hv, side="right") - 1          # last alert started at or before h
        ok = k >= 0
        out[ok, j] = (fmax[k[ok]] > hv[ok]).astype(np.float32)
    return out


def causal_P(dp, hours, oblasts, act_now):
    """Per-lead forecast tensor P[h, j, o] for a departure at h:00 (v2, no look-ahead).
    dp: long frame (hour = feature row, oblast, k, p) from direct models k = 1..K_MAX+1.
    The feature row of hour h-1 is the last complete one at h:00, so lead j (clock hour h+j)
    uses the model k = j+1 applied to row h-1.  Lead 0 additionally uses the observed status:
    P[h,0,o] = max(active at h:00, p(row h-1, k=1))."""
    rows = hours - pd.Timedelta(hours=1)
    d = dp[dp["hour"].isin(rows) & dp["oblast"].isin(oblasts) & (dp["k"] <= K_MAX + 1)]
    P = np.zeros((len(hours), K_MAX + 1, len(oblasts)), dtype=np.float32)
    hi = pd.Series(np.arange(len(hours)), index=rows)
    oi = {o: j for j, o in enumerate(oblasts)}
    P[hi[d["hour"]].values, d["k"].values.astype(int) - 1, d["oblast"].map(oi).values] = d["p"].values
    miss = pd.Index(rows).difference(pd.Index(d["hour"].unique()))
    if len(miss):
        raise ValueError(f"no forecasts for {len(miss)} feature rows, e.g. {miss[:3]}")
    P[:, 0, :] = np.maximum(P[:, 0, :], act_now)
    return P


def read_probs(path):
    dp = pd.read_parquet(path)
    dp["hour"] = pd.to_datetime(dp["hour"], utc=True)
    return dp


# ---------------------------------------------------------------- routes
def dedupe(routes, tol_m=50.0):
    keep, dup = [], {}
    for r in routes:
        sig = [(s["oblast"], s["distance_m"]) for s in r["traversal"]]
        twin = None
        for k in keep:
            sk = [(s["oblast"], s["distance_m"]) for s in k["traversal"]]
            if len(sk) == len(sig) and abs(k["D"] - r["D"]) < tol_m and all(
                    a[0] == b[0] and abs(a[1] - b[1]) < tol_m for a, b in zip(sk, sig)):
                twin = k["id"]
                break
        if twin:
            dup[r["id"]] = twin
        else:
            keep.append(r)
    return keep, dup


def hour_weights(route, oblasts, n_bins=K_MAX + 1):
    """W[j, i]: share of route distance driven in oblast i during hour j after departure."""
    Wm = np.zeros((n_bins, len(oblasts)))
    col = {o: j for j, o in enumerate(oblasts)}
    if route.get("hour_bins"):
        tot = sum(d for _, _, d in route["hour_bins"])
        for j, o, d in route["hour_bins"]:
            Wm[min(j, n_bins - 1), col[o]] += d / tot
        return Wm
    D = sum(s["distance_m"] for s in route["traversal"])
    for s in route["traversal"]:
        t0, t1, d = s["t_enter_s"], s["t_exit_s"], s["distance_m"]
        dur = max(t1 - t0, 1e-9)
        j0, j1 = int(t0 // 3600), int(min(t1, n_bins * 3600 - 1e-6) // 3600)
        for j in range(j0, j1 + 1):
            ov = min(t1, (j + 1) * 3600) - max(t0, j * 3600)
            if ov > 0:
                Wm[j, col[s["oblast"]]] += d * ov / dur / D
    return Wm


def time_weights(route, oblasts, n_bins=K_MAX + 1):
    """H[j, i]: hours spent in oblast i during hour j after departure (needs t_enter_s/t_exit_s)."""
    Hm = np.zeros((n_bins, len(oblasts)))
    col = {o: j for j, o in enumerate(oblasts)}
    for s in route["traversal"]:
        t0, t1 = s["t_enter_s"], s["t_exit_s"]
        for j in range(int(t0 // 3600), int(min(t1, n_bins * 3600 - 1e-6) // 3600) + 1):
            ov = min(t1, (j + 1) * 3600) - max(t0, j * 3600)
            if ov > 0:
                Hm[j, col[s["oblast"]]] += ov / 3600
    return Hm


def dist_weights(route, oblasts):
    col = {o: j for j, o in enumerate(oblasts)}
    w = np.zeros(len(oblasts))
    D = sum(s["distance_m"] for s in route["oblast_segments"])
    for s in route["oblast_segments"]:
        w[col[s["oblast"]]] += s["distance_m"] / D
    return w


class Corridor:
    def __init__(self, key, act_all, routes=None):
        routes = G.load_corridor_geo(key) if routes is None else routes
        self.routes, self.duplicates = dedupe(routes)
        self.key = key
        self.ids = [r["id"] for r in self.routes]
        self.D = np.array([r["D"] for r in self.routes]) / 1000.0   # km
        self.T = np.array([r["T"] for r in self.routes]) / 3600.0   # h
        self.obl = sorted({s["oblast"] for r in self.routes for s in r["traversal"]})
        self.act = act_all[self.obl]
        self.Wh = np.stack([hour_weights(r, self.obl) for r in self.routes])   # R x J x O
        self.Wd = np.stack([dist_weights(r, self.obl) for r in self.routes])   # R x O
        self.A = self.act.values.astype(np.float64)
        self.pos = pd.Series(np.arange(len(self.act)), index=self.act.index)

    # realized metrics for decision hours (DatetimeIndex) -> H x R
    def dua(self, hours):
        i0 = self.pos[hours].values
        J = self.Wh.shape[1]
        out = np.zeros((len(hours), len(self.routes)))
        for j in range(J):
            out += self.A[i0 + j] @ self.Wh[:, j, :].T
        return out

    def km_under_alert(self, hours):
        return self.dua(hours) * self.D[None, :]

    def hours_under_alert(self, hours):
        if not hasattr(self, "Wt"):
            self.Wt = np.stack([time_weights(r, self.obl) for r in self.routes])
        i0 = self.pos[hours].values
        out = np.zeros((len(hours), len(self.routes)))
        for j in range(self.Wt.shape[1]):
            out += self.A[i0 + j] @ self.Wt[:, j, :].T
        return out

    # v2 objective: absolute km under alert (share x route length)
    def metric(self, hours):
        return self.km_under_alert(hours)

    def expected_metric(self, P):
        return self.expected_dua(P) * self.D[None, :]

    def current_metric(self, cur):   # km of the route in regions under alert now
        return self.weighted_mean(cur) * self.D[None, :]

    def aes_legacy(self, hours, W=W_LEGACY):
        i0 = self.pos[hours].values
        anyw = np.zeros((len(hours), self.A.shape[1]))
        for j in range(W + 1):
            anyw = np.maximum(anyw, self.A[i0 + j])
        return anyw @ self.Wd.T

    # expected DUA given per-lead probabilities P[h, j, o]  (j = 0..J-1)
    def expected_dua(self, P):
        return np.einsum("hjo,rjo->hr", P, self.Wh)

    def weighted_mean(self, p):      # p: H x O  -> H x R  (paper's S_pred risk = 1 - S_pred)
        return p @ self.Wd.T


# ---------------------------------------------------------------- statistics
def block_bootstrap_impr(base, x, block=24, n=2000, seed=42):
    """Moving-block bootstrap (24 h blocks) CI for 100*(mean(base)-mean(x))/mean(base)."""
    rng = np.random.default_rng(seed)
    H = len(base)
    nb = int(np.ceil(H / block))
    starts = rng.integers(0, H - block + 1, size=(n, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n, -1)[:, :H]
    b, xx = base[idx].mean(1), x[idx].mean(1)
    v = 100 * (b - xx) / b
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def impr(base, x):
    return float(100 * (np.mean(base) - np.mean(x)) / np.mean(base))
