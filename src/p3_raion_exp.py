"""
Paper 3: anchored per-lead forecasts and the decision layer (development rounds, rolling windows, confirmation).

Windows (fixed before running):
  val    train < 2025-12-01, isotonic calibration Dec 2025, evaluation departures 2026-01-01 .. 2026-02-28
         (fully in the raion-declaration regime)
  final  train < 2026-02-01, calibration Feb 2026, evaluation departures 2026-03-01 .. 2026-09-06
         (the test block of the earlier raion_w2026 run; used once, for the configuration chosen on val)

  python p3_raion_exp.py NAME '{"feats": ["base", "inst"], "win": "val"}'
Config keys:
  feats       base (as p3_raion_model) | inst (state at the departure instant from the log: active, elapsed,
              time since end, hromada active, oblast and national active share) | lags | regime | onset
  truth       "raion" (oblast OR raion alert of the raion; default) | "oblast_any" (any alert in the oblast,
              assigned to all its raions: the period-vs-granularity control)
  train_from  first training hour (default data start)
  max_iter, lr, leaves   HGB hyperparameters (defaults as p3_raion_model)
Outputs: results/p3v6/raion_exp/NAME/{probs.npz, eval.json}; one summary line appended to log.txt.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import p3_core as C  # noqa: E402
import p3_geo as G  # noqa: E402
import p3_raion as RA  # noqa: E402
import p3_raion_model as RM  # noqa: E402

SRC = C.RES / "p3v3" / "raion"       # presence + adjacency (unchanged)
OUTROOT = C.RES / "p3v6" / "raion_exp"
K_MAX = C.K_MAX
NK = K_MAX + 1                                                         # models k = 1..19
WIN = {
    "val": dict(T_CAL="2025-12-01 00:00", T_EVAL="2026-01-01 00:00", T_END="2026-02-28 23:00"),
    "val2": dict(T_CAL="2025-11-01 00:00", T_EVAL="2025-12-01 00:00", T_END="2025-12-31 23:00"),
    "c2024": dict(T_CAL="2024-06-01 00:00", T_EVAL="2024-07-31 15:00", T_END="2024-12-31 23:00"),
    "c2025": dict(T_CAL="2024-12-01 00:00", T_EVAL="2025-01-01 00:00", T_END="2025-04-30 23:00"),
    "W0": dict(T_CAL="2023-01-01 00:00", T_EVAL="2023-04-01 00:00", T_END="2023-09-30 23:00"),
    "W1": dict(T_CAL="2023-07-01 00:00", T_EVAL="2023-10-01 00:00", T_END="2024-03-31 23:00"),
    "W2": dict(T_CAL="2024-01-01 00:00", T_EVAL="2024-04-01 00:00", T_END="2024-09-30 23:00"),
    "W3c": dict(T_CAL="2024-07-01 00:00", T_EVAL="2024-10-01 00:00", T_END="2024-12-31 23:00"),
    "final": dict(T_CAL="2026-02-01 00:00", T_EVAL="2026-03-01 00:00", T_END="2026-09-06 23:00"),
}
EPS = (0.0, 0.02, 0.05, 0.1, 0.2)                                      # relative switching margins
UTC = lambda s: pd.Timestamp(s, tz="UTC")
H1 = pd.Timedelta(hours=1)


# ----------------------------------------------------------------------------- inputs
_LOG = None


def log():
    global _LOG
    if _LOG is None:
        lg = RA.read_log(RM.LOG2)
        lg["s"] = pd.to_datetime(lg["started_at"], utc=True, format="ISO8601")
        lg["f"] = pd.to_datetime(lg["finished_at"], utc=True, format="ISO8601")
        _LOG = lg.dropna(subset=["s", "f"])
    return _LOG


OBL_EXTRA_ADJ = {"Луганська область": ["Донецька область", "Харківська область"]}


def load_oblast_inputs():
    """Oblast units (the paper's 25 regions); presence = any oblast- or raion-level alert in the oblast
    (in 2022-2024 identical to the paper's oblast truth); adjacency aggregated from the raion adjacency."""
    obl = list(C.presence().columns)
    lg = RA.read_log(RM.LOG2)
    lg = lg[lg["level"].isin(("oblast", "raion"))].copy()
    lg["level"] = "oblast"
    pres = RA.presence_units(obl, ("oblast",), lg, RM.START, RM.END + pd.Timedelta(hours=K_MAX + 2))
    radj = json.loads((SRC / "adjacency.json").read_text())
    adj = {o: set() for o in obl}
    for u, vs in radj.items():
        for v in vs:
            a, b = RA.oblast_of(u), RA.oblast_of(v)
            if a != b and a in adj and b in adj:
                adj[a].add(b)
                adj[b].add(a)
    for o, vs in OBL_EXTRA_ADJ.items():
        for v in vs:
            adj[o].add(v)
            adj[v].add(o)
    return pres, {o: sorted(v) for o, v in adj.items()}


def load_inputs(truth, units="raion"):
    if units == "oblast":
        assert truth == "oblast_any"
        return load_oblast_inputs()
    pres = pd.read_parquet(SRC / "presence.parquet")
    adj = json.loads((SRC / "adjacency.json").read_text())
    if truth == "oblast_any":
        us = list(pres.columns)
        ob = [RA.oblast_of(u) for u in us]
        anyo = pres.T.groupby(ob).max().T
        pres = pd.DataFrame(anyo[ob].values, index=pres.index, columns=us)
    return pres, adj


def unit_intervals(us, truth, levels):
    """merged [s, f) alert intervals per unit from the log records of `levels`."""
    lg = log()
    lg = lg[lg["level"].isin(levels)]
    by = {u: [] for u in us}
    obl_units = {}
    for u in us:
        obl_units.setdefault(RA.oblast_of(u), []).append(u)
    for lev, o, r, s, f in zip(lg["level"], lg["oblast"], lg["raion"], lg["s"].values, lg["f"].values):
        if truth == "oblast_any" or lev == "oblast":
            tgt = obl_units.get(o, [])
        else:
            u = RA.unit(o, r)
            tgt = [u] if u in by else []
        for u in tgt:
            by[u].append((s, f))
    out = {}
    for u, iv in by.items():
        if not iv:
            out[u] = (np.array([], "datetime64[ns]"), np.array([], "datetime64[ns]"))
            continue
        iv.sort()
        S, F = [iv[0][0]], [iv[0][1]]
        for s, f in iv[1:]:
            if s <= F[-1]:
                F[-1] = max(F[-1], f)
            else:
                S.append(s)
                F.append(f)
        out[u] = (np.array(S, "datetime64[ns]"), np.array(F, "datetime64[ns]"))
    return out


def instant_state(instants, us, truth):
    """state at the instants (e.g. departure h:00): active, elapsed h, h since last end, hromada active."""
    tau = instants.tz_convert(None).values.astype("datetime64[ns]")
    res = {k: np.zeros((len(tau), len(us)), np.float32) for k in ("act", "elap", "since_end", "hrom")}
    for lev, key in ((("oblast", "raion"), "main"), (("hromada",), "hrom")):
        iv = unit_intervals(us, truth, lev)
        for j, u in enumerate(us):
            S, F = iv[u]
            if not len(S):
                if key == "main":
                    res["since_end"][:, j] = 168
                continue
            i = np.searchsorted(S, tau, side="right") - 1
            ok = i >= 0
            ic = np.clip(i, 0, None)
            act = ok & (tau < F[ic])
            if key == "hrom":
                res["hrom"][:, j] = act
                continue
            res["act"][:, j] = act
            el = (tau - S[ic]) / np.timedelta64(1, "h")
            se = (tau - F[ic]) / np.timedelta64(1, "h")
            res["elap"][:, j] = np.where(act, np.minimum(el, 72), 0)
            res["since_end"][:, j] = np.where(act, 0, np.where(ok, np.minimum(se, 168), 168))
    return res


# ----------------------------------------------------------------------------- features
def run_length(A):
    V = A.values
    R = np.zeros_like(V)
    for t in range(len(V)):
        R[t] = (R[t - 1] + 1) * V[t] if t else V[t]
    return pd.DataFrame(np.minimum(R, 72), index=A.index, columns=A.columns)


def build(pres, adj, feats, truth, end=None):
    """Long frame (rows = unit x feature hour r); features use only data up to (r+1):00."""
    A = pres.loc[RM.START:(end or RM.END)].astype(np.float32)
    us = list(A.columns)
    roll = lambda X, w: X.rolling(w, min_periods=1).mean()
    EXTRA.clear()
    f = {"a0": A, "a1": A.shift(1), "a2": A.shift(2), "a5": A.shift(5),
         "r3": roll(A, 3), "r6": roll(A, 6), "r24": roll(A, 24), "r72": roll(A, 72), "r168": roll(A, 168)}
    idx = np.arange(len(A))[:, None] * np.ones((1, A.shape[1]))
    lt = pd.DataFrame(np.where(A.values > 0, idx, np.nan), index=A.index, columns=us).ffill()
    f["since"] = pd.DataFrame(np.minimum(idx - lt.values, 720), index=A.index, columns=us).fillna(720)
    M = np.zeros((len(us), len(us)), np.float32)
    for i, u in enumerate(us):
        nb = [us.index(q) for q in adj[u] if q in us] or [i]
        M[i, nb] = 1.0 / len(nb)
    O = np.zeros((len(us), len(us)), np.float32)
    for i, u in enumerate(us):
        mates = [j for j, v in enumerate(us) if RA.oblast_of(v) == RA.oblast_of(u) and j != i] or [i]
        O[i, mates] = 1.0 / len(mates)
    mix = lambda X, W: pd.DataFrame(np.asarray(X, np.float32) @ W.T, index=A.index, columns=us)
    f["nb0"], f["nb6"], f["nb24"] = mix(A, M), mix(roll(A, 6), M), mix(roll(A, 24), M)
    f["ob0"], f["ob6"], f["ob24"] = mix(A, O), mix(roll(A, 6), O), mix(roll(A, 24), O)
    nat = {"nat0": A.mean(axis=1), "nat24": A.mean(axis=1).rolling(24, min_periods=1).mean()}
    if "lags" in feats:
        for L in (3, 4, 6, 8, 12, 23, 24):
            f[f"a{L}"] = A.shift(L)
        f["run"] = run_length(A)
        nat["nat_d1"] = A.mean(axis=1).diff(1)
        nat["nat_d3"] = A.mean(axis=1).diff(3)
        nat["nat_d24"] = A.mean(axis=1).diff(24)
    if "onset" in feats:
        on = ((A == 1) & (A.shift(1) == 0)).astype(np.float32)
        f["on3"] = on.rolling(3, min_periods=1).sum()
        f["nbon3"] = mix(f["on3"], M)
        nat["naton3"] = on.mean(axis=1).rolling(3, min_periods=1).sum()
    if "inst" in feats:                                              # state at (r+1):00, the departure instant
        st = instant_state(A.index + H1, us, truth)
        act = st["act"]
        f["i_act"] = pd.DataFrame(act, index=A.index, columns=us)
        f["i_elap"] = pd.DataFrame(st["elap"], index=A.index, columns=us)
        f["i_send"] = pd.DataFrame(st["since_end"], index=A.index, columns=us)
        f["i_hrom"] = pd.DataFrame(st["hrom"], index=A.index, columns=us)
        f["i_nb"] = mix(act, M)
        f["i_ob"] = mix(act, O)
        nat["i_nat"] = pd.Series(act.mean(1), index=A.index)
        nat["i_nat_d"] = pd.Series(act.mean(1) - A.mean(axis=1).values, index=A.index)
    if "regime" in feats:                                            # share of partially covered oblast-hours, 30 d
        ob = [RA.oblast_of(u) for u in us]
        g = A.T.groupby(ob)
        part = ((g.max().T > 0) & (g.min().T < 1)).astype(np.float32)
        sh = part.rolling(720, min_periods=1).mean()
        f["regime"] = pd.DataFrame(sh[ob].values, index=A.index, columns=us)
        nat["regime_nat"] = part.mean(axis=1).rolling(720, min_periods=1).mean()
    if "anchor" in feats:                                            # rolling 42-day Markov levels (hours <= r)
        W = 1008
        prev = A.shift(1)
        n1 = prev.rolling(W, min_periods=24).sum()
        n0 = (1 - prev).rolling(W, min_periods=24).sum()
        p11 = (prev * A).rolling(W, min_periods=24).sum() / n1.clip(lower=1)
        p01 = ((1 - prev) * A).rolling(W, min_periods=24).sum() / n0.clip(lower=1)
        pi = (p01 / (1 - p11 + p01).clip(lower=1e-6)).fillna(A.expanding().mean())
        f["m_pi"], f["m_p11"], f["m_p01"] = pi, p11.fillna(0), p01.fillna(0)
        f["rate28"], f["rate42"] = A.rolling(672, min_periods=24).mean(), A.rolling(W, min_periods=24).mean()
        EXTRA.update(pi=pi.values.astype(np.float32), lam=(p11 - p01).fillna(0).values.astype(np.float32),
                     A=A.values.astype(np.float32))
    rows = []
    for j, u in enumerate(us):
        d = pd.DataFrame({k: np.asarray(v[u].values, np.float32) for k, v in f.items()}, index=A.index)
        for k, v in nat.items():
            d[k] = np.asarray(v.values, np.float32)
        if "no_region" not in feats:
            d["region"] = j
        rows.append(d.astype(np.float32))
    X = pd.concat(rows, keys=us, names=["r", "hour"]).reset_index()
    h = X.hour
    X["hs"], X["hc"] = np.sin(2 * np.pi * h.dt.hour / 24), np.cos(2 * np.pi * h.dt.hour / 24)
    doy = h.dt.dayofyear
    X["ds"], X["dc"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    X["dow"] = h.dt.dayofweek
    return X


# ----------------------------------------------------------------------------- train
def train(cfg, X, pres, T_CAL, T_EVAL, T_END):
    """Returns Q[row, k-1, unit] for feature rows T_CAL-1h .. T_END and the row index."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.isotonic import IsotonicRegression
    cols = [c for c in X.columns if c not in ("r", "hour")]
    us = list(pres.columns)
    rj = {u: j for j, u in enumerate(us)}
    Apos = pd.Series(np.arange(len(pres)), index=pres.index)
    hi = Apos[X.hour].values
    ri = X.r.map(rj).values
    Av = pres.values
    Xv = X[cols].values.astype(np.float32)
    cat = [cols.index("region")] if "region" in cols else None
    hrs = X.hour
    feats = cfg.get("feats", ["base"])
    t_from = UTC(cfg.get("train_from", str(RM.START)))
    tr = ((hrs >= t_from) & (hrs < T_CAL)).values
    ca = ((hrs >= T_CAL) & (hrs < T_EVAL)).values
    rows = pd.date_range(T_CAL - H1, T_END, freq="h", tz="UTC")
    out_m = ((hrs >= rows[0]) & (hrs <= rows[-1])).values
    sub = ((hi + ri) % 2 == 0)                                      # every second (hour, unit) cell, both parities
    if cfg.get("train_units") == "one_per_oblast":
        first = {}
        for u in us:
            first.setdefault(RA.oblast_of(u), u)
        sub &= X.r.isin(set(first.values())).values
    if "anchor" in feats:
        assert (X.r.iloc[:EXTRA["A"].shape[0]] == us[0]).all() and X.hour.iloc[EXTRA["A"].shape[0]] == X.hour.iloc[0]
        Apos_A = pd.Series(np.arange(EXTRA["A"].shape[0]), index=pd.DatetimeIndex(X.hour.iloc[:EXTRA["A"].shape[0]]))
        hiA = Apos_A[pd.DatetimeIndex(X.hour)].values
    rpos = pd.Series(np.arange(len(rows)), index=rows)[pd.DatetimeIndex(X.hour[out_m])].values
    Q = np.full((len(rows), NK, len(us)), np.nan, np.float32)
    for k in range(1, NK + 1):
        y = Av[hi + k, ri]
        tgt = hrs + pd.Timedelta(hours=k)
        m_tr = tr & sub & (tgt < T_CAL).values
        m_ca = ca & (tgt < T_EVAL).values
        Xk = Xv
        if "anchor" in feats:
            pi_, lam_, A_ = EXTRA["pi"], EXTRA["lam"], EXTRA["A"]
            mk = np.clip(pi_ + (A_ - pi_) * np.power(lam_, k), 1e-4, 1 - 1e-4)
            acc, n = np.zeros_like(A_), 0
            for dd in range(1, 40):                                  # 28-day rate at the target clock hour
                lag = 24 * dd - k
                if lag < 0:
                    continue
                sh = np.full_like(A_, np.nan)
                sh[lag:] = A_[:len(A_) - lag]
                acc += np.nan_to_num(sh)
                n += 1
                if n == 28:
                    break
            Xk = np.hstack([Xv, np.log(mk / (1 - mk))[hiA, ri][:, None], (acc / n)[hiA, ri][:, None]]).astype(np.float32)
        clf = HistGradientBoostingClassifier(max_iter=int(cfg.get("max_iter", 400)),
                                             learning_rate=float(cfg.get("lr", 0.08)),
                                             max_leaf_nodes=int(cfg.get("leaves", 63)),
                                             l2_regularization=1.0, early_stopping=False, random_state=RM.SEED,
                                             categorical_features=cat)
        clf.fit(Xk[m_tr], y[m_tr])
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(
            clf.predict_proba(Xk[m_ca])[:, 1], y[m_ca])
        Q[rpos, k - 1, ri[out_m]] = iso.predict(clf.predict_proba(Xk[out_m])[:, 1])
        del Xk
        print("k", k, "train", int(m_tr.sum()), flush=True)
    return Q, rows


# ----------------------------------------------------------------------------- decision-side helpers
def recalibrate(Q, rows, pres, T_EVAL, days=42, d0=14.0, cap=1.5):  # T_EVAL: first day recalibrated
    """Per-unit, per-lead logit offset (exact intercept MLE by Newton), refit daily from matured targets of the
    previous `days` days (rows r with r + k < start of the day); shrunk with d/(d+d0), d = days with data;
    |offset| <= cap."""
    lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
    Apos = pd.Series(np.arange(len(pres)), index=pres.index)
    ri = Apos[rows].values
    Av = pres.values
    L = lg(Q)
    Qr = Q.copy()
    for D in pd.date_range(T_EVAL.normalize(), rows[-1].normalize(), freq="D", tz="UTC"):
        sel_rows = np.where((rows >= D - pd.Timedelta(days=days)) & (rows < D))[0]
        tgt_rows = np.where((rows >= D - H1) & (rows < D + pd.Timedelta(days=1) - H1))[0]
        if not len(tgt_rows):
            continue
        for k in range(1, NK + 1):
            r = sel_rows[(rows[sel_rows] + pd.Timedelta(hours=k)) < D]
            if not len(r):
                continue
            x = L[r, k - 1]
            y = Av[ri[r] + k].astype(np.float64)
            ok = ~np.isnan(x)
            x = np.where(ok, x, 0)
            off = np.zeros(x.shape[1])
            for _ in range(6):                                          # Newton on sum(y - sigmoid(x + off)) = 0
                q = 1 / (1 + np.exp(-(x + off[None, :])))
                g = ((y - q) * ok).sum(0)
                h = ((q * (1 - q)) * ok).sum(0) + 1e-6
                off = np.clip(off + g / h, -5, 5)
            dd = len(np.unique(rows[r].normalize()))
            off = np.clip(off * dd / (dd + d0), -cap, cap)
            Qr[tgt_rows, k - 1] = 1 / (1 + np.exp(-(L[tgt_rows, k - 1] + off[None, :])))
    return Qr


def causal(Q, rows, hours, act0):
    rp = pd.Series(np.arange(len(rows)), index=rows)[hours - H1].values
    P = Q[rp].astype(np.float64)                                    # (H, NK, U); lead j = model k=j+1
    P = P[:, :K_MAX + 1]
    assert not np.isnan(P).any(), "missing forecasts"
    P[:, 0] = np.maximum(P[:, 0], act0)
    return P


def markov_rolling(pres, hours, act0, days=42):
    """Per-unit Markov chain refit every day on the previous `days` days (hours up to D-1, known at D:00)."""
    Ap = pres.values.astype(np.float64)
    pos = pd.Series(np.arange(len(pres)), index=pres.index)
    PM = np.zeros((len(hours), K_MAX + 1, Ap.shape[1]))
    jj = np.arange(K_MAX + 1)
    for D in pd.date_range(hours[0].normalize(), hours[-1].normalize(), freq="D", tz="UTC"):
        hs = np.where((hours >= D) & (hours < D + pd.Timedelta(days=1)))[0]
        if not len(hs):
            continue
        a, b = pos[D - pd.Timedelta(days=days)], pos[D - H1]
        a0, a1 = Ap[a:b], Ap[a + 1:b + 1]
        p11 = (a0 * a1).sum(0) / np.maximum(a0.sum(0), 1)
        p01 = ((1 - a0) * a1).sum(0) / np.maximum((1 - a0).sum(0), 1)
        pi = p01 / np.maximum(1 - p11 + p01, 1e-9)
        lam = p11 - p01
        prev = Ap[pos[hours[hs] - H1].values]
        PM[hs] = pi[None, None] + (prev[:, None, :] - pi[None, None]) * lam[None, None] ** (jj[None, :, None] + 1)
    PM[:, 0] = np.maximum(PM[:, 0], act0)
    return PM


EDGES_I = np.array([0, 1, 3, 12, 1e9])
EDGES_A = np.array([0, 0.5, 1, 2, 4, 8, 24, 1e9])


def state_bins(st):
    bi = np.searchsorted(EDGES_I, st["since_end"], side="right") - 1
    ba = np.searchsorted(EDGES_A, st["elap"], side="right") - 1
    return np.where(st["act"] > 0, 4 + ba, bi)                      # 0..3 inactive, 4..10 active


def semi_markov(pres, hours, us, truth, days=60):
    """Rolling (daily refit, previous `days` days) duration-hazard baseline: P(alert in hour t+j | state bin at t:00),
    pooled over units, with a per-unit level factor on the inactive bins (from lead 6)."""
    span = pd.date_range(hours[0] - pd.Timedelta(days=days + 1), hours[-1], freq="h", tz="UTC")
    st = instant_state(span, us, truth)
    B = state_bins(st)
    act = st["act"]
    Ap = pres.values
    pos = pd.Series(np.arange(len(pres)), index=pres.index)
    spos = pd.Series(np.arange(len(span)), index=span)
    Y = np.stack([Ap[pos[span].values + j] for j in range(K_MAX + 1)], axis=1)   # (S, J, U) clock hour t+j
    P = np.zeros((len(hours), K_MAX + 1, len(us)))
    for D in pd.date_range(hours[0].normalize(), hours[-1].normalize(), freq="D", tz="UTC"):
        hs = np.where((hours >= D) & (hours < D + pd.Timedelta(days=1)))[0]
        if not len(hs):
            continue
        a = spos[D - pd.Timedelta(days=days)]
        b = spos[D - pd.Timedelta(hours=K_MAX + 1)] + 1               # targets t+j <= D-1
        bf, yf = B[a:b], Y[a:b]
        rate = np.full((11, K_MAX + 1), np.nan)
        for bb in range(11):
            m = bf == bb
            if m.any():
                rate[bb] = yf.transpose(0, 2, 1)[m].mean(0)
        rate = np.where(np.isnan(rate), np.nanmean(rate, 0, keepdims=True), rate)
        inact = act[a:b] == 0
        y6, e6 = yf[:, 6, :], rate[bf, 6]
        num = (y6 * inact).sum(0) + 0.5
        den = (e6 * inact).sum(0) + 0.5
        fac = np.clip(num / den, 0.2, 5)
        be = B[spos[hours[hs]].values]
        Pd = rate[be].transpose(0, 2, 1)
        ia = (act[spos[hours[hs]].values] == 0)[:, None, :]
        P[hs] = np.where(ia, np.clip(Pd * fac[None, None, :], 0, 1), Pd)
    return P


def stack(P1, P2, hours, pres, start, days=42, lam=1.0):
    """C2: per lead j, daily refit of y ~ a + b logit(P1) + c logit(P2) on departures t with t + j <= D - 1 h
    in the previous `days` days (not before `start`), pooled over units; ridge `lam` toward (b, c) = (0.5, 0.5)."""
    lg = lambda p: np.clip(np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))), -9, 9)
    Ap = pres.values
    ii = pd.Series(np.arange(len(pres)), index=pres.index)[hours].values
    L1, L2 = lg(P1), lg(P2)
    out = np.zeros_like(P1)
    prior = np.array([0.0, 0.5, 0.5])
    pen = lam * np.array([0.0, 1.0, 1.0])
    for D in pd.date_range(max(hours[0], start).normalize(), hours[-1].normalize(), freq="D", tz="UTC"):
        hs = np.where((hours >= D) & (hours < D + pd.Timedelta(days=1)))[0]
        if not len(hs):
            continue
        for j in range(K_MAX + 1):
            fr = np.where((hours >= max(D - pd.Timedelta(days=days), start)) &
                          (hours + pd.Timedelta(hours=j) <= D - H1))[0]
            w = prior.copy()
            if len(fr) > 48:
                Xf = np.stack([np.ones(L1[fr, j].size), L1[fr, j].ravel(), L2[fr, j].ravel()], 1)
                y = Ap[ii[fr] + j].ravel().astype(np.float64)
                for _ in range(8):
                    q = 1 / (1 + np.exp(-(Xf @ w)))
                    g = Xf.T @ (y - q) - pen * (w - prior)
                    Hm = (Xf * (q * (1 - q))[:, None]).T @ Xf + np.diag(pen) + 1e-6 * np.eye(3)
                    w = w + np.linalg.solve(Hm, g)
            z = w[0] + w[1] * L1[hs, j] + w[2] * L2[hs, j]
            out[hs, j] = 1 / (1 + np.exp(-z))
    return out


# ----------------------------------------------------------------------------- evaluation
def within_auc(Y, p):
    from sklearn.metrics import roc_auc_score
    out = [roc_auc_score(Y[:, u], p[:, u]) for u in range(Y.shape[1]) if 0 < Y[:, u].sum() < len(Y)]
    return float(np.mean(out))


def paired_ci(base, a, b, n=1000, seed=7, block=3):
    """moving-block (block days) bootstrap CI of impr(base, a) - impr(base, b), same resampled blocks."""
    rng = np.random.default_rng(seed)
    nd = len(base) // 24
    B, Aa, Bb = (x[:nd * 24].reshape(nd, 24).sum(1) for x in (base, a, b))
    nb = int(np.ceil(nd / block))
    st = rng.integers(0, nd - block + 1, size=(n, nb))
    i = (st[:, :, None] + np.arange(block)[None, None, :]).reshape(n, -1)[:, :nd]
    v = 100 * (Bb[i].sum(1) - Aa[i].sum(1)) / B[i].sum(1)
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def select_margin(score, pref, eps):
    """switch away from route `pref` only if its expected exposure exceeds the best by more than eps * E[pref]."""
    best = score.argmin(1)
    sp = score[:, pref]
    gain = sp - score[np.arange(len(score)), best]
    return np.where(gain > eps * sp + 1e-9, best, pref)


def evaluate(Q, rows, pres, T_EVAL, T_END, truth, osm=True, units="raion"):
    from sklearn.metrics import roc_auc_score
    import networkx as nx
    us = list(pres.columns)
    hours = pd.date_range(T_EVAL, T_END - pd.Timedelta(hours=K_MAX), freq="h", tz="UTC")
    pre = pd.date_range(RM.START, T_EVAL - pd.Timedelta(hours=K_MAX + 1), freq="h", tz="UTC")
    pre_mask = (pres.index >= RM.START) & (pres.index < T_EVAL)
    act0 = instant_state(hours, us, truth)["act"].astype(np.float64)
    t_ext = (rows[0] + H1 + pd.Timedelta(days=7)).normalize()          # P_recal / markov_roll also before T_EVAL
    hx = pd.date_range(t_ext, hours[-1], freq="h", tz="UTC")
    ax = instant_state(hx, us, truth)["act"].astype(np.float64)
    Qr = recalibrate(Q, rows, pres, t_ext)
    Prx, Pmx = causal(Qr, rows, hx, ax), markov_rolling(pres, hx, ax)
    Pst = stack(Prx, Pmx, hx, pres, t_ext)
    ev_ix = pd.Series(np.arange(len(hx)), index=hx)[hours].values
    F = {"pred": causal(Q, rows, hours, act0), "pred_recal": Prx[ev_ix]}
    F["markov"], lam = RM.markov(pres, hours, pre_mask, act0)
    F["markov_roll"] = Pmx[ev_ix]
    F["semimarkov"] = semi_markov(pres, hours, us, truth)
    F["blend50"] = np.maximum(0.5 * F["pred_recal"] + 0.5 * F["markov_roll"], 0)
    F["blend50"][:, 0] = np.maximum(F["blend50"][:, 0], act0)
    F["stack"] = Pst[ev_ix]
    F["stack"][:, 0] = np.maximum(F["stack"][:, 0], act0)
    print("forecasts ready", flush=True)
    Ap = pres.values
    ii = pd.Series(np.arange(len(pres)), index=pres.index)[hours].values
    lead = {}
    for j in (0, 1, 2, 3, 6, 12, 18):
        Y = Ap[ii + j]
        bs_ref = np.mean((F["markov_roll"][:, j] - Y) ** 2)
        lead[j] = {n: {"auc": float(roc_auc_score(Y.ravel(), P[:, j].ravel())), "wauc": within_auc(Y, P[:, j]),
                       "bss_vs_mkroll": float(1 - np.mean((P[:, j] - Y) ** 2) / bs_ref),
                       "bias_sd": float(np.std(P[:, j].mean(0) - Y.mean(0)))} for n, P in F.items()}
    rr = np.arange(len(hours))

    def run(c):
        oi = [us.index(o) for o in c.obl]
        K = c.metric(hours)
        i_s = int(c.metric(pre).mean(0).argmin())
        ch = {"static": np.full(len(hours), i_s)}
        for n, P in F.items():
            sc = c.expected_metric(P[:, :, oi])
            for e in EPS:
                ch[n if e == 0 else f"{n}@{e:g}"] = select_margin(sc, i_s, e)
        ch["reactive"] = select_margin(c.current_metric(act0[:, oi]), i_s, 0)
        ch["oracle"] = K.argmin(1)
        hh = pd.date_range(hours[0] - pd.Timedelta(days=29), hours[-1], freq="h", tz="UTC")
        Kx = c.metric(hh)
        cs = np.vstack([np.zeros((1, Kx.shape[1])), np.cumsum(Kx, 0)])
        e_ = np.arange(len(hours)) + 29 * 24 - K_MAX
        roll = cs[e_] - cs[e_ - 28 * 24]
        from p3_eval import select
        ch["static_roll"] = select(roll, i_s)
        dT = {k: float(60 * (c.T[v].mean() - c.T[i_s])) for k, v in ch.items()}
        return {k: K[rr, v] for k, v in ch.items()}, dT

    out = {"n_hours": len(hours), "eval": [str(hours[0]), str(hours[-1])], "lam": lam, "truth": truth,
           "base_rate": float(Ap[ii].mean()), "lead": lead, "corridors": {}}
    for key in G.CORRIDORS:
        c0 = C.Corridor(key, C.presence())
        if units == "oblast":
            c = C.Corridor(key, pres)
        else:
            routes = [r for r in RA.corridor_routes_raion(key) if r["id"] in c0.ids]
            c = C.Corridor(key, pres, routes=routes)
        d, dT = run(c)
        out["corridors"][key] = {n: C.impr(d["static"], v) for n, v in d.items() if n != "static"}
        out["corridors"][key]["static_km"] = float(d["static"].mean())
    if osm:
        os.environ.setdefault("P3_GRAPH_DIR", str(C.P3_IN / "roadgraph_osm_primary"))
        import p3_graph_candidates as GC
        from p3_graph_eval import od_pairs
        g, cities = GC.load_graph()
        if units == "oblast":
            gr = g
        else:
            E = pd.read_csv(C.P3_IN / "roadgraph_osm_primary_raion" / "edges.csv")
            gr = nx.Graph()
            for r in E.itertuples():
                gr.add_edge(int(r.u), int(r.v), length_m=r.length_m, time_s=r.time_s,
                            profile=json.loads(r.profile_json), fwd=(int(r.u), int(r.v)))
        csr = GC.Csr(g)
        per, pooled, dts = [], {}, {}
        for a_, b_ in od_pairs(g, cities):
            cand = _cand(GC, g, cities, a_, b_, csr)
            c = C.Corridor("od", pres, routes=[GC.path_route(gr, p, f"{a_}-{b_}#{i}") for i, p in enumerate(cand)])
            d, dT = run(c)
            for k, v in d.items():
                pooled.setdefault(k, []).append(v)
                dts.setdefault(k, []).append(dT[k])
            per.append({"od": f"{a_}-{b_}", **{k: C.impr(d["static"], v) for k, v in d.items() if k != "static"}})
        per = pd.DataFrame(per)
        S = {k: np.sum(v, axis=0) for k, v in pooled.items()}
        o = {"pooled": {k: C.impr(S["static"], v) for k, v in S.items() if k != "static"},
             "better": {k: float((per[k] > 0).mean()) for k in per.columns if k != "od"},
             "worse2": {k: float((per[k] < -2).mean()) for k in per.columns if k != "od"},
             "dT_min": {k: float(np.mean(v)) for k, v in dts.items()},
             "ci_vs_static": {}, "paired": {}}
        for k in S:
            if k != "static":
                o["ci_vs_static"][k] = C.block_bootstrap_impr(S["static"], S[k], block=72)
        sfx = lambda e: "" if e == 0 else f"@{e:g}"
        for a in ("pred", "pred_recal", "blend50", "stack"):
            for e in EPS:
                for b in ("markov", "markov_roll", "semimarkov"):
                    o["paired"][f"{a}{sfx(e)}-{b}{sfx(e)}"] = paired_ci(S["static"], S[a + sfx(e)], S[b + sfx(e)])
                o["paired"][f"{a}{sfx(e)}-static_roll"] = paired_ci(S["static"], S[a + sfx(e)], S["static_roll"])
        out["osm"] = o
        per.to_csv(OUTROOT / NAME / "per_od.csv", index=False)
        np.savez_compressed(OUTROOT / NAME / "pooled.npz", hours=hours.tz_convert(None).values,
                            **{k.replace("@", "_at_"): v for k, v in S.items()})
    return out


_CAND = {}


def _cand(GC, g, cities, a, b, csr):
    key = (a, b)
    cache = OUTROOT / "_cand.json"
    if not _CAND and cache.exists():
        _CAND.update({tuple(k.split("||")): v for k, v in json.loads(cache.read_text()).items()})
    if key not in _CAND:
        _CAND[key] = [list(map(int, p)) for p in GC.penalty_lo(g, cities[a], cities[b], 10, csr=csr)]
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"||".join(k): v for k, v in _CAND.items()}))
    return _CAND[key]


def summary_line(name, ev):
    L6, L1 = ev["lead"][6], ev["lead"][1]
    s = (f"{name:22s} wAUC1 " + "/".join(f"{L1[n]['wauc']:.3f}" for n in ("pred", "pred_recal", "markov_roll", "semimarkov"))
         + " wAUC6 " + "/".join(f"{L6[n]['wauc']:.3f}" for n in ("pred", "pred_recal", "markov_roll", "semimarkov"))
         + f" BSS6vsMk {L6['pred']['bss_vs_mkroll']:.3f}/{L6['pred_recal']['bss_vs_mkroll']:.3f}/{L6['semimarkov']['bss_vs_mkroll']:.3f}"
         + f" biasSD6 {L6['pred']['bias_sd']:.3f}/{L6['pred_recal']['bias_sd']:.3f}")
    if "osm" in ev:
        o = ev["osm"]["pooled"]
        s += (" | OSM " + " ".join(f"{k} {o[k]:.2f}" for k in ("pred", "pred_recal", "markov", "markov_roll",
                                                                  "semimarkov", "blend50", "stack", "static_roll", "reactive", "oracle"))
              + f" better {ev['osm']['better']['pred']:.2f}/{ev['osm']['better']['pred_recal']:.2f}")
    s += " | cor " + " ".join(f"{v['pred']:+.1f}" for v in ev["corridors"].values())
    return s


NAME = None
EXTRA = {}


def main():
    global NAME
    NAME = sys.argv[1]
    cfg = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    truth = cfg.get("truth", "raion")
    win = WIN[cfg.get("win", "val")]
    T_CAL, T_EVAL, T_END = (UTC(win[k]) for k in ("T_CAL", "T_EVAL", "T_END"))
    out = OUTROOT / NAME
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({"cfg": cfg, "win": win}, indent=1))
    t0 = time.time()
    units = cfg.get("units", "raion")
    pres, adj = load_inputs(truth, units)
    src = cfg.get("probs_from")
    if src or (out / "probs.npz").exists():
        z = np.load(OUTROOT / (src or NAME) / "probs.npz", allow_pickle=True)
        Q, rows = z["Q"], pd.DatetimeIndex(z["rows"]).tz_localize("UTC")
    else:
        X = build(pres, adj, cfg.get("feats", ["base"]), truth, end=T_END + pd.Timedelta(hours=1))
        print("features", X.shape, round(time.time() - t0), "s", flush=True)
        Q, rows = train(cfg, X, pres, T_CAL, T_EVAL, T_END)
        np.savez_compressed(out / "probs.npz", Q=Q, rows=rows.tz_convert(None).values)
        del X
    print("trained", round(time.time() - t0), "s", flush=True)
    ev = evaluate(Q, rows, pres, T_EVAL, T_END, truth, osm=cfg.get("osm", True), units=units)
    ev["cfg"] = cfg
    ev["code_md5"] = __import__("hashlib").md5(Path(__file__).read_bytes()).hexdigest()
    ev["minutes"] = (time.time() - t0) / 60
    (out / "eval.json").write_text(json.dumps(ev, indent=1, ensure_ascii=False))
    line = summary_line(NAME, ev)
    print(line, flush=True)
    with open(OUTROOT / "log.txt", "a") as fh:
        fh.write(line + "\n")


if __name__ == "__main__":
    main()
