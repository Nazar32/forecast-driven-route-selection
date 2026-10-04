"""
Paper 3 (v4): export the C4 forecast and its components in the format of the paper's pipeline
(direct_probs.parquet: feature row `hour`, region `oblast`, model k = 1..19, probability p), so that every
experiment of the paper (corridors, networks, risk-weighted Dijkstra, multi-criteria, sensitivity, tail, ...)
runs unchanged with P3_RESULTS pointing to the exported folder.

C4 = 0.5 * (anchored ML, per-region daily recalibration) + 0.5 * (rolling Markov, 42 days), see
results/p3v6/raion_exp/PREREG.md. Row r serves the departure r + 1 h; the recalibration offsets and the
rolling-Markov parameters are those of the departure day (data up to the previous hour only).
Lead 0 is max(active at departure, p) inside the pipeline (p3_core.causal_P), as in the harness.

  python p3_c4_export.py RUN OUT_C4 [OUT_MARKOV_ROLL] [OUT_ML_RECAL]
RUN is a p3_raion_exp run with "units": "oblast" (results/p3v6/raion_exp/RUN/probs.npz + config.json).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import p3_raion_exp as X  # noqa: E402

H1 = pd.Timedelta(hours=1)


def markov_rows(pres, rows, days=42):
    """rolling-Markov forecasts for feature rows (departure r + 1 h), models k = 1..NK."""
    Ap = pres.values.astype(np.float64)
    pos = pd.Series(np.arange(len(pres)), index=pres.index)
    out = np.zeros((len(rows), X.NK, Ap.shape[1]), np.float32)
    kk = np.arange(1, X.NK + 1)
    dep = rows + H1
    for D in pd.date_range(dep[0].normalize(), dep[-1].normalize(), freq="D", tz="UTC"):
        ix = np.where((dep >= D) & (dep < D + pd.Timedelta(days=1)))[0]
        if not len(ix):
            continue
        a, b = pos[D - pd.Timedelta(days=days)], pos[D - H1]
        a0, a1 = Ap[a:b], Ap[a + 1:b + 1]
        p11 = (a0 * a1).sum(0) / np.maximum(a0.sum(0), 1)
        p01 = ((1 - a0) * a1).sum(0) / np.maximum((1 - a0).sum(0), 1)
        pi = p01 / np.maximum(1 - p11 + p01, 1e-9)
        lam = p11 - p01
        prev = Ap[pos[rows[ix]].values]
        out[ix] = pi[None, None] + (prev[:, None, :] - pi[None, None]) * lam[None, None] ** kk[None, :, None]
    return out


def write(P, rows, us, path):
    path.mkdir(parents=True, exist_ok=True)
    R, K, U = P.shape
    df = pd.DataFrame({"hour": np.repeat(rows.tz_convert(None).values, K * U),
                       "oblast": np.tile(np.array(us, dtype=object), R * K),
                       "k": np.tile(np.repeat(np.arange(1, K + 1, dtype=np.int8), U), R),
                       "p": P.reshape(-1).astype(np.float32)})
    df["hour"] = pd.to_datetime(df["hour"]).dt.tz_localize("UTC")
    df["oblast"] = df["oblast"].astype(str)
    df.to_parquet(path / "direct_probs.parquet")
    print("wrote", path / "direct_probs.parquet", len(df), flush=True)


def main():
    run = sys.argv[1]
    outs = [Path(a) for a in sys.argv[2:]]
    cfg = json.loads((X.OUTROOT / run / "config.json").read_text())
    assert cfg["cfg"].get("units") == "oblast", "C4 export needs an oblast-unit run"
    win = cfg["win"]
    T_EVAL, T_END = X.UTC(win["T_EVAL"]), X.UTC(win["T_END"])
    pres, _ = X.load_inputs("oblast_any", "oblast")
    us = list(pres.columns)
    z = np.load(X.OUTROOT / run / "probs.npz", allow_pickle=True)
    Q, rows = z["Q"], pd.DatetimeIndex(z["rows"]).tz_localize("UTC")
    Qr = X.recalibrate(Q, rows, pres, T_EVAL)
    sel = np.where((rows >= T_EVAL - H1) & (rows <= T_END))[0]
    r_sel = rows[sel]
    MK = markov_rows(pres, r_sel)
    MLr = Qr[sel]
    assert not np.isnan(MLr).any()
    write(0.5 * MLr + 0.5 * MK, r_sel, us, outs[0])
    if len(outs) > 1:
        write(MK, r_sel, us, outs[1])
    if len(outs) > 2:
        write(MLr, r_sel, us, outs[2])


if __name__ == "__main__":
    main()
