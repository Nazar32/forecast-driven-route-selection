"""
Paper 3: rolling static reference on the four corridors (Table 4, row "Rolling static (28 days)").

Every hour the candidate with the fewest kilometres under alert over the trips of the previous 28 days that
have already ended (departures up to t - 19 h) is chosen; ties keep the static baseline. Evaluation as in
p3_eval.py (reduction of the mean km under alert against the static baseline, 24-h moving-block bootstrap,
change of the mean travel time). Output: results/<P3_RESULTS>/static_roll_corridors.json

  P3_RESULTS=p3v7 P3_END="2024-12-31 23:00" P3_TEST_START="2024-07-31 15:00" python src/p3_static_roll.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import p3_core as C  # noqa: E402
import p3_geo as G  # noqa: E402
from p3_eval import select  # noqa: E402


def main(days=28):
    act_all = C.presence()
    hours = pd.date_range(C.TEST_START, C.TEST_END - pd.Timedelta(hours=C.K_MAX), freq="h", tz="UTC")
    pre = pd.date_range(C.DATA_START, C.TEST_START - pd.Timedelta(hours=C.K_MAX + 1), freq="h", tz="UTC")
    out = {}
    for key in G.CORRIDORS:
        c = C.Corridor(key, act_all)
        K = c.metric(hours)
        i_s = int(c.metric(pre).mean(0).argmin())
        hh = pd.date_range(hours[0] - pd.Timedelta(days=days + 1), hours[-1], freq="h", tz="UTC")
        Kx = c.metric(hh)
        cs = np.vstack([np.zeros((1, Kx.shape[1])), np.cumsum(Kx, 0)])
        e_ = np.arange(len(hours)) + (days + 1) * 24 - C.K_MAX          # trips that have ended by t:00
        ch = select(cs[e_] - cs[e_ - days * 24], i_s)
        rr = np.arange(len(hours))
        base, d = K[:, i_s], K[rr, ch]
        out[key] = {"impr": C.impr(base, d), "ci": C.block_bootstrap_impr(base, d),
                    "dT": float(60 * (c.T[ch].mean() - c.T[i_s]))}
        print(key, round(out[key]["impr"], 2), [round(x, 2) for x in out[key]["ci"]], round(out[key]["dT"], 1), flush=True)
    (C.P3 / "static_roll_corridors.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
