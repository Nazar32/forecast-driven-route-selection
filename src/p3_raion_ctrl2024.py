"""
Control: the paper's oblast-level forecasts (results/p3v5, test 31.07-31.12.2024) evaluated in the raion-experiment
harness, i.e. with the same decision code and the same rolling baselines (rolling Markov, rolling semi-Markov,
rolling static) as the raion experiments. Units are raions, every raion gets the forecast of its oblast; truth
"oblast_any" (in 2024 practically identical to the oblast-level truth of the paper).
Output: results/p3v6/raion_exp/ctrl2024/eval.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import p3_raion as RA  # noqa: E402
import p3_raion_exp as X  # noqa: E402

T_CAL, T_EVAL, T_END = X.UTC("2024-06-01 00:00"), X.UTC("2024-07-31 15:00"), X.UTC("2024-12-31 23:00")


def main():
    X.NAME = "ctrl2024"
    (X.OUTROOT / X.NAME).mkdir(parents=True, exist_ok=True)
    pres, _ = X.load_inputs("oblast_any")
    us = list(pres.columns)
    dp = pd.read_parquet(X.C.RES / "p3v5" / "direct_probs.parquet")
    dp["hour"] = pd.to_datetime(dp["hour"], utc=True)
    rows = pd.date_range(T_CAL - X.H1, T_END, freq="h", tz="UTC")
    dp = dp[dp["hour"].isin(rows)]
    obl = sorted(dp["oblast"].unique())
    W = pd.pivot_table(dp, index="hour", columns=["k", "oblast"], values="p").reindex(rows)
    Q = np.full((len(rows), X.NK, len(us)), np.nan, np.float32)
    for j, u in enumerate(us):
        o = RA.oblast_of(u)
        if o not in obl:
            raise SystemExit(f"no forecast for {o}")
        for k in range(1, X.NK + 1):
            Q[:, k - 1, j] = W[(k, o)].values
    ev = X.evaluate(Q, rows, pres, T_EVAL, T_END, "oblast_any")
    (X.OUTROOT / X.NAME / "eval.json").write_text(json.dumps(ev, indent=1, ensure_ascii=False))
    line = X.summary_line(X.NAME, ev)
    print(line)
    with open(X.OUTROOT / "log.txt", "a") as fh:
        fh.write(line + "\n")


if __name__ == "__main__":
    main()
