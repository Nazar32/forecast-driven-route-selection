"""Re-run forecast runs of the development protocol with their stored configurations.

Every run folder results/p3v6/raion_exp/<RUN>/ holds config.json with the configuration that produced it.
If the folder contains probs.npz, the stored forecasts are re-evaluated; otherwise the models are trained
first (about 10 min per oblast run and 40 min per district run on two cores; run one at a time).

  python scripts/rerun_forecast_runs.py obl_c2025 anchor_final      # selected runs
  python scripts/rerun_forecast_runs.py --all                        # every run in the protocol
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "results" / "p3v6" / "raion_exp"

names = sys.argv[1:]
if names == ["--all"]:
    names = sorted(p.name for p in RUNS.iterdir() if (p / "config.json").exists())
for name in names:
    cfg = json.loads((RUNS / name / "config.json").read_text())["cfg"]
    script = "p3_raion_ctrl2024.py" if name == "ctrl2024" else "p3_raion_exp.py"
    args = [sys.executable, str(ROOT / "src" / script), name] + ([] if name == "ctrl2024" else [json.dumps(cfg)])
    print(" ".join(args), flush=True)
    subprocess.run(args, check=True)
