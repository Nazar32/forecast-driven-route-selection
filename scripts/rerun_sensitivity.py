"""Re-run the sensitivity analysis of candidate generation (Section S4) with the arguments stored in
results/p3v7/sens/*.json (27 settings on the OSM graphs, two on the corridor graph).

  P3_RESULTS=p3v7 P3_END="2024-12-31 23:00" P3_TEST_START="2024-07-31 15:00" python scripts/rerun_sensitivity.py
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SENS = ROOT / "results" / os.environ.get("P3_RESULTS", "p3v7") / "sens"

for f in sorted(SENS.glob("*.json")):
    d = json.loads(f.read_text())
    a = d["args"]
    env = dict(os.environ, P3_GRAPH_DIR=str(ROOT / "results" / "p3" / d["graph"]))
    cmd = [sys.executable, str(ROOT / "src" / "p3_sens.py"), "--theta", str(a["theta"]), "--stretch", str(a["stretch"]),
           "--factor", str(a["factor"]), "--kmax", str(a["kmax"]), "--maxiter", str(a["maxiter"]),
           "--tag", a["tag"], "--method", a["method"]]
    print(d["graph"], " ".join(cmd[2:]), flush=True)
    subprocess.run(cmd, check=True, env=env)
