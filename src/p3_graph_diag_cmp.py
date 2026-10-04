"""Compare graph-feature runs (results/p3v8/graph_diag/g_<win>) with the C4 reference runs (p3v6/raion_exp/obl_<win>)."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import p3_raion_exp as X  # noqa: E402

R = X.C.RES
G, REF = R / "p3v8" / "graph_diag", R / "p3v6" / "raion_exp"
out = {}
for w in sys.argv[1:] or ["c2024", "W1", "W2", "W3c"]:
    if not (G / f"g_{w}" / "eval.json").exists():
        continue
    eg, er = (json.loads((d / "eval.json").read_text()) for d in (G / f"g_{w}", REF / f"obl_{w}"))
    zg, zr = (np.load(d / "pooled.npz") for d in (G / f"g_{w}", REF / f"obl_{w}"))
    assert np.allclose(zg["static"], zr["static"])
    st = zg["static"]
    o = {"lead_wauc": {}, "lead_auc": {}}
    for j in eg["lead"]:
        for n in ("pred_recal", "blend50"):
            o["lead_wauc"][f"{n}@{j}"] = [round(er["lead"][j][n]["wauc"], 4), round(eg["lead"][j][n]["wauc"], 4)]
        o["lead_auc"][f"blend50@{j}"] = [round(er["lead"][j]["blend50"]["auc"], 4), round(eg["lead"][j]["blend50"]["auc"], 4)]
    for n in ("pred_recal", "blend50"):
        o[n] = {"ref": er["osm"]["pooled"][n], "graph": eg["osm"]["pooled"][n],
                "graph-ref CI": X.paired_ci(st, zg[n], zr[n]),
                "better_ref": er["osm"]["better"][n], "better_graph": eg["osm"]["better"][n]}
    o["markov_roll"] = er["osm"]["pooled"]["markov_roll"]
    o["cor_ref"] = {k: round(v["pred"], 1) for k, v in er["corridors"].items()}
    o["cor_graph"] = {k: round(v["pred"], 1) for k, v in eg["corridors"].items()}
    out[w] = o
    print(w, json.dumps(o, ensure_ascii=False))
(G / "compare.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
