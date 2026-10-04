# Results

Folder names follow the internal versions of the experiments; the scripts select a folder with
`P3_RESULTS`.

| Folder | Content |
|---|---|
| `p3/` | shared inputs and outputs: road graphs (`roadgraph` corridor graph, `roadgraph_osm_primary`, `roadgraph_osm_secondary`, `roadgraph_osm_primary_raion` with district pieces) and the storage benchmark (`bench_*.json`, `scale_*.json`, `scale_summary.md`) |
| `p3v3/raion/` | hourly presence of alerts for the 119 districts and the 25 oblasts, district adjacency |
| `p3v5/` | forecast of the previous version (unanchored: region identifier, history features, calibrated once): forecasts (`direct_probs.parquet`, look-ahead variant for the ablation), corridor and network evaluation, rolling windows of that forecast (`rolling_summary.json`), sensitivity |
| `p3v6/raion_exp/` | development of the proposed forecast: protocol `PREREG.md`, every run (`config.json`, `eval.json`, `per_od.csv`), stored forecasts (`probs.npz`) of the runs with the frozen configuration (`obl_c2024` development block, `obl_W0`–`obl_W3c` rolling windows, `obl_c2025` and `anchor_final` confirmation tests), `robustness_ci.json`, `log.txt` (one line per run), `_cand.json` (cached candidate routes on OSM primary) |
| `p3v7/` | all experiments of the paper with the proposed forecast (corridors, networks, risk-weighted Dijkstra, tail, multi-criteria, sensitivity, transfer); `direct_probs.parquet` is the proposed forecast exported by `src/p3_c4_export.py` |
| `p3v7mk/` | the same pipeline with the rolling Markov chain |
| `p3v7ml/` | the anchored models with daily recalibration, without the averaging |
| `p3v8/graph_diag/` | spatial propagation features (Section S16): one folder per window and `compare.json` |

Not stored, because the scripts recreate them: the feature cache of the unanchored forecast
(`p3/features.parquet`), the forecasts of the rejected candidates of the development rounds and of the
propagation check, and the forecasts and raw inputs of the US transfer.
