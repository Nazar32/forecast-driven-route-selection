# Forecast-driven route selection under time-varying regional risk

Code, data and results for the paper

> N. Melnyk and O. Pysarchuk, "Forecast-Driven Multi-Criteria Route Selection Technology on a Graph
> Database Under Time-Varying Regional Risk," submitted to *IEEE Access*, 2026.

The paper turns hourly forecasts of regional hazards (air-raid alerts in Ukraine as the case study, severe
weather in the United States as a transfer) into road-route decisions. Per-lead gradient-boosting models
forecast the alert probability of every region 0 to 18 hours ahead, the forecasts are anchored to the recent
alert level of each region and averaged with a rolling Markov chain, and every candidate route is scored by
its expected kilometres under alert along its own time profile. Routes are chosen among limited-overlap
candidates with a nonlinear multi-criteria scheme or by risk-weighted shortest paths on the whole network.
The evaluation is non-anticipative: every decision uses only information available at departure.

## Repository layout

```
src/                    all code (module names as in Table S15 of the Supplementary Material)
scripts/
  reproduce.sh          the commands behind every result, grouped in blocks
  rerun_forecast_runs.py  re-runs forecast runs of the development protocol from their stored configs
  rerun_sensitivity.py  re-runs the candidate-generation sensitivity analysis
  download_transfer_data.sh  NOAA and Natural Earth inputs of the US transfer (not stored here)
benchmark/              Docker setup of PostgreSQL 16 + pgRouting and Neo4j 5.26 + GDS
data/                   alert logs, boundaries, OSM road extract, corridor routes (see data/README.md)
results/                results of every experiment in the paper (see results/README.md)
  p3v6/raion_exp/PREREG.md   development protocol written before each round, and both confirmation tests
paper/                  tables and figures written by the scripts
```

## Installation

Python 3.11. The versions used for the paper are pinned in `requirements.txt`.

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`osmium` is needed only to rebuild the OSM road graphs and the static road criteria, `geopandas` only for the
US transfer, `psycopg2-binary` and `neo4j` only for the storage benchmark.

## Quick checks (minutes, no model training)

The forecasts of the frozen configuration are stored, so the central results can be recomputed without
training any model.

```
# both confirmation tests (oblasts, Jan-Apr 2025; districts, Mar-Sep 2026) from the stored forecasts;
# compare blend50 and the paired interval "blend50-markov_roll" in eval.json with Table 5 of the paper
python scripts/rerun_forecast_runs.py obl_c2025 anchor_final

# corridor results of the development block (Table 3) with the proposed forecast
P3_RESULTS=p3v7 P3_END="2024-12-31 23:00" P3_TEST_START="2024-07-31 15:00" python src/p3_eval.py

# supplementary tables, written into paper/
bash scripts/reproduce.sh tables
```

Re-running these commands overwrites the corresponding files in `results/` with identical numbers.

## Full reproduction

`scripts/reproduce.sh` lists every step in order; run it block by block:

| Block | What it does |
|---|---|
| `graphs` | corridor graph, OSM primary and secondary graphs, district profiles, district presence |
| `unanchored` | forecast of the previous version (region identifier, history features), its evaluation, the ablation and the rolling windows of that forecast |
| `forecast` | proposed forecast: development block, rolling windows W0 to W3, both confirmation tests (an oblast training takes about 10 min, a district training about 40 min on two cores) |
| `export` | proposed forecast, rolling Markov chain and the anchored models alone in the format of the pipeline |
| `pipeline` | corridors, networks, risk-weighted Dijkstra, tail and onset analysis, static criteria, multi-criteria selection, sensitivity (OSM secondary is the slow part) |
| `transfer` | severe weather in the United States |
| `propagation` | spatial propagation features (Section S16) |
| `benchmark` | storage backends (needs the databases of `benchmark/`) |
| `tables` | tables and figures into `paper/` |

Model trainings use all cores; run one training at a time.

### Environment variables

| Variable | Meaning |
|---|---|
| `P3_RESULTS` | results folder the scripts read and write (`p3v7` proposed forecast, `p3v7mk` rolling Markov chain, `p3v5` unanchored forecast) |
| `P3_END`, `P3_TEST_START` | end of the data and start of the development block; the paper uses `2024-12-31 23:00` and `2024-07-31 15:00` |
| `P3_GRAPH_DIR` | road graph for the network experiments (`results/p3/roadgraph`, `..._osm_primary`, `..._osm_secondary`) |
| `P3_CAND`, `P3_OUT_SUFFIX`, `P3_DYN_STEP`, `P3_MUS` | candidate generator (`yen` or `penalty`), suffix of the output files, every n-th hour for risk-weighted Dijkstra, risk weights mu |
| `P3_CASE` | corridor of the worked example (`kharkiv_lviv`) |
| `P3_MKROLL` | folder with the rolling-Markov results for the per-OD table (`p3v7mk`) |
| `P3_LEAKY` | `1` trains the look-ahead variant used only in the ablation |

## Where each result comes from

| Paper | Script | Results |
|---|---|---|
| Corridors (Section VI-A) | `p3_eval.py`, `p3_static_roll.py`, `p3_review_extra.py` | `p3v7/eval_summary.json`, `p3v7mk/eval_summary.json`, `p3v7/static_roll_corridors.json`, `p3v7/review_extra.json`, `p3v5/eval_summary.json` (unanchored forecast) |
| Tail exposure, source of the gain (VI-B) | `p3_review_extra.py` | `p3v7/review_extra.json` |
| Ablation, window measure (VI-C) | `p3_eval.py` with `P3_RESULTS=p3v5` | `p3v5/eval_summary.json` |
| Networks, value of K, risk-weighted Dijkstra (VI-D) | `p3_graph_eval.py` | `p3v7/graph_eval_summary*.json`, per-OD CSV files; same files in `p3v7mk`, `p3v5` |
| Sensitivity to candidate generation (VI-D, S4) | `p3_sens.py` | `p3v7/sens/` |
| Rolling windows, components of the forecast (VI-E, S5, S15) | `p3_raion_exp.py` | `p3v6/raion_exp/obl_W0` to `obl_W3c`, development runs listed in `PREREG.md` |
| Confirmation tests (VI-F) | `p3_raion_exp.py` | `p3v6/raion_exp/obl_c2025`, `anchor_final`, `robustness_ci.json` |
| Transfer to severe weather (VI-G) | `p3_transfer_graph.py`, `p3_transfer.py`, `p3_transfer_c4.py`, `p3_transfer_lead.py` | `p3v7/transfer/`, `p3v7/transfer_c4.json` |
| Price of lower exposure: time (VI-H) | `p3_tradeoff.py` | `p3v7/tradeoff_summary.json`, `p3v7/pareto_*.csv` |
| Price of lower exposure: road conditions (VI-I) | `p3_infocriteria.py`, `p3_multicrit.py` | `p3v7/infocriteria*.{csv,json}`, `p3v7/multicrit_summary.json` |
| Operational aspects, worked example (VI-J, S3, S9) | `p3_extra.py` | `p3v7/extra_summary.json` |
| Storage backends (VI-K, S11) | `p3_graphdb_bench.py`, `p3_graphdb_scale.py`, `p3_scale_summary.py` | `p3/bench_*.json`, `p3/scale_*.json`, `p3/scale_summary.md` |
| Size of the effect, propagation features (VII-A, S16) | `p3_graph_diag.py`, `p3_graph_diag_cmp.py` | `p3v8/graph_diag/compare.json` |
| Supplementary tables | `p3_supp_tables.py`, `p3_supp_perod_table.py` | `paper/*.tex` |
| Figures | `p3_paper_figs.py` | `paper/images/` |

## Pre-specification

`results/p3v6/raion_exp/PREREG.md` records, round by round, the candidates and the selection rule of the
forecast before the results of that round were read, the frozen configuration, and the two confirmation tests,
each run once. Every run folder keeps the configuration (`config.json`) and the evaluation (`eval.json`); the
`code_md5` field in `eval.json` is the hash of `p3_raion_exp.py` in the working copy where the run was made.
The file in `src/` differs from that copy only in the file paths of this repository layout, so its hash is
different; the stored forecasts re-evaluated with it give identical numbers.

## Data

The alert log comes from the public dataset of official air-alert declarations
(<https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset>, MIT licence). Road data are
OpenStreetMap (ODbL), corridor routes were generated with OpenRouteService, boundaries come from
geoBoundaries (CC BY 4.0) and OpenStreetMap. Details and attributions are in `data/README.md`.

## Licence

Code: MIT (see `LICENSE`). Data files keep the licences of their sources (`data/README.md`).

## Citation

See `CITATION.cff`. Until the paper is published, please cite it as submitted to IEEE Access, 2026.
