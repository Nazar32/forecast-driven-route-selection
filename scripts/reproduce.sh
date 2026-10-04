#!/usr/bin/env bash
# Commands that produce the results of the paper, in order. Each block can be run on its own: the inputs
# it needs are stored in the repository (road graphs, district presence, forecasts of the frozen runs).
# Run from the repository root:  bash scripts/reproduce.sh <block>   (blocks: graphs, unanchored, forecast,
# export, pipeline, transfer, propagation, benchmark, tables).  Run model trainings one at a time.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
DEV=(P3_END="2024-12-31 23:00" P3_TEST_START="2024-07-31 15:00")   # development block of the paper
OSMP=(P3_GRAPH_DIR=results/p3/roadgraph_osm_primary P3_CAND=penalty P3_OUT_SUFFIX=_osm_primary P3_DYN_STEP=6 P3_MUS=8,16,32,64)
OSMS=(P3_GRAPH_DIR=results/p3/roadgraph_osm_secondary P3_CAND=penalty P3_OUT_SUFFIX=_osm_secondary P3_DYN_STEP=6 P3_MUS=8,16,32,64)

block=${1:-}
case "$block" in
graphs)       # road graphs (already in results/p3); the OSM graphs need data/osm/ukraine-roads.osm.pbf
  $PY src/p3_roadgraph.py
  $PY src/p3_osm_graph.py --level primary
  $PY src/p3_osm_graph.py --level secondary
  $PY src/p3_osm_graph_raion.py --level primary
  P3_RESULTS=p3v3 $PY src/p3_raion_model.py --phase data        # district presence and adjacency (results/p3v3/raion)
  ;;
unanchored)   # forecast of the previous version (region identifier, history features): reference rows and ablation
  env P3_RESULTS=p3v5 "${DEV[@]}" $PY src/p3_direct_horizon.py
  env P3_RESULTS=p3v5 "${DEV[@]}" P3_LEAKY=1 $PY src/p3_direct_horizon.py
  env P3_RESULTS=p3v5 "${DEV[@]}" $PY src/p3_eval.py
  env P3_RESULTS=p3v5 "${DEV[@]}" $PY src/p3_graph_eval.py
  env P3_RESULTS=p3v5 "${DEV[@]}" "${OSMP[@]}" $PY src/p3_graph_eval.py
  env P3_RESULTS=p3v5 "${DEV[@]}" "${OSMS[@]}" $PY src/p3_graph_eval.py
  env P3_RESULTS=p3v5 "${DEV[@]}" $PY src/p3_rolling.py --phase train
  env P3_RESULTS=p3v5 "${DEV[@]}" $PY src/p3_rolling.py --phase eval
  ;;
forecast)     # proposed forecast: development rounds, rolling windows, both confirmation tests (PREREG.md)
  $PY scripts/rerun_forecast_runs.py obl_c2024 obl_W0 obl_W1 obl_W2 obl_W3c obl_c2025 anchor_final
  # every round of the protocol, including the rejected candidates: scripts/rerun_forecast_runs.py --all
  ;;
export)       # proposed forecast, rolling Markov chain and the anchored models alone in the pipeline format
  $PY src/p3_c4_export.py obl_c2024 results/p3v7 results/p3v7mk results/p3v7ml
  ;;
pipeline)     # all experiments of Sections VI-A to VI-J with the proposed forecast (p3v7) and the rolling chain (p3v7mk)
  for R in p3v7 p3v7mk; do
    env P3_RESULTS=$R "${DEV[@]}" $PY src/p3_eval.py
    env P3_RESULTS=$R "${DEV[@]}" $PY src/p3_tradeoff.py
    env P3_RESULTS=$R "${DEV[@]}" P3_CASE=kharkiv_lviv $PY src/p3_extra.py
    env P3_RESULTS=$R "${DEV[@]}" $PY src/p3_graph_eval.py
    env P3_RESULTS=$R "${DEV[@]}" "${OSMP[@]}" $PY src/p3_graph_eval.py
    env P3_RESULTS=$R "${DEV[@]}" "${OSMS[@]}" $PY src/p3_graph_eval.py
    env P3_RESULTS=$R "${DEV[@]}" $PY src/p3_review_extra.py
  done
  env P3_RESULTS=p3v7 "${DEV[@]}" $PY src/p3_infocriteria.py        # static criteria (needs pyosmium)
  env P3_RESULTS=p3v7 "${DEV[@]}" $PY src/p3_multicrit.py
  env P3_RESULTS=p3v7 "${DEV[@]}" $PY scripts/rerun_sensitivity.py
  ;;
transfer)     # severe weather in the United States (needs scripts/download_transfer_data.sh)
  env P3_RESULTS=p3v7 $PY src/p3_transfer_graph.py
  env P3_RESULTS=p3v7 $PY src/p3_transfer.py --phase data
  env P3_RESULTS=p3v7 $PY src/p3_transfer.py --phase train
  env P3_RESULTS=p3v7 $PY src/p3_transfer.py --phase eval
  env P3_RESULTS=p3v7 $PY src/p3_transfer_c4.py
  env P3_RESULTS=p3v7 "${DEV[@]}" $PY src/p3_transfer_lead.py
  ;;
propagation)  # spatial propagation features (Section S16), development data only
  for w in c2024 W0 W1 W2 W3c; do $PY src/p3_graph_diag.py g_$w "{\"win\": \"$w\"}"; done
  $PY src/p3_graph_diag_cmp.py
  ;;
benchmark)    # storage backends; start the databases first: docker compose -f benchmark/docker-compose.yml up -d
  for b in memory postgres neo4j; do
    for g in roadgraph roadgraph_osm_primary roadgraph_osm_secondary; do $PY src/p3_graphdb_scale.py --graph $g --backend $b; done
  done
  $PY src/p3_scale_summary.py
  ;;
tables)       # tables and figures of the paper into paper/
  env P3_RESULTS=p3v7 "${DEV[@]}" $PY src/p3_supp_tables.py
  env P3_RESULTS=p3v5 "${DEV[@]}" $PY src/p3_supp_tables.py ablation
  env P3_RESULTS=p3v7 P3_MKROLL=p3v7mk "${DEV[@]}" $PY src/p3_supp_perod_table.py
  env P3_RESULTS=p3v7 "${DEV[@]}" P3_CASE=kharkiv_lviv $PY src/p3_paper_figs.py map pareto graph backends lead netfront perod hyst case sens rolling4
  ;;
*)
  sed -n '2,6p' "$0"; exit 1 ;;
esac
