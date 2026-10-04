#!/usr/bin/env bash
# Inputs of the transfer experiment (severe weather in the United States), not stored in the repository.
# NOAA NCEI Storm Events Database, "details" files 2021-2025, and Natural Earth 1:10m cultural vectors.
# The file names below are the versions used in the paper (NOAA creation dates in the "_c" suffix);
# NOAA replaces files when it revises them, in which case take the current file of the same year.
set -euo pipefail
cd "$(dirname "$0")/../data/transfer"
NOAA=https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles
for f in StormEvents_details-ftp_v1.0_d2021_c20260323.csv.gz StormEvents_details-ftp_v1.0_d2022_c20260625.csv.gz \
         StormEvents_details-ftp_v1.0_d2023_c20260323.csv.gz StormEvents_details-ftp_v1.0_d2024_c20260728.csv.gz \
         StormEvents_details-ftp_v1.0_d2025_c20260819.csv.gz; do
  [ -f "$f" ] || curl -fLO "$NOAA/$f"
done
NE=https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/10m_cultural
for layer in ne_10m_roads ne_10m_admin_1_states_provinces ne_10m_populated_places_simple; do
  for ext in shp shx dbf prj; do
    [ -f "$layer.$ext" ] || curl -fLO "$NE/$layer.$ext"
  done
done
ls -l
