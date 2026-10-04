# Data

| Path | Content | Source and licence |
|---|---|---|
| `alerts/official_data_uk_2026-07.csv.gz` | official air-alert declarations (start, end, oblast, district, community level), snapshot of July 2026; used for the oblast-level pipeline (development block, rolling windows) | [ukrainian-air-raid-sirens-dataset](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset), file `official_data_uk.csv`, MIT licence |
| `alerts/official_data_uk_2026-09-07_dedup.csv.gz` | the same dataset downloaded on 7 September 2026, 113 845 exact duplicate rows removed; used for the district experiments, the forecast of the paper and both confirmation tests | as above |
| `geo/geoBoundaries-UKR-ADM1_simplified.geojson` | oblast boundaries (the city of Kyiv as a separate unit) | [geoBoundaries](https://www.geoboundaries.org), CC BY 4.0 |
| `geo/ukraine_raions_osm_20261001.geojson` | district (raion) boundaries of the 2020 division (OSM `boundary=administrative`, `admin_level=6`) from the Geofabrik Ukraine extract of 1 October 2026, simplified to 0.0003° | © OpenStreetMap contributors, ODbL |
| `osm/ukraine-roads.osm.pbf` | Geofabrik extract of Ukraine (data as of 25 September 2026) reduced to motorway, trunk, primary and secondary roads and their links | © OpenStreetMap contributors, ODbL |
| `corridors/*.json` | candidate routes of the four corridors (Lviv–Kyiv, Odesa–Kyiv, Kharkiv–Lviv, Dnipro–Kyiv) with geometry and travel time, generated with OpenRouteService through different intermediate cities | © openrouteservice.org by HeiGIT, map data © OpenStreetMap contributors, ODbL |
| `transfer/` | inputs of the US transfer, downloaded by `scripts/download_transfer_data.sh`: NOAA NCEI Storm Events Database details files 2021–2025 and Natural Earth 1:10m roads, states and populated places | NOAA (public domain), Natural Earth (public domain) |

All times in the alert log are UTC. Hourly presence is built from the start and end of every record; see
`src/p3_core.py` (oblasts) and `src/p3_raion.py`, `src/p3_raion_model.py` (oblasts and districts).
