| Graph | Nodes / edges | Backend | S1 Q=1 | S1 Q=100 | S1 Q=1000 | S2 Yen per query | S3 k=6 per query | Mismatches |
|---|---|---|---|---|---|---|---|---|
| corridor | 86 / 126 | memory | 0.147 ms | 0.431 ms | 0.441 ms | 3.3 ms | 0.014 ms | ref |
| corridor | 86 / 126 | postgres | 2.4 ms | 3.9 ms | 10.7 ms | 0.830 ms | 0.356 ms | 0 |
| corridor | 86 / 126 | neo4j | 28.9 ms | 44.9 ms | 302.6 ms | 7.2 ms | 1.4 ms | 0 |
| OSM primary | 11852 / 17100 | memory | 1.6 ms | 78.1 ms | 755.7 ms | 8109.6 ms | 0.015 ms | ref |
| OSM primary | 11852 / 17100 | postgres | 68.0 ms | 107.8 ms | 551.5 ms | 424.7 ms | 0.385 ms | 0 |
| OSM primary | 11852 / 17100 | neo4j | 90.8 ms | 254.7 ms | 1801.9 ms | 82.8 ms | 1.1 ms | 0 |
| OSM secondary | 19580 / 29114 | memory | 2.8 ms | 162.9 ms | 1583.5 ms | 23.1 s | 0.023 ms | ref |
| OSM secondary | 19580 / 29114 | postgres | 118.3 ms | 184.6 ms | 925.7 ms | 1219.8 ms | 0.354 ms | 0 |
| OSM secondary | 19580 / 29114 | neo4j | 144.9 ms | 392.4 ms | 2822.1 ms | 231.6 ms | 1.1 ms | 0 |
