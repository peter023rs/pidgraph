# Mirror hazop-ai's output contracts instead of designing fresh ones

pidgraph is a standalone product, but its two outputs deliberately copy
another project's internal formats: the plant model is emitted as DEXPI JSON
identical in shape to hazop-ai's `plant_model_dexpi.json`, and the Neo4j
graph uses hazop-ai's `s2_pml` schema (equipment-level nodes, `FLOWS_TO`
with evidence source for known direction, `CONNECTED_TO` for unknown —
direction is never guessed). We chose this over a fresh, possibly cleaner
schema because pidgraph's endgame is reintegration into hazop-ai once proven:
mirroring makes that reintegration "point hazop-ai at the same database,"
keeps its existing Cypher queries, dashboard and Stage 2 adapter working
unchanged, and imports a direction-honesty model we already trust.

## Consequences

- Schema changes should be coordinated with hazop-ai's `s2_pml`, or the
  reintegration guarantee erodes.
- A constraint not visible in the code: the corpus is sensitive
  (Chinese oil-company drawings) — it must never enter git and must never
  be sent to cloud APIs; all inference runs locally.
