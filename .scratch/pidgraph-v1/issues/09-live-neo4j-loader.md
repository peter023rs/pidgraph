# 09 — Live Neo4j loader behind the GraphStore seam

**What to build:** An operator loads an extracted plant model into a live local Neo4j instance in the s2_pml schema, selected by configuration behind the GraphStore seam — so existing Cypher queries, the Graph Explorer, and later hazop-ai reintegration work unchanged. Offline Cypher-script emission remains the default store output: a run needs no live database to be useful.

**Blocked by:** 01 (Walking skeleton).

**Status:** resolved

- [x] A live Neo4j GraphStore implementation exists behind the seam and is selected by configuration; the Cypher-script emitter stays the default.
- [x] Loaded graphs conform to s2_pml per ADR-0001: equipment-level nodes, FLOWS_TO with evidence source for known direction, CONNECTED_TO otherwise.
- [x] Loading is local-only — the loader connects to an operator-controlled local database, never a remote endpoint; no component calls a network endpoint at inference time.
- [x] Schema conformance is asserted offline against the emitted Cypher (prior art: s2_pml's offline to_cypher path); the test suite never requires a live Neo4j.
- [x] Loading the same run twice is safe (idempotent or clean-replace, stated and tested at the Cypher level).

## Comments

Implemented as `src/pidgraph/neo4j_store.py` behind the GraphStore seam:
`PipelineConfig(graph_store="neo4j")` selects it; `cypher-script` remains the
default. The loader executes exactly the statements the offline emitter
produces (`cypher_store.to_cypher_statements`, shared source), so the offline
schema-conformance tests cover live loads by construction. Connection
settings come from `PIDGRAPH_NEO4J_URI/USER/PASSWORD/DATABASE`; any URI that
is not a loopback `bolt://`/`neo4j://` endpoint is refused at construction
and again at connection time. Idempotency chosen (not clean-replace): the
constraint is IF NOT EXISTS and every statement MERGEs on the unique tag,
tested at the Cypher level. Driver package is the optional `pidgraph[neo4j]`
extra; the suite uses a fake driver, never a live database.

Noted pre-existing limitation (out of scope, shared with the offline
emitter since ticket 01): two runs between the same node pair collapse into
one relationship under the unparameterized `MERGE (a)-[r:TYPE]->(b)`.
Changing edge keying is an s2_pml semantics change to coordinate with
hazop-ai per ADR-0001.
