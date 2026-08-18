# 07 — Flow direction asserted only from evidence

**What to build:** Flow direction appears in the output only when there is evidence readable off the Sheet — flow arrows detected as symbols, or off-page-connector direction text — and is propagated conservatively along the line network. A process engineer never inherits a guessed direction: connections with evidence become FLOWS_TO carrying their evidence source; everything else stays explicitly CONNECTED_TO; disagreements surface as conflicts, never overwrites.

**Blocked by:** 05 (Line network extraction), 06 (Lexicon-constrained decoding).

**Status:** resolved

- [x] A synthetic fixture with a flow arrow on a run yields FLOWS_TO for that connection in DEXPI JSON and Cypher, carrying the evidence source (the arrow detection).
- [x] A fixture with off-page-connector direction text yields FLOWS_TO seeded from that text, carrying its evidence source.
- [x] A fixture with no direction evidence stays entirely CONNECTED_TO — no FLOWS_TO appears anywhere without an evidence source.
- [x] Conservative propagation extends direction along unbranched runs only as far as the evidence justifies; propagation provenance distinguishes seeded from propagated direction.
- [x] A fixture with contradictory evidence (arrows opposing) surfaces a conflict in the run artifacts; neither direction overwrites the other.
- [x] The honesty model mirrors s2_pml per ADR-0001: FLOWS_TO always with evidence source, CONNECTED_TO otherwise. Offline test invariant holds.

## Comments

Implemented in `src/pidgraph/assemble.py`. A Run now accumulates
`FlowEvidence` (orientation, source kind, evidence id, propagated flag)
instead of holding one overwritable direction: arrows seed kind "arrow",
off-page-connector direction text ("TO ..."/"FROM ...", new text class
`opc_direction`) seeds kind "connector" on every run attached to the OPC —
both kinds already in the mirrored contract's flowDirectionSource vocab.
Propagation is conservative: direction crosses a junction only where
exactly two runs meet (a plain continuation); branches, terminals, dead
ends, and conflicts stop it. Propagated evidence keeps its seed's identity
and is emitted as `propagated(arrow:<id>)` in graph edge sources and as
flowDirectionSource "propagated" in DEXPI. Evidence disagreeing on one
run's orientation surfaces as edge `direction: "conflict"` with
`direction_conflicts` (Cypher: CONNECTED_TO carrying both refs); no
orientation is asserted anywhere for a conflicted run. Two-terminal
junction chains whose runs all agree become a single directed FLOWS_TO
edge carrying every evidence ref. Tests: `tests/test_flow_direction.py`
(digitize-level fixtures for each evidence rule plus assembly-level
propagation mechanics).
