# 07 — Flow direction asserted only from evidence

**What to build:** Flow direction appears in the output only when there is evidence readable off the Sheet — flow arrows detected as symbols, or off-page-connector direction text — and is propagated conservatively along the line network. A process engineer never inherits a guessed direction: connections with evidence become FLOWS_TO carrying their evidence source; everything else stays explicitly CONNECTED_TO; disagreements surface as conflicts, never overwrites.

**Blocked by:** 05 (Line network extraction), 06 (Lexicon-constrained decoding).

**Status:** ready-for-agent

- [ ] A synthetic fixture with a flow arrow on a run yields FLOWS_TO for that connection in DEXPI JSON and Cypher, carrying the evidence source (the arrow detection).
- [ ] A fixture with off-page-connector direction text yields FLOWS_TO seeded from that text, carrying its evidence source.
- [ ] A fixture with no direction evidence stays entirely CONNECTED_TO — no FLOWS_TO appears anywhere without an evidence source.
- [ ] Conservative propagation extends direction along unbranched runs only as far as the evidence justifies; propagation provenance distinguishes seeded from propagated direction.
- [ ] A fixture with contradictory evidence (arrows opposing) surfaces a conflict in the run artifacts; neither direction overwrites the other.
- [ ] The honesty model mirrors s2_pml per ADR-0001: FLOWS_TO always with evidence source, CONNECTED_TO otherwise. Offline test invariant holds.

## Comments
