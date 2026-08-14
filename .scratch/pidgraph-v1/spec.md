# Spec: pidgraph v1 — scanned P&ID digitization to DEXPI JSON and Neo4j

Status: ready-for-agent

## Problem Statement

Peter has access to a large corpus of real P&ID drawings from Chinese oil
companies — more than ten Documents of 200–500 Sheets each, in multiple
company conventions, arriving in about a month. Many are scans. The drawings
are sensitive: they can never be sent to a cloud API and never enter git.

Today there is no way to turn these scans into a trustworthy plant topology
graph. hazop-ai's deterministic extractor only understands one vector
drawing convention, and its DiagEx fallback depends on a cloud VLM — which
is undeployable on this data, expensive per sheet, non-deterministic, and
measurably weak at exactly what matters (connectivity F1 0.29, repeated
symbols missed, 7.6% tag near-misses). Process engineers have no way to
verify or correct machine output, so even a good extraction would not be
trusted.

## Solution

pidgraph: standalone, fully local software that digitizes scanned P&ID
Documents. A scanned PDF plus a per-company Convention Profile goes in; the
Raster Path extracts symbols, line networks, and text; the result is a plant
topology graph with per-element confidence and provenance, emitted as DEXPI
JSON and loaded into Neo4j using the FLOWS_TO / CONNECTED_TO honesty model
(direction only with evidence, never guessed).

Process engineers check the output in the Review Workbench — the digitized
P&ID overlaid on the original Sheet, verdicts pass / reject / edit — and
every verdict is stored as a labeled example, feeding per-company
fine-tuning so the system improves on exactly the deployed corpus. "Works"
is defined by pinned component gates plus a product KPI measured in the
Workbench; once proven, pidgraph reintegrates with hazop-ai by pointing it
at the same database (ADR-0001).

## User Stories

1. As an operator, I want to run pidgraph on a scanned P&ID Document and get a DEXPI JSON plant model out, so that a scan becomes structured data without any manual redrawing.
2. As an operator, I want the intake to detect a vector-PDF Document and refuse it with a clear message, so that v1 never silently produces raster-quality output for a document that deserves the deterministic path.
3. As an operator, I want to process a 200–500 Sheet Document as a batch run with per-Sheet progress and a resumable state, so that one bad Sheet doesn't cost me the night's run.
4. As an operator, I want every run to work fully offline on my own hardware, so that sensitive drawings never touch a cloud API.
5. As an operator, I want scanned Sheets normalized (deskewed, resolution-normalized, binarized) before extraction, so that detector and OCR accuracy doesn't depend on scanner quirks.
6. As an onboarding engineer, I want to create a Convention Profile for a new company — its Legend Dictionary, tag grammar, and line-type semantics — so that the engine adapts to that company's drawing convention.
7. As an onboarding engineer, I want Convention Profiles stored as versioned artifacts, so that a re-run of an old Document with an old profile is reproducible.
8. As a process engineer, I want symbols on a Sheet detected and classified against the company's Legend Dictionary, so that valves, instruments, and equipment are found even when the convention is company-specific.
9. As a process engineer, I want the pipe line network extracted with junctions resolved and runs split at symbol boundaries, so that connectivity reflects the real process — not polylines traced through valves.
10. As a process engineer, I want instrument tags, line numbers, and equipment tags read by OCR and corrected against the company's tag grammar, so that a smudged "O" never becomes a wrong tag silently.
11. As a process engineer, I want flow direction asserted only when there is evidence (arrows, connector text), with everything else left explicitly undirected, so that I never inherit a guessed direction.
12. As a process engineer, I want every extracted element to carry its confidence and provenance (which component produced it, from what evidence), so that I know what to check first.
13. As a reviewer, I want the digitized P&ID overlaid on the original Sheet in the Review Workbench, so that checking the extraction is visual and fast.
14. As a reviewer, I want detections presented in confidence-sorted queues, so that my minutes go to the elements most likely to be wrong.
15. As a reviewer, I want to pass, reject, or edit each detection — with edit letting me supply the corrected tag text or geometry — so that fixing a mistake takes one action.
16. As a reviewer, I want my verdicts saved as labeled examples automatically, so that reviewing is simultaneously training-data creation.
17. As a reviewer, I want to see which Sheets of a Document are fully reviewed and which are untouched, so that a 400-Sheet review can be split across days.
18. As an ML developer, I want reviewer verdicts exportable as a training set per Convention Profile, so that per-company fine-tuning has supervised data.
19. As an ML developer, I want a dev-time label factory that renders hazop-ai's 2401 vector artifacts to raster with projected ground-truth boxes and strings, so that detector and OCR training/eval data exists before the corpus arrives.
20. As an ML developer, I want synthetic degradation (blur, skew, noise, compression) applied to rendered Sheets, so that models trained on clean renders survive real scans.
21. As an ML developer, I want an eval harness with pinned component gates — symbol F1, tag exact-match, connectivity F1 — runnable on demand against labeled eval sets, so that any model or prompt change is gated by numbers, not impressions.
22. As an ML developer, I want the harness to compare per-company fine-tuning against Legend Dictionary nearest-neighbor classification, so that the cheaper adaptation mechanism can win on evidence.
23. As an operator, I want the product KPI — corrections per 100 symbols and reviewer minutes per accepted Sheet — computed from Workbench activity, so that "the software works" is measurable on real corpus Sheets.
24. As an operator, I want the extracted plant model loaded into Neo4j in the s2_pml schema, so that existing Cypher queries, the Graph Explorer, and later hazop-ai reintegration work unchanged.
25. As an operator, I want an offline Cypher-script export as the default store output, so that a run needs no live database to be useful.
26. As a HAZOP analyst, I want the DEXPI JSON to match hazop-ai's plant-model contract exactly, so that Stage 2 can consume a pidgraph run with zero adaptation.
27. As an operator, I want failed Sheets and low-confidence extractions reported as failures and gaps — never silently dropped or papered over — so that the output's coverage is honest.
28. As a developer, I want the full test suite to run with no GPU, no trained models, no network, and no Neo4j, so that CI is fast and deterministic on every commit.

## Implementation Decisions

- pidgraph is a standalone repository and product. It deliberately mirrors
  hazop-ai's output contracts — DEXPI JSON in the shape of hazop-ai's plant
  model, and the s2_pml Neo4j schema (FLOWS_TO with evidence source,
  CONNECTED_TO otherwise) — per ADR-0001. Schema changes must be
  coordinated with hazop-ai or the reintegration guarantee erodes.
- v1 intake is raster-only. Vector PDFs are detected at intake and refused
  with a clear message; there is no router and no deterministic vector path
  in v1 (that returns at reintegration time).
- The Raster Path runs per Sheet: raster normalization → symbol detection →
  line network extraction (binarize → thin → vectorize → junction analysis →
  split runs at symbol boxes) → OCR → lexicon-constrained decoding → graph
  assembly → DEXPI JSON emission → graph store.
- One top seam: digitize(Document, Convention Profile) → run artifacts
  (per-Sheet detections + DEXPI JSON). All product behavior is observable
  and testable at this seam.
- Three component seams, each with an offline deterministic default and a
  real implementation selected by configuration: SymbolDetector (stub /
  trained detector), TextRecognizer (stub / the chosen OCR engine),
  GraphStore (Cypher-script emission / live Neo4j loader). This is
  hazop-ai's proven seam pattern.
- The lexicon-constrained decoder is a pure function above the
  TextRecognizer seam: (candidate strings + tag grammar) → corrected tags
  with correction provenance. It is engine-independent by construction.
- Line network extraction is classical CV and deterministic; it gets no
  seam and no ML.
- The OCR engine choice is deliberately deferred behind the TextRecognizer
  seam. Pinned requirements: mixed Chinese + Latin recognition, rotated and
  vertical text, fully local execution. The eval harness picks the engine
  empirically.
- A Convention Profile is a versioned bundle: Legend Dictionary (symbol
  glyphs + semantics), tag grammar (lexicons/patterns per text class), and
  line-type semantics. In v1 profiles are built manually per company;
  automated Legend Sheet ingest is future work.
- Per-company adaptation: fine-tuning the detector on that company's
  labeled examples is the plan of record. Legend Dictionary
  nearest-neighbor classification is implemented as the candidate cheap
  alternative and judged by the harness; architecture is identical either
  way, only onboarding cost differs.
- Flow direction is seeded only from evidence readable off the Sheet
  (flow arrows detected as symbols, off-page-connector direction text) and
  propagated conservatively; disagreements surface as conflicts, never
  overwrites. Elements without evidence stay CONNECTED_TO.
- The Review Workbench is a local web application: Sheet overlay, confidence-
  sorted queues, pass / reject / edit verdicts, verdict storage as labeled
  examples keyed to Convention Profile and Sheet. Its data comes only from
  run artifacts — it never re-runs extraction.
- The eval harness is a separate concern from the test suite: the harness
  runs real models against labeled eval data to produce gate scores
  (provisional gates: symbol F1 ≥ 0.90, tag exact-match ≥ 0.98,
  connectivity F1 ≥ 0.70, refined when the corpus lands); the test suite
  runs stubs offline on every commit.
- Definition of "works" (the reintegration trigger): all component gates
  green on the eval set AND the product KPI measured in the Workbench on
  real corpus Sheets (corrections per 100 symbols, reviewer minutes per
  accepted Sheet) judged acceptable by the operator.
- Dev-time training/eval data before the corpus: render hazop-ai's 2401
  vector artifacts to raster, project the deterministic detections and text
  spans down as pixel-level labels, and apply synthetic degradation. This
  is a dependency on hazop-ai's artifacts, not its code, and dissolves as
  Workbench corrections accumulate.
- Sensitive-data rules are structural: the corpus and all derived drawing
  content live outside git (enforced by ignore rules), and no component may
  call a network endpoint at inference time.

## Testing Decisions

- Tests assert external behavior at the seams — the artifacts and contracts
  a component emits — never internal implementation details.
- The offline invariant is inherited from hazop-ai and non-negotiable: the
  entire suite runs with no GPU, no trained models, no network, no Neo4j,
  and no API keys. Prior art: hazop-ai's suite, including the DiagEx
  fallback tests that run against a mocked extractor.
- Pipeline-seam tests digitize tiny synthetic Sheets — programmatically
  drawn fixtures with exactly known symbols, lines, and tag strings — and
  assert on the emitted DEXPI JSON and detection records.
- TextRecognizer stub tests seed realistic OCR noise (O/0, S/5 flips) so
  the lexicon-constrained decoder's corrections are proven deterministically.
- The lexicon decoder is additionally tested as a pure function across tag
  grammars and ambiguity cases (fail-closed when no lexicon entry fits).
- GraphStore tests assert emitted Cypher and schema conformance offline;
  prior art: s2_pml's offline to_cypher path.
- Workbench tests use the Flask test client against prepared run artifacts,
  asserting overlays render, queues sort by confidence, and verdicts
  persist as labeled examples; prior art: hazop-ai's s1_dim app tests.
- The eval harness is not part of the test suite and never runs in CI.

## Out of Scope

- Vector-PDF extraction, the document-type router, and any deterministic
  geometry path (returns at hazop-ai reintegration).
- Automated Legend Sheet ingest (profiles are hand-built in v1).
- The local-VLM arbitration tier.
- DEXPI/Proteus XML export (DEXPI JSON is the canonical output).
- Revision diffing, plant-wide tag search, completeness checks, and other
  graph-derived product features.
- hazop-ai integration itself (the contracts guarantee it; the work happens
  after v1 is proven).
- Multi-user auth, deployment packaging, or delivering the software to the
  oil company (v1 user is the operator on local hardware).
- Any cloud inference, telemetry, or data egress.

## Further Notes

- Deliberately deferred by the owner: milestone sequencing within v1, the
  OCR engine choice, and the exact set of text classes in scope (candidate
  classes: instrument tags, line numbers, equipment tags, off-page-connector
  direction text, equipment labels; free-text notes are unlikely for v1).
- The corpus arrives ~September 2026; written permission for its use is
  being arranged and its vector/scan/convention mix should be surveyed on
  arrival — the survey picks the pilot Convention Profile.
- Vocabulary lives in CONTEXT.md; the contract-mirroring decision and the
  sensitive-data constraint live in ADR-0001. The older pidcv scaffold is
  superseded by this repo.
- Hardware reality: development and inference on an M5 Pro / 24 GB MacBook;
  detector fine-tuning may use the second LAN machine if needed.
