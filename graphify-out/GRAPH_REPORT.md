# Graph Report - pidgraph  (2026-08-23)

## Corpus Check
- 90 files · ~82,218 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1460 nodes · 3743 edges · 59 communities (56 shown, 3 thin omitted)
- Extraction: 88% EXTRACTED · 12% INFERRED · 0% AMBIGUOUS · INFERRED: 456 edges (avg confidence: 0.93)
- Token cost: 0 input · 54,351 output

## Community Hubs (Navigation)
- Sheet Annotations & Fixtures
- Cypher Graph Store
- Raster Normalization
- KPI & Activity Log
- PDF Intake & Refusal
- Batch Run Reporting
- Symbol Detector Internals
- Issue Tickets & Spec
- Sheet Assembly & Flow Evidence
- Convention Profile Loading
- Workbench Run Artifacts
- Training-Set Export
- Detection & Provenance
- Domain Glossary & Docs
- Legend NN Classifier
- Eval Metrics & Matching
- Line Network Extraction
- Degradation Transforms
- Label Factory Synthesis
- OCR Eval-Set Generation
- Review Flow Tests
- Degraded Dataset Variants
- Eval Harness CLI
- Trained Symbol Detector
- Batch Resume State
- Lexicon Tag Decoding
- Recognizer Test Backends
- Raster Degradation Ops
- Variant Configuration
- RapidOCR & Offline Guard
- KPI Reporting
- Grayscale PNG Codec
- Workbench Overlay Tests
- Eval Harness Core
- Plant Graph & Pipeline Config
- OCR Grammar Classification
- Label Factory Tests
- Workbench KPI Tests
- Artifact Label Projection
- OCR Engine Selection
- Piping Graph Emitter
- OCR Backend Protocol
- Label Store
- Detector Training
- Render Projection Tests
- Degradation Geometry Mapping
- Verdict Labels
- Rotated Read Merging
- Compression Transform
- OCR Eval-Set Tests
- Apple Vision Backend
- PDF Page Rendering
- Review State
- EasyOCR Backend
- Pinned Gates
- Artifact Validation
- Lossy Detector Stub
- Corpus Constraints
- Pidgraph Root

## God Nodes (most connected - your core abstractions)
1. `Sheet` - 62 edges
2. `digitize()` - 56 edges
3. `LabelStore` - 53 edges
4. `ConventionProfile` - 46 edges
5. `PipelineConfig` - 40 edges
6. `load_profile()` - 37 edges
7. `run_batch()` - 34 edges
8. `make_artifact()` - 32 edges
9. `SymbolDetection` - 31 edges
10. `SheetAnnotations` - 30 edges

## Surprising Connections (you probably didn't know these)
- `Batch Runs with Resume and Gap Reporting` --semantically_similar_to--> `LabelStore`  [INFERRED] [semantically similar]
  .scratch/pidgraph-v1/issues/08-batch-runs-resume-gap-reporting.md → src/pidgraph/labels.py
- `digitize()` --implements--> `digitize() Top Seam`  [EXTRACTED]
  src/pidgraph/pipeline.py → .scratch/pidgraph-v1/spec.md
- `Flow Direction from Evidence` --references--> `FlowEvidence`  [EXTRACTED]
  .scratch/pidgraph-v1/issues/07-flow-direction-from-evidence.md → src/pidgraph/assemble.py
- `Lexicon-Constrained Tag Decoding` --references--> `assemble_sheet()`  [EXTRACTED]
  .scratch/pidgraph-v1/issues/06-lexicon-constrained-decoding.md → src/pidgraph/assemble.py
- `Batch Runs with Resume and Gap Reporting` --references--> `run_batch()`  [EXTRACTED]
  .scratch/pidgraph-v1/issues/08-batch-runs-resume-gap-reporting.md → src/pidgraph/batch.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Three Component Seams with Offline Deterministic Defaults** — _scratch_pidgraph_v1_spec_symboldetector_seam, _scratch_pidgraph_v1_spec_textrecognizer_seam, _scratch_pidgraph_v1_spec_graphstore_seam [EXTRACTED 1.00]
- **Fail-Closed Honesty Pattern (never guess, surface gaps)** — _scratch_pidgraph_v1_issues_06_lexicon_constrained_decoding_lexicon_decoder, _scratch_pidgraph_v1_issues_07_flow_direction_from_evidence_flow_direction_evidence, _scratch_pidgraph_v1_issues_08_batch_runs_resume_gap_reporting_batch_runs, _scratch_pidgraph_v1_issues_17_legend_nn_classifier_legend_nn_classifier, _scratch_pidgraph_v1_issues_18_ocr_engine_selection_ocr_engine_selection [INFERRED 0.85]
- **Training-Data Loop (review to fine-tuning)** — _scratch_pidgraph_v1_issues_10_workbench_overlay_verdicts_workbench_overlay, _scratch_pidgraph_v1_issues_12_training_set_export_training_set_export, _scratch_pidgraph_v1_issues_13_label_factory_label_factory, _scratch_pidgraph_v1_issues_14_synthetic_degradation_synthetic_degradation, _scratch_pidgraph_v1_spec_per_company_fine_tuning [INFERRED 0.85]
- **OCR Engine Candidates Behind the TextRecognizer Seam** — docs_adr_0002_ocr_engine_rapidocr_rapidocr, docs_adr_0002_ocr_engine_rapidocr_apple_vision, docs_adr_0002_ocr_engine_rapidocr_tesseract, docs_adr_0002_ocr_engine_rapidocr_easyocr, docs_adr_0002_ocr_engine_rapidocr_textrecognizer_seam [EXTRACTED 1.00]
- **hazop-ai Output Contract Mirroring** — docs_adr_0001_mirror_hazop_ai_contracts_mirror_hazop_ai_contracts, docs_adr_0001_mirror_hazop_ai_contracts_hazop_ai, docs_adr_0001_mirror_hazop_ai_contracts_s2_pml, docs_adr_0001_mirror_hazop_ai_contracts_plant_model_dexpi_json, context_dexpi_json, context_flows_to, context_connected_to [EXTRACTED 1.00]
- **Agent Workflow Guidance System** — claude_pidgraph, docs_agents_issue_tracker_local_markdown_tracker, docs_agents_triage_labels_triage_labels, docs_agents_domain_domain_docs [EXTRACTED 1.00]

## Communities (59 total, 3 thin omitted)

### Community 0 - "Sheet Annotations & Fixtures"
Cohesion: 0.06
Nodes (73): _load_eval_sheet(), Path, LineAnnotation, Exactly known ground truth carried by a synthetic Sheet., SheetAnnotations, SymbolAnnotation, TextAnnotation, digitize() (+65 more)

### Community 1 - "Cypher Graph Store"
Cohesion: 0.06
Nodes (53): CypherScriptGraphStore, _label(), _lit(), _map(), _node_props(), Path, Offline Cypher-script GraphStore — the default store implementation. Mirrors…, Full idempotent Cypher script for the equipment-level graph (constraint, nodes,… (+45 more)

### Community 2 - "Raster Normalization"
Cohesion: 0.07
Nodes (47): _binarized(), _deskewed(), _estimate_skew(), _ink_points(), _mapped_bbox(), Normalization, normalize_sheet(), _normalized_annotations() (+39 more)

### Community 3 - "KPI & Activity Log"
Cohesion: 0.09
Nodes (42): ActivityLog, _identity(), kpi_report(), main(), _metrics(), Clock, Path, The KPI over a set of Sheet rows — one shape for a Document, a Convention… (+34 more)

### Community 4 - "PDF Intake & Refusal"
Cohesion: 0.09
Nodes (43): _extract_raster(), _image_refusal(), _inspect_stream(), IntakeError, load_document(), Exception, Path, PDF Document intake: enumerate Sheets, refuse vector PDFs, rasterize. v1 intake… (+35 more)

### Community 5 - "Batch Run Reporting"
Cohesion: 0.09
Nodes (34): SheetAssembly, BatchRunResult, _collect_gaps(), _gap(), _manifest(), _print_progress(), _profile_fingerprint(), Batch runs over a multi-Sheet Document (ticket 08). run_batch() is digitize()'s… (+26 more)

### Community 6 - "Symbol Detector Internals"
Cohesion: 0.07
Nodes (39): _box_ink(), _Candidate, _class_variants(), _ClassMatcher, _collect_crops(), _correlation(), _Crop, _ink_bits() (+31 more)

### Community 7 - "Issue Tickets & Spec"
Cohesion: 0.09
Nodes (41): Walking Skeleton: digitize() Seam, PDF Intake and Vector-PDF Refusal, Raster Normalization (Deskew, Scale, Binarize), Convention Profile Versioned Bundle, Line Network Extraction with Junctions, Lexicon-Constrained Tag Decoding, Flow Direction from Evidence, Batch Runs with Resume and Gap Reporting (+33 more)

### Community 8 - "Sheet Assembly & Flow Evidence"
Cohesion: 0.10
Nodes (33): assemble_sheet(), _attach_endpoint(), _center(), _chain_direction(), _dist(), FlowEvidence, _group_junctions(), Junction (+25 more)

### Community 9 - "Convention Profile Loading"
Cohesion: 0.14
Nodes (36): LegendEntry, Semantics of one symbol class in a Legend Dictionary., _identity(), _legend(), _legend_entry(), _line_semantics(), _line_styles(), load_profile() (+28 more)

### Community 10 - "Workbench Run Artifacts"
Cohesion: 0.09
Nodes (33): Flask, document_identifier(), iter_detections(), load_record(), load_sheets(), NamedTuple, Path, Run-artifact readers for the review side — the Review Workbench (ticket 10/11)… (+25 more)

### Community 11 - "Training-Set Export"
Cohesion: 0.11
Nodes (36): main(), make_example(), profile_key(), One directory per Convention Profile identity + version; both parts percent-…, One verdict validated into its labeled-example form. Pass and reject stand…, export_training_set(), main(), Path (+28 more)

### Community 12 - "Detection & Provenance"
Cohesion: 0.14
Nodes (24): _integral_image(), Summed-area table of the ink bits, one extra zero row/column, so any window's…, _suppresses(), ConventionProfile, Provenance, A recognized string, after lexicon-constrained decoding against the Convention…, Per-company adaptation bundle: Legend Dictionary, tag grammar, line-type…, The stamp run artifacts carry: which profile version produced them… (+16 more)

### Community 13 - "Domain Glossary & Docs"
Cohesion: 0.09
Nodes (34): pidgraph Agent Instructions (CLAUDE.md), CONNECTED_TO Relationship, Convention Profile, DEXPI JSON, Document, FLOWS_TO Relationship, Legend Dictionary, Legend Sheet (+26 more)

### Community 14 - "Legend NN Classifier"
Cohesion: 0.11
Nodes (32): LegendNNSymbolDetector, The candidate cheap adaptation mechanism behind the SymbolDetector seam:…, build_nn_sheet(), nn_bundle(), fixture, Path, The Legend Dictionary nearest-neighbor classifier (ticket 17): symbol…, One lone shape on an otherwise blank operating-scale Sheet. (+24 more)

### Community 15 - "Eval Metrics & Matching"
Cohesion: 0.10
Nodes (31): Boxed, bbox_iou(), match_boxes(), Bbox, Tag exact-match: each ground-truth tag is read correctly only if a predicted…, Connectivity F1 over links as a multiset of unordered terminal pairs — the…, Greedy one-to-one matching of predicted to ground-truth boxes: candidate pairs…, score_links() (+23 more)

### Community 16 - "Line Network Extraction"
Cohesion: 0.16
Nodes (30): Pixel, _aligned(), _bridge_flow_arrows(), _crosses_box(), _dash_mergeable(), _dist(), extract_line_network(), _in_box() (+22 more)

### Community 17 - "Degradation Transforms"
Cohesion: 0.13
Nodes (29): Blur, degrade_raster(), Noise, Additive gaussian noise; severity is sigma in gray levels., Apply the transforms in order. Each stage draws from its own stream keyed by…, Box blur; severity is the radius in pixels (window 2r+1)., Synthetic degradation transforms (ticket 14): blur, skew, noise, and…, test_a_full_chain_is_reproducible_and_seed_sensitive() (+21 more)

### Community 18 - "Label Factory Synthesis"
Cohesion: 0.10
Nodes (29): _Attachment, BoxSynthesis, _candidate_strings(), _connectivity_links(), _edge_direction(), _inside(), _line_labels(), _node_bbox_pt() (+21 more)

### Community 19 - "OCR Eval-Set Generation"
Cohesion: 0.13
Nodes (28): clip_bbox(), The box intersected with the canvas — or None when the canvas- preserving…, _advance(), _choose_rotation(), degrade_sheet(), _digits(), _Item, _letters() (+20 more)

### Community 20 - "Review Flow Tests"
Cohesion: 0.14
Nodes (27): client(), _correction_for(), _detections(), _give(), _progress(), fixture, Path, _queue() (+19 more)

### Community 21 - "Degraded Dataset Variants"
Cohesion: 0.15
Nodes (27): The label factory emitting degraded dataset variants (ticket 14): clean and…, scan' and 'Scan' are one directory on a case-insensitive filesystem — the…, The canvas-preserving skew paper-fills ink it rotates off the page: a symbol…, An artifact that fails at the variant stage must contribute to no dataset at…, test_a_failed_batch_writes_no_variant_dataset(), test_a_photometric_variant_keeps_the_clean_geometry(), test_a_variant_write_failure_keeps_the_datasets_aligned(), test_cli_loads_variants_from_the_file() (+19 more)

### Community 22 - "Eval Harness CLI"
Cohesion: 0.13
Nodes (27): compare(), evaluate(), load_eval_set(), main(), Load a labeled eval set — a label factory output directory (labels/,…, Score one configured pipeline against the eval set: gate scores with pass/fail…, Two or more configurations against the same eval set in one report — fine-…, The eval harness (ticket 15): a configured pipeline scored against a labeled… (+19 more)

### Community 23 - "Trained Symbol Detector"
Cohesion: 0.14
Nodes (25): The trained artifact behind the SymbolDetector seam: slides each prototype…, TrainedSymbolDetector, build_glyph_sheet(), fixture, The trained SymbolDetector (ticket 16): a training procedure turns label-…, A dataset the way the label factory lays one out: symbol labels as pass…, A store file whose recorded identity disagrees with its directory key (a case-…, Trained over layouts at both raster parities, so calibrated thresholds price in… (+17 more)

### Community 24 - "Batch Resume State"
Cohesion: 0.14
Nodes (24): ProgressCallback, BatchStateError, _check_manifest(), _load_sheet_state(), Exception, Path, A run directory belongs to one (Document, Convention Profile, configuration)…, Run a Document as a resumable batch with per-Sheet progress and an honest run… (+16 more)

### Community 25 - "Lexicon Tag Decoding"
Cohesion: 0.14
Nodes (24): _decode(), decode_tags(), _noted(), Lexicon-constrained decoding — a pure function above the TextRecognizer seam:…, Every string reachable by swapping confusable characters (the candidate itself…, _swaps(), _unresolved(), _variants() (+16 more)

### Community 26 - "Recognizer Test Backends"
Cohesion: 0.12
Nodes (22): EngineTextRecognizer, OfflineViolation, One engine behind the TextRecognizer seam. The engine reads the (normalized)…, An engine tried to reach the network while reading a Sheet., build_synthetic_sheet(), FakeBackend, OptionRecordingBackend, PhoningHomeBackend (+14 more)

### Community 27 - "Raster Degradation Ops"
Cohesion: 0.13
Nodes (18): _blur_rows(), _blurred_line(), _matmul(), polyline_visible(), Random, _quant_table(), Raster, Synthetic degradation of rendered Sheets (ticket 14): blur, skew, noise, and… (+10 more)

### Community 28 - "Variant Configuration"
Cohesion: 0.11
Nodes (19): _check_number(), load_variants(), Any, Path, One degraded dataset the factory emits beside the clean one: a name (its…, Variants from a JSON file: a non-empty list of variant configs. An empty or…, transform_from_config(), Variant (+11 more)

### Community 29 - "RapidOCR & Offline Guard"
Cohesion: 0.11
Nodes (13): ModuleNotFoundError, _content_version(), _pymupdf(), _missing(), offline_guard(), Path, RapidOCRBackend, Twelve hex digits of the sorted model files' names and bytes: the part of an… (+5 more)

### Community 30 - "KPI Reporting"
Cohesion: 0.12
Nodes (21): attribute_minutes(), basis(), _by_kind(), _counts(), display_kpi(), Event, format_summary(), _metric_lines() (+13 more)

### Community 31 - "Grayscale PNG Codec"
Cohesion: 0.17
Nodes (21): _chunk(), _chunks(), decode_gray_png(), encode_gray_png(), _paeth(), Grayscale PNG encoding and decoding with the stdlib alone, so persisting Sheet…, Encode a row-major 8-bit grayscale raster (the Sheet.raster form) as a PNG:…, Decode a Sheet-raster PNG — 8-bit grayscale, non-interlaced, any row filter —… (+13 more)

### Community 32 - "Workbench Overlay Tests"
Cohesion: 0.13
Nodes (18): Element, client(), _overlay(), fixture, Path, Review Workbench seam tests (ticket 10): the Flask test client against prepared…, Prepared run artifacts — the only data the Workbench reads., Reads run artifacts only: the Workbench module holds no reference into the… (+10 more)

### Community 33 - "Eval Harness Core"
Cohesion: 0.12
Nodes (21): _aggregate(), _component_name(), EvalSheet, Gate, _missed_sheet(), _pair(), _parse_config(), _predicted_links() (+13 more)

### Community 34 - "Plant Graph & Pipeline Config"
Cohesion: 0.14
Nodes (21): build_plant_graph(), The s2_pml equipment-level graph (ADR-0001): terminals become nodes, runs…, build_components(), PipelineConfig, Selects the implementation behind each component seam. The defaults are the…, flaky_config(), fixture, eval_root() (+13 more)

### Community 35 - "OCR Grammar Classification"
Cohesion: 0.15
Nodes (18): classify_candidate(), Which tag-grammar class a raw read belongs to — for a recognizer that reads…, _bar_sheet(), _bbox_of(), fixture_reads(), fixture, OCR engines behind the TextRecognizer seam (ticket 18). A real engine reads…, Reads a bar of ink only when it lies horizontally — a stand-in for an engine… (+10 more)

### Community 36 - "Label Factory Tests"
Cohesion: 0.16
Nodes (18): build_sheet_labels(), Project one validated artifact (plus the page's text spans, in PDF points) down…, artifact_dir_with(), Label factory (ticket 13): projecting hazop-ai vector artifacts into pixel-…, Two drawn runs between the same terminals are two connections — the…, On the real 2401 sheets an off-page connector drawn as an equipment-like glyph…, test_chain_through_pass_through_node_links_terminals_undirected(), test_direct_link_folds_nozzle_and_carries_direction_evidence() (+10 more)

### Community 37 - "Workbench KPI Tests"
Cohesion: 0.21
Nodes (19): _activity(), client(), clock(), _detections(), _give(), _kpi_value(), fixture, Path (+11 more)

### Community 38 - "Artifact Label Projection"
Cohesion: 0.16
Nodes (17): PageReader, _degraded_labels(), _discovered(), _examples(), _process_artifact(), Path, Transform, One artifact's ground truth, projected to pixels: detection-shaped label dicts… (+9 more)

### Community 39 - "OCR Engine Selection"
Cohesion: 0.15
Nodes (15): AppleVisionTextRecognizer, EasyOCRTextRecognizer, engine_options(), _EngineRecognizer, RapidOCRTextRecognizer, OCR engines behind the TextRecognizer seam (ticket 18). The engine choice was…, The raster turned `angle` degrees clockwise, with its new size., Nearest-neighbor integer upscaling: engines trained on text tens of pixels tall… (+7 more)

### Community 40 - "Piping Graph Emitter"
Cohesion: 0.30
Nodes (6): Run, _center(), Point, One shared PipingNode per branch point, however many runs meet there., _SheetEmitter, _xy()

### Community 41 - "OCR Backend Protocol"
Cohesion: 0.16
Nodes (9): Backend, _is_cjk(), Protocol, What an engine reports for one piece of text, in the coordinates of the raster…, One OCR engine. `name` and `version` identify it in provenance…, RawRead, EqualConfidenceBackend, _ink_bbox() (+1 more)

### Community 42 - "Label Store"
Cohesion: 0.22
Nodes (6): LabelStore, Path, Labeled examples on disk — <root>/<name>@<version>/sheet_<N>.json, one file per…, The store's partitions — quoted name@version directory keys., Sheet numbers the profile holds labeled examples for. The store owns its on-…, All of one Sheet's new examples in a single write — the label factory (ticket…

### Community 43 - "Detector Training"
Cohesion: 0.20
Nodes (11): _artifact_version(), _behavior_payload(), Path, The manifest fields that determine what the detector does — what the version…, Write-then-replace, like every other persisted artifact here., Train a symbol detector for one Convention Profile from label-factory datasets…, train_detector(), _write_json() (+3 more)

### Community 44 - "Render Projection Tests"
Cohesion: 0.29
Nodes (11): decode_gray_png(), ink_pixels(), Label factory rendering edge (ticket 13): real PyMuPDF rasterization of a…, A center-only symbol's box is synthesized from convention constants — it too…, The ticket-14 alignment criterion end to end: real rendered ink, skewed by the…, Decode the factory's own PNG form (8-bit gray, filter 0 rows)., test_degraded_variant_box_surrounds_the_skewed_ink(), test_projected_box_surrounds_the_rendered_symbol() (+3 more)

### Community 45 - "Degradation Geometry Mapping"
Cohesion: 0.24
Nodes (11): degrade_bbox(), degrade_polyline(), map_point(), Transform, The composed forward point map of the sequence — exactly the geometry the…, A label box under the composed map: the axis-aligned box enclosing the mapped…, test_degrade_bbox_encloses_the_mapped_corners(), test_degrade_polyline_maps_each_point() (+3 more)

### Community 46 - "Verdict Labels"
Cohesion: 0.25
Nodes (9): _check_bbox(), _check_number(), _check_polyline(), _check_string(), Reviewer verdicts as labeled examples (ticket 10): every pass / reject / edit…, The supervised label of one stored example — or a refusal naming the record:…, supervised_label(), Training-set export (ticket 12): reviewer verdicts leave the LabelStore as a… (+1 more)

### Community 47 - "Rotated Read Merging"
Cohesion: 0.20
Nodes (9): _iou(), Bbox, NamedTuple, A box in the turned frame mapped back onto the original raster of `width` x…, A raw read brought back into Sheet coordinates, with the sweep step that…, Whether the sweep step that produced the read stood its region upright: a tall…, One read per text region, with the reads it set aside: the most confident wins;…, _Read (+1 more)

### Community 48 - "Compression Transform"
Cohesion: 0.22
Nodes (8): Compression, JPEG-style quantization artifacts; severity is the IJG quality (1 crushes, 100…, The transform as a plain dict, echoed into run manifests so a dataset states…, transform_config(), gradient(), test_compression_handles_sizes_not_a_multiple_of_the_block(), test_compression_quality_scales_the_artifacts(), test_transform_configs_round_trip()

### Community 49 - "OCR Eval-Set Tests"
Cohesion: 0.22
Nodes (9): _is_cjk(), Synthetic OCR eval sets (ticket 18): tag Sheets rendered at the operating scale…, Mixed Chinese + Latin, rotated and vertical text: every set has Chinese-only…, Internal consistency: ground truth, raster and grammar agree, so the stub…, test_cli_writes_the_sets_and_reports_counts(), test_the_sets_are_reproducible_from_their_seed(), test_the_sets_load_as_harness_eval_sets_for_their_profile(), test_the_stub_pipeline_reads_every_tag_back() (+1 more)

### Community 50 - "Apple Vision Backend"
Cohesion: 0.32
Nodes (4): RuntimeError, AppleVisionBackend, Apple's Vision framework (VNRecognizeTextRequest, accurate level, zh-Hans + en-…, _Explodes

### Community 51 - "PDF Page Rendering"
Cohesion: 0.29
Nodes (7): PageRender, One span of the page's text layer, in PDF points — the lossless OCR ground…, One rendered PDF page: the raster (row-major 8-bit grayscale, the Sheet.raster…, The PyMuPDF-backed reader. Imported lazily so the factory's one rendering…, read_page(), TextSpan, test_a_tag_token_outside_its_bubble_does_not_class_as_instrument()

### Community 52 - "Review State"
Cohesion: 0.40
Nodes (6): Workbench Confidence Queues and Review Progress, Product KPI from Workbench Activity, One Sheet's review state from verdict coverage (ticket 11) — the one rule the…, review_state(), One Sheet's review state from verdict coverage. Only verdicts on detections…, _review_state()

### Community 53 - "EasyOCR Backend"
Cohesion: 0.40
Nodes (4): EasyOCRBackend, _quad_bbox(), _quad_note(), EasyOCR (CRAFT detector + CRNN recognizer, PyTorch) for ch_sim + en. Models are…

### Community 54 - "Pinned Gates"
Cohesion: 0.50
Nodes (4): apply_gates(), Pass/fail per pinned gate. A None score (nothing to score) fails: the absence…, test_an_unscorable_gate_fails_closed(), test_gates_state_pass_and_fail_per_gate()

### Community 55 - "Artifact Validation"
Cohesion: 0.50
Nodes (4): Refuse an artifact that is not one of hazop-ai's current topology_page…, validate_artifact(), test_artifact_missing_top_level_keys_is_refused_by_name(), test_stale_artifact_without_direction_stats_is_refused()

## Knowledge Gaps
- **16 isolated node(s):** `pidgraph`, `Gate`, `GraphStore Seam`, `Label Factory`, `Synthetic Degradation` (+11 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `LabelStore` connect `Label Store` to `Sheet Annotations & Fixtures`, `KPI & Activity Log`, `Symbol Detector Internals`, `Issue Tickets & Spec`, `Workbench Run Artifacts`, `Training-Set Export`, `Label Factory Synthesis`, `OCR Eval-Set Generation`, `Review Flow Tests`, `Degraded Dataset Variants`, `Eval Harness CLI`, `Trained Symbol Detector`, `KPI Reporting`, `Eval Harness Core`, `Plant Graph & Pipeline Config`, `Artifact Label Projection`, `Detector Training`, `Verdict Labels`, `Review State`?**
  _High betweenness centrality (0.087) - this node is a cross-community bridge._
- **Why does `Sheet` connect `Detection & Provenance` to `Sheet Annotations & Fixtures`, `Eval Harness Core`, `Raster Normalization`, `OCR Grammar Classification`, `PDF Intake & Refusal`, `Batch Run Reporting`, `Symbol Detector Internals`, `OCR Engine Selection`, `Sheet Assembly & Flow Evidence`, `Degradation Geometry Mapping`, `Legend NN Classifier`, `Line Network Extraction`, `Trained Symbol Detector`, `Batch Resume State`, `Recognizer Test Backends`?**
  _High betweenness centrality (0.061) - this node is a cross-community bridge._
- **Why does `digitize()` connect `Sheet Annotations & Fixtures` to `Workbench Overlay Tests`, `Cypher Graph Store`, `Plant Graph & Pipeline Config`, `KPI & Activity Log`, `Raster Normalization`, `Batch Run Reporting`, `PDF Intake & Refusal`, `Issue Tickets & Spec`, `Workbench KPI Tests`, `Convention Profile Loading`, `Detection & Provenance`, `Legend NN Classifier`, `Review Flow Tests`, `Trained Symbol Detector`, `Batch Resume State`, `Lexicon Tag Decoding`, `Recognizer Test Backends`?**
  _High betweenness centrality (0.045) - this node is a cross-community bridge._
- **Are the 31 inferred relationships involving `Sheet` (e.g. with `assemble_sheet()` and `SheetAssembly`) actually correct?**
  _`Sheet` has 31 INFERRED edges - model-reasoned connections that need verification._
- **Are the 56 inferred relationships involving `ValueError` (e.g. with `assemble_sheet()` and `_legend_entry()`) actually correct?**
  _`ValueError` has 56 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `digitize()` (e.g. with `ConventionProfile` and `Document`) actually correct?**
  _`digitize()` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 22 inferred relationships involving `LabelStore` (e.g. with `Batch Runs with Resume and Gap Reporting` and `_load_eval_sheet()`) actually correct?**
  _`LabelStore` has 22 INFERRED edges - model-reasoned connections that need verification._