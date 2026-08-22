# Select RapidOCR (PP-OCR on ONNX Runtime) as the OCR engine behind the TextRecognizer seam

The OCR engine choice was deliberately deferred behind the TextRecognizer
seam (spec) and pinned to three requirements: mixed Chinese + Latin
recognition, rotated and vertical text, fully local execution. Ticket 18
made the choice empirically with the eval harness (ticket 15). Every
candidate sits behind the same adapter (`pidgraph.ocr`), selected by
configuration, so the comparison is a set of `--config` strings and the
winner is a default, not an architecture.

## Candidates

| Selection      | Engine                                                        | Local models                         |
| -------------- | ------------------------------------------------------------- | ------------------------------------ |
| `rapidocr`     | PP-OCRv6 det/rec + v4 angle cls (PaddleOCR's models) on ONNX Runtime via RapidOCR 3.9.2 | ship inside the wheel |
| `apple-vision` | macOS Vision `VNRecognizeTextRequest` rev 3, zh-Hans + en-US, language correction off (PyObjC) | ship with macOS; macOS-only |
| `tesseract`    | Tesseract 5.5.3 LSTM, chi_sim + eng, sparse-text psm, 3× upscale | traineddata in `data/ocr/tessdata` |
| `easyocr`      | EasyOCR 1.7.2 (CRAFT + CRNN, PyTorch), ch_sim + en             | one online fetch into `~/.EasyOCR` |

PaddleOCR proper was not a candidate in this repo's Python (3.14): it has
no `paddlepaddle` wheel; its model family is what `rapidocr` runs. No
cloud or VLM engine was considered (ADR-0001).

## Evidence (harness v1, tag exact-match, 12 Sheets / 144 tags)

Synthetic OCR eval sets at the operating scale (`python -m
pidgraph.ocr_evalset`, seed 0): Latin tags, Chinese equipment labels,
mixed Chinese + Latin service labels, 40 % standing vertically or
rotated, among bubbles, pipes and outlines; a clean set and a degraded
one (skew 0.8°, blur r=1, noise σ=16). The stub scores 1.000 on both —
the sets are internally consistent. Reports:
`data/eval/ticket18-ocr/report-*.json` (outside git, like every run
artifact).

| Configuration                               | clean     | degraded |
| ------------------------------------------- | --------- | -------- |
| `rapidocr:rotations=0/90/270` (the default) | **0.694** | 0.014    |
| `rapidocr:rotations=0` (read once)          | 0.576     | 0.007    |
| `rapidocr:scale=2,rotations=0/90/270`       | 0.681     | 0.000    |
| `apple-vision` (0/90/270 sweep)             | 0.479     | 0.000    |
| `apple-vision:rotations=0`                  | 0.444     | 0.000    |
| `tesseract` (3×, 0/90/270 sweep)            | 0.097     | 0.000    |
| `easyocr`                                   | 0.042     | 0.000    |

RapidOCR leads on every class — Chinese labels 21/24 against Vision's
9/24 — at ≈0.2 s per read of a 400 px Sheet on CPU (Vision is faster
still, but macOS-only and weaker). Its remaining clean-set misses are
mostly conventions, not reading: 18 mixed labels read without the space
between the Chinese and Latin halves (`凝结水CD`), 7 instrument bubbles
whose arcs read as brackets (`(LIC-978)`), 4 middle-dots for hyphens
(`LT·161`), one `I`→`l`; the rest are small-glyph digit or CJK
confusions at 9–10 pt. The breakdowns, the binarization probe and the
real-Sheet smoke are kept beside the reports (`analysis-clean-
misses.txt`, `probe-binarization.txt`, `smoke-real-sheets.txt`).

Two findings sit above the seam and bound what any engine can score:

- **Binarization, not the engine, zeroes the degraded set.** The
  recognizer is handed the normalized frame, which Otsu-binarizes the
  noisy raster into speckle; the same swept rapidocr reads 2/144 of
  the degraded tags from that frame, 33/144 from the deskewed grayscale
  and 42/144 from the raw grayscale. OCR wants the grayscale frame.
- **The operating scale hides real tags.** On a real label-factory Sheet
  (hazop-ai 2401, Sheet 4, 4967 px → 400 px) every engine reads 0/51
  tags; at native resolution they start to (tesseract 7, rapidocr 2;
  Sheet 5: rapidocr 7/78 exact, 36 % of tag strings read somewhere).
  This is the same `TARGET_LONG_SIDE` revisit ticket 16 left to the
  corpus — the real-corpus tag gate (≥ 0.98) is not claimable before it.

## Decision

`rapidocr` is the selected engine: `PipelineConfig(text_recognizer=
"rapidocr")`, installed by the `ocr` extra, with the adapter's 0/90/270
rotation sweep as its default. The other candidates stay selectable
(the `ocr-candidates` extra) so the comparison reruns as a command.

## Consequences

- Inference is fully local and enforced: the models live in the wheel,
  and the adapter holds an offline guard while any engine is built or
  runs — a tripwire on Python-level connections (a would-be model
  download raises); native inference code is trusted by inspection to
  have no network path. Provenance names the engine and its exact
  weights — `text_recognizer:rapidocr@3.9.2+onnxruntime1.29.0+<models
  hash>` — and refuses to run an engine whose weights it cannot name.
- The adapter owns what no engine does: assigning each raw read its
  tag-grammar class (`lexicon.classify_candidate`; a read no grammar
  fits is the reserved `unclassified` text class, which the decoder
  fails closed), sweeping quarter-turn rotations and upscaling for
  engines that need it (one read per region reaches the decoder, the
  string unchanged; the reads set aside by the sweep stay in the
  evidence), and handing raw strings to the lexicon decoder unchanged.
  "Vertical" here is rotated lines of text; stacked CJK columns are not
  generated or swept. Swapping engines later is a selection string.
- Follow-ups the evidence points at, none of them engine changes:
  hand the TextRecognizer the deskewed grayscale frame (normalization,
  ticket 03); revisit the operating scale with the corpus (ticket 16's
  open item); teach the decoder the conventions above — enclosing
  brackets from bubble arcs, middle-dot/hyphen and I/l confusions,
  whitespace between scripts (ticket 06's confusion set is letter/digit
  pairs only).
- Re-run the comparison when any of those land: `python -m
  pidgraph.ocr_evalset data/eval/<dir>` then `python -m
  pidgraph.eval_harness data/eval/<dir>/clean data/eval/<dir>/profile
  --config ...` with one `--config` per candidate.
