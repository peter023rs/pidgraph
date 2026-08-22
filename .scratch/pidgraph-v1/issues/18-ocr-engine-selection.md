# 18 — OCR engine selection and integration behind the TextRecognizer seam

**What to build:** The deliberately deferred OCR engine choice, made empirically: candidate engines are scored by the eval harness on tag exact-match, and the winner is integrated behind the TextRecognizer seam, selected by configuration. Pinned requirements: mixed Chinese + Latin recognition, rotated and vertical text, fully local execution. The lexicon-constrained decoder sits above the seam and is untouched by the choice.

**Blocked by:** 15 (Eval harness).

**Status:** ready-for-human

- [x] Candidate engines meeting the pinned requirements (mixed Chinese + Latin, rotated and vertical text, fully local) are evaluated via the harness on labeled eval sets; the comparison report records the evidence for the choice. *(Four candidates on synthetic operating-scale sets; RapidOCR wins — ADR-0002. The real-corpus tag gate stays blocked on the normalization-scale revisit, as for ticket 16 — see the comment.)*
- [x] The chosen engine runs behind the TextRecognizer seam, selected by configuration; its raw candidates feed the lexicon-constrained decoder unchanged.
- [x] Recognized text carries confidence and provenance identifying the engine and version.
- [x] Inference is fully local — no network endpoint is called at inference time; models/weights live outside git.
- [x] The test suite is unaffected: the stub remains the test-time default, and CI still needs no models or network.

## Comments

2026-08-21 — Implemented in `src/pidgraph/ocr.py` (the adapter and the
four engine backends, registered in `seams.TEXT_RECOGNIZERS` as
`rapidocr`, `tesseract`, `apple-vision`, `easyocr`, each selectable with
`:scale=<n>,rotations=<a>/<b>`), `src/pidgraph/ocr_evalset.py` (the
synthetic OCR eval-set renderer, `python -m pidgraph.ocr_evalset`), and
the decision record `docs/adr/0002-ocr-engine-rapidocr.md`. The choice:
**RapidOCR** (PP-OCRv6 det/rec + v4 cls on ONNX Runtime; models ship in
the wheel), `PipelineConfig(text_recognizer="rapidocr")`, installed by
the new `ocr` extra; the other candidates stay behind `ocr-candidates`
so the comparison reruns as a command.

What the adapter does that no engine does: a real engine reads pixels
without knowing the tag class, so `lexicon.classify_candidate` (pure,
above-seam logic shared with the decoder's confusion set) assigns each
raw read its grammar class — exactly one exact-match class, else exactly
one confusion-set-reachable class, else the new reserved
`UNCLASSIFIED_TEXT` (`"unclassified"`), which has no grammar by
construction, so the decoder fails it closed and `load_profile` refuses
a tag grammar that defines it (mirror of ticket 17's symbol class). The
raw string goes to the decoder unchanged. Provenance is
`text_recognizer:<engine>@<version>` where the version binds the engine
package/runtime and a content hash of its weights (e.g.
`rapidocr@3.9.2+onnxruntime1.29.0+3358ac6b2a09`). Fully local is
enforced: engines are built and run under an offline guard — a tripwire
on Python-level connections that turns a would-be model download into
`OfflineViolation` (native inference code is outside Python's socket
layer and trusted by inspection); models live in the wheel (RapidOCR),
in `data/ocr/tessdata` (Tesseract), `~/.EasyOCR` (EasyOCR) or the OS
(Vision) — never in git — and the RapidOCR adapter refuses to run if it
cannot name its weight files. The adapter also sweeps quarter-turn
rotations (pure-Python raster turns, boxes mapped back, one read per
region with the most confident winning, ties to the turn that stood a
tall region upright, the set-aside reads kept in the evidence) and
upscales for engines that want larger glyphs. "Vertical" is covered as
rotated lines of text; stacked CJK columns are neither generated nor
swept.

Evidence (harness v1, tag exact-match; sets and reports under
`data/eval/ticket18-ocr/`, outside git): clean set, 12 Sheets / 144
tags — rapidocr with the 0/90/270 sweep **0.694** (read once 0.576;
2× upscale adds nothing), apple-vision 0.479, tesseract 0.097, easyocr
0.042; stub 1.000 as the consistency bar. RapidOCR leads every class and
reads Chinese labels 21/24 against Vision's 9/24, at ≈0.2 s per read of
a 400 px Sheet on CPU (Vision is faster still, but macOS-only and
weaker). Its remaining misses are
mostly conventions the decoder could learn, not misreads: 18 mixed
labels read without the space between scripts (`凝结水CD`), 7 bubble
arcs read as brackets (`(LIC-978)`), 4 middle-dots for hyphens, one
I→l. Degraded set: every engine ≈ 0 (rapidocr 2/144) — and the probe
shows why: the recognizer gets the Otsu-binarized normalized frame,
which turns the noisy raster into speckle; the same rapidocr reads
33/144 from the deskewed grayscale and 42/144 from the raw grayscale.
The harness CLI now keeps commas inside an implementation's options and
reports the component identities and seconds per configuration. The
breakdowns behind these numbers are kept beside the reports
(`analysis-clean-misses.txt`, `probe-binarization.txt`,
`smoke-real-sheets.txt`).

The human part (`ready-for-human`, as with 16/17): the real-corpus gate.
On a real label-factory Sheet normalized to 400 px every engine reads
0/51 tags (40 px text becomes ~3 px); at native resolution they start
to (Sheet 5: rapidocr 7/78 exact, 36 % of tag strings read). Three
decisions follow from the numbers, none of them engine changes —
(1) revisit `TARGET_LONG_SIDE` with the corpus (ticket 16's open item);
(2) hand the TextRecognizer the deskewed grayscale frame instead of the
binarized one (normalization, ticket 03 territory; worth its own
ticket); (3) teach the decoder the conventions above — enclosing
brackets, `·`/`-` and I/l confusions, whitespace between scripts
(ticket 06's confusion set is letter/digit pairs only). With (3) alone
the clean-set score would be ≈ 0.90. Also noted: under the hazop
profile's permissive `instrument_tag` grammar (`[A-Z]{1,4}|\d{3,6}`)
garbage one-letter reads resolve — ticket 06's documented permissive-
grammar limitation, now visible with a real engine. PaddleOCR proper
has no `paddlepaddle` wheel for this repo's Python 3.14; its models are
what `rapidocr` runs.

Tests (offline, no engine, no models, no network; 351 green, mypy
clean; also green in a fresh Python 3.12 venv with only the dev
extras): `tests/test_ocr_recognizer.py` — class assignment as a pure
function, the adapter with fake backends (provenance, class, raw
strings through the decoder into DEXPI via `digitize()`, the rotation
sweep and back-mapping, upscaling, tie-breaks, the offline guard, the
registry and install hints, options, the harness scoring an engine
beside the stub, the CLI config grammar), `tests/test_ocr_evalset.py`
(the sets load, cover the pinned requirements, and the stub reads every
tag back), plus the reserved text class in `tests/test_profile.py`.
