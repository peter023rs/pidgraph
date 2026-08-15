# 18 — OCR engine selection and integration behind the TextRecognizer seam

**What to build:** The deliberately deferred OCR engine choice, made empirically: candidate engines are scored by the eval harness on tag exact-match, and the winner is integrated behind the TextRecognizer seam, selected by configuration. Pinned requirements: mixed Chinese + Latin recognition, rotated and vertical text, fully local execution. The lexicon-constrained decoder sits above the seam and is untouched by the choice.

**Blocked by:** 15 (Eval harness).

**Status:** ready-for-human

- [ ] Candidate engines meeting the pinned requirements (mixed Chinese + Latin, rotated and vertical text, fully local) are evaluated via the harness on labeled eval sets; the comparison report records the evidence for the choice.
- [ ] The chosen engine runs behind the TextRecognizer seam, selected by configuration; its raw candidates feed the lexicon-constrained decoder unchanged.
- [ ] Recognized text carries confidence and provenance identifying the engine and version.
- [ ] Inference is fully local — no network endpoint is called at inference time; models/weights live outside git.
- [ ] The test suite is unaffected: the stub remains the test-time default, and CI still needs no models or network.

## Comments
