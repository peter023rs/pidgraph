# 12 — Training-set export from reviewer verdicts

**What to build:** An ML developer exports reviewer verdicts as a supervised training set per Convention Profile, so per-company fine-tuning has data drawn from exactly the deployed corpus. Pass verdicts become positive examples, rejects become negatives, and edits carry the human-corrected label.

**Blocked by:** 10 (Review Workbench: overlay and verdicts).

**Status:** resolved

- [x] An export produces a training set filtered to one Convention Profile (identity + version), from the stored labeled examples.
- [x] The export represents all three verdict kinds usefully: pass (confirmed detection), reject (negative), edit (detection with corrected tag text or geometry as the label).
- [x] Examples reference Sheets by identifier and carry the geometry/text needed for training; exports containing drawing-derived content live outside git per ADR-0001.
- [x] The export is deterministic over a given verdict store — re-export yields the same set.
- [x] Tests export from a prepared verdict store fixture and assert content and filtering; offline test invariant holds.

## Comments

Implemented as `src/pidgraph/training.py` —
`export_training_set(labels_root, profile, out_path, source=None)` plus
`python -m pidgraph.training <labels_root> --name N --version V
[--source S] [--out P]` (default out: a `training/` directory beside the
labels, outside git with the rest of the run artifacts, ADR-0001). The
store's layout enumeration moved into `labels.py` as
`LabelStore.profiles()` / `LabelStore.sheets(profile)` so only the store
owns its on-disk shape.

One JSONL line per labeled example, sorted by Sheet then detection id —
byte-deterministic over a given store, independent of the order
reviewing happened in. Each line: profile, source, sheet (the store
partition's number — the snapshot's internal field is provenance only),
kind, verdict, its role (`positive` / `negative` / `corrected`), the
detection snapshot untouched, and `label` — the ground-truth fields per
kind, which a pass confirms as recorded, an edit overrides exactly where
corrected (a string-only text edit keeps the recorded bbox), and a
reject nulls (the detection itself is the negative). `source` tags the
run/Document, since the labels store is per-run and per-profile sets
from different Documents will be merged for fine-tuning.

Refusals are named, never silent: a profile the store holds no examples
for (typo'd identity/version, or a directory an interrupted first save
left empty) lists the store's actual partitions; a store file recorded
for a different identity than requested — reachable via case-insensitive
filesystems resolving another partition's directory — is refused rather
than re-stamped; store-legal-but-malformed records (unknown kind,
detection lacking a label field) are refused naming the Sheet and
detection. The set is written whole via write-then-replace, mirroring
the LabelStore, so an interrupted export never leaves a
truncated-but-parseable training set. Tests
(`tests/test_training_export.py`) prepare verdict stores directly
through the LabelStore — offline throughout — including Chinese profile
names and tag strings per the corpus ADR-0001 describes.
