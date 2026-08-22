"""OCR engines behind the TextRecognizer seam (ticket 18).

The engine choice was deliberately deferred and made empirically: every
candidate sits behind the same thin adapter, selected by configuration
("rapidocr", "tesseract", "apple-vision", "easyocr"), so the eval
harness scores them side by side on tag exact-match and the winner is
just the recommended selection. Pinned requirements all candidates must
meet: mixed Chinese + Latin recognition, rotated and vertical text, fully
local execution.

What the adapter does that no engine does: it assigns each raw read its
tag-grammar class (a real engine reads pixels without knowing what it is
looking at — lexicon.classify_candidate decides, and a read no grammar
fits is the reserved UNCLASSIFIED_TEXT, which the decoder fails closed),
stamps provenance naming the engine and its version, and hands the raw
string to the lexicon-constrained decoder unchanged. The decoder above
the seam is untouched by the choice.

Fully local is enforced, not hoped for: models and weights live outside
git (the engine package or a data directory) and are fetched, where an
engine needs fetching at all, by a deliberate one-time step; while an
engine is built or runs the adapter holds the offline guard, a tripwire
on Python-level connections — a model auto-download or a telemetry ping
raises OfflineViolation instead of quietly calling out. Native inference
code (ONNX Runtime, PyTorch, the tesseract binary) is outside Python's
socket layer; it is trusted to have no network path at inference, which
is what it is.
"""

from __future__ import annotations

import hashlib
import platform
import socket
from contextlib import contextmanager
from importlib import metadata
from pathlib import Path
from typing import Any, Iterator, NamedTuple, Protocol, Sequence

from .lexicon import classify_candidate
from .model import (
    UNCLASSIFIED_TEXT,
    Bbox,
    ConventionProfile,
    Provenance,
    Sheet,
    TextDetection,
)


class RawRead(NamedTuple):
    """What an engine reports for one piece of text, in the coordinates of
    the raster it was handed: an axis-aligned box (the hull of whatever
    quad the engine drew), the string as read, a confidence in 0..1, and
    the engine-specific evidence worth keeping (its quad, angle, ...)."""
    bbox: Bbox
    string: str
    confidence: float
    note: str = ""


class Backend(Protocol):
    """One OCR engine. `name` and `version` identify it in provenance
    (`text_recognizer:<name>@<version>`); `read` turns a grayscale raster
    (row-major, 0 = ink, 255 = paper) into raw reads."""
    name: str
    version: str
    rotations: tuple[int, ...]
    scale: int

    def read(self, width: int, height: int,
             pixels: bytes) -> list[RawRead]: ...


# --------------------------------------------------------------------------
# The offline guard: no component may call a network endpoint at inference
# time (spec). Holding the guard while an engine runs turns any attempt to
# open a connection — a model auto-download, a telemetry ping — into a
# loud refusal (the connect itself is refused; a name lookup that precedes
# it is not intercepted, so nothing is ever sent). Process-wide for its
# duration; the Raster Path runs one Sheet at a time and nothing else in
# it connects during extraction.

class OfflineViolation(RuntimeError):
    """An engine tried to reach the network while reading a Sheet."""


@contextmanager
def offline_guard(component: str) -> Iterator[None]:
    def refuse(*args, **kwargs):
        raise OfflineViolation(
            f"{component} tried to open a network connection during"
            " inference; the Raster Path runs fully local — install its"
            " models ahead of time instead")

    targets = ((socket.socket, "connect"), (socket.socket, "connect_ex"),
               (socket, "create_connection"))
    saved = [(owner, name, owner.__dict__.get(name))
             for owner, name in targets]
    for owner, name in targets:
        setattr(owner, name, refuse)
    try:
        yield
    finally:
        for owner, name, original in saved:
            if original is None:   # inherited (socket's C base class)
                delattr(owner, name)
            else:
                setattr(owner, name, original)


# --------------------------------------------------------------------------
# Raster geometry for the sweep: quarter-turn rotations and integer upscaling
# of a row-major grayscale raster, and the maps that bring an engine's boxes
# back into Sheet coordinates. Pure Python on bytes — fast enough at the
# operating scale, and the adapter stays testable without numpy.

ROTATIONS = (0, 90, 180, 270)   # clockwise quarter turns the sweep may use

# Two reads whose boxes overlap this much report the same text region; the
# sweep keeps the more confident one.
_SAME_REGION_IOU = 0.5


def _rotated(pixels: bytes, width: int, height: int,
             angle: int) -> tuple[bytes, int, int]:
    """The raster turned `angle` degrees clockwise, with its new size."""
    if angle == 0:
        return pixels, width, height
    if angle == 180:
        return pixels[::-1], width, height
    if angle == 90:
        # row v of the turned raster is column v of the original, bottom up
        return (b"".join(pixels[v::width][::-1] for v in range(width)),
                height, width)
    if angle == 270:
        return (b"".join(pixels[width - 1 - v::width] for v in range(width)),
                height, width)
    raise ValueError(f"rotation is one of {ROTATIONS}, got {angle!r}")


def _unrotated_bbox(bbox: Bbox, width: int, height: int,
                    angle: int) -> Bbox:
    """A box in the turned frame mapped back onto the original raster of
    `width` x `height` (continuous pixel-edge coordinates)."""
    u0, v0, u1, v1 = bbox
    if angle == 0:
        return bbox
    if angle == 90:
        return (v0, height - u1, v1, height - u0)
    if angle == 180:
        return (width - u1, height - v1, width - u0, height - v0)
    if angle == 270:
        return (width - v1, u0, width - v0, u1)
    raise ValueError(f"rotation is one of {ROTATIONS}, got {angle!r}")


def _upscaled(pixels: bytes, width: int, height: int,
              scale: int) -> bytes:
    """Nearest-neighbor integer upscaling: engines trained on text tens of
    pixels tall read the operating frame's small glyphs better grown."""
    if scale == 1:
        return pixels
    out = bytearray()
    wide = bytearray(width * scale)
    for y in range(height):
        row = pixels[y * width:(y + 1) * width]
        for k in range(scale):
            wide[k::scale] = row
        out += wide * scale
    return bytes(out)


def _iou(a: Bbox, b: Bbox) -> float:
    inter_w = min(a[2], b[2]) - max(a[0], b[0])
    inter_h = min(a[3], b[3]) - max(a[1], b[1])
    if inter_w <= 0 or inter_h <= 0:
        return 0.0
    inter = inter_w * inter_h
    union = ((a[2] - a[0]) * (a[3] - a[1])
             + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / union if union > 0 else 0.0


class _Read(NamedTuple):
    """A raw read brought back into Sheet coordinates, with the sweep
    step that produced it."""
    bbox: Bbox
    string: str
    confidence: float
    note: str
    angle: int


# --------------------------------------------------------------------------
# The adapter

class EngineTextRecognizer:
    """One engine behind the TextRecognizer seam.

    The engine reads the (normalized) Sheet raster, upscaled by `scale`
    when it wants larger glyphs and turned through each angle in
    `rotations` when it cannot read rotated text itself — an engine that
    handles rotated and vertical text natively declares rotations (0,)
    and is read once. Boxes come back into Sheet coordinates, duplicate
    reads of one region across the sweep collapse to the most confident,
    and every detection carries the raw string, its grammar class, and
    `text_recognizer:<engine>@<version>` provenance."""

    def __init__(self, backend: Backend,
                 rotations: "tuple[int, ...] | None" = None,
                 scale: "int | None" = None):
        self.backend = backend
        self.rotations = (backend.rotations if rotations is None
                          else tuple(rotations))
        self.scale = backend.scale if scale is None else scale
        for angle in self.rotations:
            if angle not in ROTATIONS:
                raise ValueError(
                    f"{backend.name}: rotations are quarter turns"
                    f" {ROTATIONS}, got {angle!r}")
        if not self.rotations:
            raise ValueError(f"{backend.name}: at least one rotation is"
                             " read (0 reads the Sheet upright)")
        if isinstance(self.scale, bool) or not isinstance(self.scale, int) \
                or self.scale < 1:
            raise ValueError(f"{backend.name}: scale is a positive integer"
                             f" upscaling factor, got {self.scale!r}")
        self.component = f"text_recognizer:{backend.name}@{backend.version}"

    def _sweep(self, sheet: Sheet, raster: bytes) -> list[_Read]:
        pixels = _upscaled(raster, sheet.width, sheet.height, self.scale)
        width, height = sheet.width * self.scale, sheet.height * self.scale
        reads: list[_Read] = []
        for angle in self.rotations:
            turned, turned_w, turned_h = _rotated(pixels, width, height,
                                                  angle)
            with offline_guard(self.component):
                raws = self.backend.read(turned_w, turned_h, turned)
            for raw in raws:
                x0, y0, x1, y1 = _unrotated_bbox(raw.bbox, width, height,
                                                 angle)
                reads.append(_Read(
                    (x0 / self.scale, y0 / self.scale,
                     x1 / self.scale, y1 / self.scale),
                    raw.string, raw.confidence, raw.note, angle))
        return reads

    @staticmethod
    def _turned_to_fit(read: _Read) -> bool:
        """Whether the sweep step that produced the read stood its region
        upright: a tall region is vertical text, read through a quarter
        turn; a wide one is horizontal, read with none."""
        x0, y0, x1, y1 = read.bbox
        tall = (y1 - y0) > (x1 - x0)
        return tall == (read.angle in (90, 270))

    def _merged(self, reads: list[_Read]) -> list[tuple[_Read, list[_Read]]]:
        """One read per text region, with the reads it set aside: the
        most confident wins; at equal confidence (engines with coarse
        scores tie across the sweep) the step that stood the region
        upright wins, then the earlier step, then position —
        deterministic for a deterministic engine. The set-aside reads
        ride along into the evidence, so the sweep hides nothing from
        review."""
        order = sorted(range(len(reads)),
                       key=lambda i: (-reads[i].confidence,
                                      not self._turned_to_fit(reads[i]),
                                      self.rotations.index(reads[i].angle),
                                      reads[i].bbox, reads[i].string))
        kept: list[tuple[_Read, list[_Read]]] = []
        for i in order:
            read = reads[i]
            for winner, others in kept:
                if _iou(read.bbox, winner.bbox) >= _SAME_REGION_IOU:
                    others.append(read)
                    break
            else:
                kept.append((read, []))
        return sorted(kept, key=lambda k: (k[0].bbox[1], k[0].bbox[0],
                                           k[0].string))

    def recognize(self, sheet: Sheet,
                  profile: ConventionProfile) -> list[TextDetection]:
        if sheet.raster is None:
            raise ValueError(
                f"{self.component} reads pixels; Sheet {sheet.number}"
                " carries no raster")
        detections = []
        merged = self._merged(self._sweep(sheet, sheet.raster))
        for i, (read, others) in enumerate(merged):
            text_class = classify_candidate(read.string, profile.tag_grammar)
            evidence = (f"{self.backend.name} read {read.string!r} at"
                        f" confidence {read.confidence:.2f}")
            if read.note:
                evidence += f" ({read.note})"
            if read.angle:
                evidence += f", Sheet rotated {read.angle}°"
            if self.scale != 1:
                evidence += f", upscaled {self.scale}x"
            if others:
                evidence += "; also read as " + ", ".join(
                    f"{o.string!r} at {o.confidence:.2f}"
                    + (f" rotated {o.angle}°" if o.angle else "")
                    for o in others)
            if text_class is None:
                text_class = UNCLASSIFIED_TEXT
                evidence += "; no tag-grammar class fits — unclassified"
            else:
                evidence += f"; classified {text_class} by grammar"
            detections.append(TextDetection(
                id=f"p{sheet.number}-text{i}",
                sheet=sheet.number,
                string=read.string,
                text_class=text_class,
                bbox=read.bbox,
                confidence=read.confidence,
                provenance=Provenance(component=self.component,
                                      evidence=evidence),
            ))
        return detections


# --------------------------------------------------------------------------
# Selection options: "<engine>:scale=<n>,rotations=<a>/<b>" — the sweep
# and the upscaling are the only knobs the pipeline configures; anything
# engine-internal stays at the engine's defaults so a selection string
# names a reproducible configuration.

def engine_options(options: str) -> dict:
    """Parse the ':<options>' part of an engine selection into
    EngineTextRecognizer keyword arguments. Refuses unknown keys,
    repeated keys, and malformed values by name."""
    parsed: dict = {}
    for part in options.split(",") if options else []:
        key, separator, value = part.partition("=")
        if key not in ("scale", "rotations") or not separator or not value:
            raise ValueError(
                f"engine option {part!r} is not scale=<positive integer>"
                f" or rotations=<angles joined by '/'>, e.g."
                f" 'rotations=0/90/270'")
        if key in parsed:
            raise ValueError(f"engine option {key!r} is given twice")
        if key == "scale":
            if not value.isdigit() or int(value) < 1:
                raise ValueError(
                    f"scale is a positive integer upscaling factor,"
                    f" got {value!r}")
            parsed[key] = int(value)
        else:
            angles = value.split("/")
            if not all(angle.isdigit() and int(angle) in ROTATIONS
                       for angle in angles):
                raise ValueError(
                    f"rotations are quarter turns from {ROTATIONS} joined"
                    f" by '/', got {value!r}")
            parsed[key] = tuple(int(angle) for angle in angles)
    return parsed


# --------------------------------------------------------------------------
# The candidate engines. Each backend imports its package lazily — the
# core package and the test suite never need any of them — and is built
# under the offline guard, so an engine that would download its models on
# first use fails with the install hint instead of calling out.

def _missing(engine: str, extra: str, what: str,
             error: ModuleNotFoundError) -> ModuleNotFoundError:
    return ModuleNotFoundError(
        f"the {engine} TextRecognizer needs {what}; install the {extra}"
        f" extra (pip install 'pidgraph[{extra}]') — {error}")


def _content_version(paths: Sequence[Path]) -> str:
    """Twelve hex digits of the sorted model files' names and bytes: the
    part of an engine's version that is its weights, so provenance tells
    two model sets apart the way the trained detector's does."""
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def _quad_bbox(points) -> Bbox:
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


def _quad_note(points) -> str:
    return "quad " + str([[round(float(p[0])), round(float(p[1]))]
                          for p in points])


class RapidOCRBackend:
    """PP-OCR (PaddleOCR's detector, angle classifier and recognizer)
    on ONNX Runtime via RapidOCR: the Chinese + Latin models ship inside
    the package, the detector draws rotated quads the recognizer reads,
    and the classifier fixes 180° flips. It reads vertical text on its
    own, but at the operating scale it reads it better upright: the
    comparison (ticket 18) put tag exact-match at 0.576 read once and
    0.694 with the 0/90/270 sweep, so the sweep is its default."""

    name = "rapidocr"
    rotations = (0, 90, 270)
    scale = 1

    def __init__(self) -> None:
        try:
            import numpy  # type: ignore[import-not-found]
            import onnxruntime  # type: ignore[import-not-found,import-untyped]
            import rapidocr  # type: ignore[import-not-found,import-untyped]
            from rapidocr import RapidOCR  # type: ignore[import-not-found,import-untyped]
        except ModuleNotFoundError as error:
            raise _missing(self.name, "ocr",
                           "RapidOCR and ONNX Runtime", error) from error
        self._numpy = numpy
        with offline_guard(f"text_recognizer:{self.name}"):
            self._engine = RapidOCR(params={"Global.log_level": "warning"})
        model_paths = self._model_files(rapidocr)
        self.models = ", ".join(p.name for p in model_paths)
        package = metadata.version("rapidocr")
        runtime = getattr(onnxruntime, "__version__", "?")
        self.version = (f"{package}+onnxruntime{runtime}"
                        f"+{_content_version(model_paths)}")

    def _model_files(self, rapidocr) -> list[Path]:
        """The weights the engine loaded, for provenance: the ONNX session
        behind each stage names its file; failing that, the models the
        package ships. Refused when neither names a file — provenance
        that cannot name the weights would fail open."""
        paths = []
        for stage in ("text_det", "text_cls", "text_rec"):
            session = getattr(getattr(self._engine, stage, None),
                              "session", None)
            model_path = getattr(getattr(session, "session", session),
                                 "_model_path", None)
            if model_path and Path(model_path).is_file():
                paths.append(Path(model_path))
        if not paths:
            bundled = Path(rapidocr.__file__).parent / "models"
            paths = sorted(bundled.glob("*.onnx"))
        if not paths:
            raise ValueError(
                "cannot identify RapidOCR's model files (no ONNX session"
                " names one and the package bundles none) — refusing to"
                " run an engine whose weights provenance cannot name")
        return paths

    def read(self, width: int, height: int,
             pixels: bytes) -> list[RawRead]:
        image = self._numpy.frombuffer(pixels, dtype=self._numpy.uint8)
        result: Any = self._engine(image.reshape(height, width))
        if result.boxes is None:
            return []
        return [RawRead(_quad_bbox(box), str(text), float(score),
                        _quad_note(box))
                for box, text, score in zip(result.boxes, result.txts,
                                            result.scores)]


TESSDATA_DIR = Path("data") / "ocr" / "tessdata"   # outside git, like data/
TESSERACT_LANGUAGES = ("chi_sim", "eng")


def _is_cjk(char: str) -> bool:
    return "一" <= char <= "鿿"


class TesseractBackend:
    """Tesseract 5 (LSTM) through pytesseract, chi_sim + eng, sparse-text
    page segmentation. Tesseract reads upright text only, so the adapter
    sweeps 0/90/270, and it wants glyphs tens of pixels tall, so the
    operating frame is upscaled 3x. Its words come back one per CJK
    character: words on one line are joined, CJK neighbours without a
    space. Language data lives outside git in data/ocr/tessdata."""

    name = "tesseract"
    rotations = (0, 90, 270)
    scale = 3

    def __init__(self) -> None:
        try:
            import pytesseract  # type: ignore[import-not-found,import-untyped]
            from PIL import Image  # type: ignore[import-not-found]
        except ModuleNotFoundError as error:
            raise _missing(self.name, "ocr-candidates",
                           "pytesseract and Pillow (plus the tesseract"
                           " binary)", error) from error
        self._pytesseract = pytesseract
        self._image = Image
        self.tessdata_dir = TESSDATA_DIR
        data_files = [self.tessdata_dir / f"{lang}.traineddata"
                      for lang in TESSERACT_LANGUAGES]
        missing = [p for p in data_files if not p.is_file()]
        if missing:
            raise ValueError(
                f"tesseract language data missing: {missing} — download"
                f" {[p.name for p in missing]} from the tesseract-ocr/"
                f"tessdata repository into {self.tessdata_dir} (outside"
                " git) before selecting the tesseract TextRecognizer")
        try:
            binary = str(pytesseract.get_tesseract_version())
        except pytesseract.TesseractNotFoundError as error:
            raise ValueError(
                "the tesseract binary is not installed or not on PATH"
                " (brew install tesseract)") from error
        self.languages = "+".join(TESSERACT_LANGUAGES)
        self.models = ", ".join(p.name for p in data_files)
        self.version = (f"{binary}+{self.languages}"
                        f"+{_content_version(data_files)}")
        self._config = f'--tessdata-dir "{self.tessdata_dir}" --psm 11'

    def read(self, width: int, height: int,
             pixels: bytes) -> list[RawRead]:
        image = self._image.frombytes("L", (width, height), pixels)
        data = self._pytesseract.image_to_data(
            image, lang=self.languages, config=self._config,
            output_type=self._pytesseract.Output.DICT)
        lines: dict[tuple, list] = {}
        for i, word in enumerate(data["text"]):
            if not str(word).strip():
                continue
            confidence = float(data["conf"][i])
            if confidence < 0:
                continue
            key = (data["block_num"][i], data["par_num"][i],
                   data["line_num"][i])
            lines.setdefault(key, []).append((
                data["left"][i], data["top"][i], data["width"][i],
                data["height"][i], str(word), confidence))
        reads = []
        for words in lines.values():
            words.sort(key=lambda w: w[0])
            string = ""
            for _, _, _, _, word, _ in words:
                if string and not (_is_cjk(string[-1]) and _is_cjk(word[0])):
                    string += " "
                string += word
            x0 = min(w[0] for w in words)
            y0 = min(w[1] for w in words)
            x1 = max(w[0] + w[2] for w in words)
            y1 = max(w[1] + w[3] for w in words)
            confidence = sum(w[5] for w in words) / len(words) / 100.0
            reads.append(RawRead((float(x0), float(y0), float(x1), float(y1)),
                                 string, confidence,
                                 f"{len(words)} word(s), psm 11"))
        return reads


class AppleVisionBackend:
    """Apple's Vision framework (VNRecognizeTextRequest, accurate level,
    zh-Hans + en-US, language correction off so tags are not 'corrected'
    into words) through PyObjC — ships with macOS, so fully local with
    nothing to download, and macOS-only. It reads some rotations itself;
    the adapter sweeps 0/90/270 and keeps the best read per region."""

    name = "apple-vision"
    rotations = (0, 90, 270)
    scale = 1

    def __init__(self) -> None:
        try:
            import Quartz  # type: ignore[import-not-found,import-untyped]
            import Vision  # type: ignore[import-not-found,import-untyped]
        except ModuleNotFoundError as error:
            raise _missing(self.name, "ocr-candidates",
                           "PyObjC's Vision and Quartz frameworks (macOS)",
                           error) from error
        self._quartz = Quartz
        self._vision = Vision
        self.languages = ["zh-Hans", "en-US"]
        request = self._request()
        self.models = f"VNRecognizeTextRequest revision {request.revision()}"
        self.version = (f"macOS{platform.mac_ver()[0]}"
                        f"+rev{request.revision()}")

    def _request(self):
        vision = self._vision
        request = vision.VNRecognizeTextRequest.alloc().init()
        request.setRecognitionLevel_(
            vision.VNRequestTextRecognitionLevelAccurate)
        request.setUsesLanguageCorrection_(False)
        request.setRecognitionLanguages_(self.languages)
        return request

    def read(self, width: int, height: int,
             pixels: bytes) -> list[RawRead]:
        quartz = self._quartz
        provider = quartz.CGDataProviderCreateWithData(None, pixels,
                                                       len(pixels), None)
        image = quartz.CGImageCreate(
            width, height, 8, 8, width, quartz.CGColorSpaceCreateDeviceGray(),
            quartz.kCGImageAlphaNone, provider, None, False,
            quartz.kCGRenderingIntentDefault)
        request = self._request()
        handler = self._vision.VNImageRequestHandler.alloc() \
            .initWithCGImage_options_(image, None)
        ok, error = handler.performRequests_error_([request], None)
        if not ok:
            raise RuntimeError(f"Vision text recognition failed: {error}")
        reads = []
        for observation in request.results() or []:
            candidates = observation.topCandidates_(1)
            if not candidates:
                continue
            candidate = candidates[0]
            box = observation.boundingBox()   # normalized, origin bottom-left
            x0 = box.origin.x * width
            x1 = x0 + box.size.width * width
            y1 = (1.0 - box.origin.y) * height
            y0 = y1 - box.size.height * height
            reads.append(RawRead((x0, y0, x1, y1), str(candidate.string()),
                                 float(candidate.confidence()),
                                 "top candidate"))
        return reads


class EasyOCRBackend:
    """EasyOCR (CRAFT detector + CRNN recognizer, PyTorch) for ch_sim + en.
    Models are fetched once into ~/.EasyOCR by a deliberate online step;
    here the reader is built with downloading disabled under the offline
    guard. Rotated text is handled by EasyOCR's own rotation trials
    (rotation_info), so the adapter reads once."""

    name = "easyocr"
    rotations = (0,)
    scale = 1
    languages = ("ch_sim", "en")

    def __init__(self) -> None:
        try:
            import easyocr  # type: ignore[import-not-found,import-untyped]
            import numpy  # type: ignore[import-not-found]
        except ModuleNotFoundError as error:
            raise _missing(self.name, "ocr-candidates",
                           "EasyOCR (with PyTorch)", error) from error
        self._numpy = numpy
        try:
            with offline_guard(f"text_recognizer:{self.name}"):
                self._reader = easyocr.Reader(
                    list(self.languages), gpu=False, verbose=False,
                    download_enabled=False)
        except OSError as error:   # missing model files
            raise ValueError(
                "EasyOCR's ch_sim + en models are not installed; fetch"
                " them once, online, with python -c \"import easyocr;"
                " easyocr.Reader(['ch_sim', 'en'])\" — inference itself"
                f" stays offline ({error})") from error
        model_dir = Path(self._reader.model_storage_directory)
        model_paths = sorted(model_dir.glob("*.pth"))
        self.models = ", ".join(p.name for p in model_paths) or "unknown"
        self.version = (f"{easyocr.__version__}+{'+'.join(self.languages)}"
                        f"+{_content_version(model_paths)}")

    def read(self, width: int, height: int,
             pixels: bytes) -> list[RawRead]:
        image = self._numpy.frombuffer(pixels, dtype=self._numpy.uint8)
        results = self._reader.readtext(image.reshape(height, width),
                                        rotation_info=[90, 180, 270])
        return [RawRead(_quad_bbox(box), str(text), float(confidence),
                        _quad_note(box))
                for box, text, confidence in results]


# --------------------------------------------------------------------------
# The seam registry entries: one recognizer class per engine, selected as
# "<engine>" or "<engine>:scale=<n>,rotations=<a>/<b>"

class _EngineRecognizer(EngineTextRecognizer):
    BACKEND: type[Any]

    def __init__(self, rotations: "tuple[int, ...] | None" = None,
                 scale: "int | None" = None):
        # built under the guard too: an engine that would fetch its
        # models on first use fails here with the install hint
        with offline_guard(f"text_recognizer:{self.BACKEND.name}"):
            backend = self.BACKEND()
        super().__init__(backend, rotations=rotations, scale=scale)

    @classmethod
    def from_options(cls, options: str) -> "_EngineRecognizer":
        return cls(**engine_options(options))


class RapidOCRTextRecognizer(_EngineRecognizer):
    BACKEND = RapidOCRBackend


class TesseractTextRecognizer(_EngineRecognizer):
    BACKEND = TesseractBackend


class AppleVisionTextRecognizer(_EngineRecognizer):
    BACKEND = AppleVisionBackend


class EasyOCRTextRecognizer(_EngineRecognizer):
    BACKEND = EasyOCRBackend


ENGINES: dict[str, type] = {
    "rapidocr": RapidOCRTextRecognizer,
    "tesseract": TesseractTextRecognizer,
    "apple-vision": AppleVisionTextRecognizer,
    "easyocr": EasyOCRTextRecognizer,
}
