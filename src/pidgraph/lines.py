"""Line network extraction — classical CV, deterministic, no seam and no ML
(spec decision).

The stage runs binarize -> thin -> vectorize -> junction analysis -> split
runs at symbol boxes, on the normalized Sheet raster:

- Split at symbol boxes: symbol ink is masked out before tracing, so a line
  drawn through a valve becomes two runs terminating at that valve —
  connectivity never tunnels through a symbol. Flow arrows are the
  exception: they sit ON a run rather than interrupting it, so the run is
  re-bridged straight across the arrow's box.
- Junction analysis: skeleton pixels with three or more branches collapse
  into one junction point shared as an endpoint by every incident run — a
  T-junction yields three runs meeting at one point.
- Dashed strokes (short collinear fragments separated by small gaps) merge
  into single runs of style "dashed". The Convention Profile's line_styles
  part maps stroke style to line_class (e.g. dashed -> signal); a style the
  profile leaves unmapped keeps the style name as its line_class.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .model import (
    Bbox,
    ConventionProfile,
    LineDetection,
    Point,
    Provenance,
    Sheet,
    SymbolDetection,
)

COMPONENT = "line_extractor:classical-cv"

_INK_THRESHOLD = 127     # binarize: <= is ink (idempotent after normalize)
_BRIDGE_PAD = 3.0        # px: endpoint counts as touching a flow-arrow box
_DASH_GAP_MAX = 6.0      # px: largest gap merged as a dash gap
_DASH_MIN_GAP = 3.0      # px: below this a gap is a raster artifact (e.g.
                         # deskew resampling), and repairing it does not
                         # make the stroke dashed
_DASH_FRAGMENT_MAX = 12.0  # px: longest polyline still considered a dash
_ALIGN_COS = math.cos(math.radians(35))  # merge collinearity tolerance
_RDP_EPSILON = 1.0       # px: polyline simplification tolerance
_MIN_RUN_PX = 3          # ink px below which a leftover fragment is noise

Pixel = tuple[int, int]


# --------------------------------------------------------------------------
# Binarize + split at symbol boxes (masking)

def _ink_pixels(raster: bytes, width: int) -> set[Pixel]:
    return {(i % width, i // width)
            for i, value in enumerate(raster) if value <= _INK_THRESHOLD}


def _mask_box(ink: set[Pixel], bbox: Bbox) -> None:
    x0, y0, x1, y1 = bbox
    for y in range(math.floor(y0), math.ceil(y1) + 1):
        for x in range(math.floor(x0), math.ceil(x1) + 1):
            ink.discard((x, y))


# --------------------------------------------------------------------------
# Thin: Zhang-Suen on the ink set (per-subiteration snapshots keep the
# result independent of set iteration order)

_OFFSETS = ((0, -1), (1, -1), (1, 0), (1, 1),
            (0, 1), (-1, 1), (-1, 0), (-1, -1))  # P2..P9 clockwise


def _thinned(ink: set[Pixel]) -> set[Pixel]:
    ink = set(ink)
    while True:
        changed = False
        for phase in (0, 1):
            to_delete = []
            for x, y in ink:
                nbrs = [1 if (x + dx, y + dy) in ink else 0
                        for dx, dy in _OFFSETS]
                b = sum(nbrs)
                if not 2 <= b <= 6:
                    continue
                a = sum(1 for i in range(8)
                        if nbrs[i] == 0 and nbrs[(i + 1) % 8] == 1)
                if a != 1:
                    continue
                p2, p4, p6, p8 = nbrs[0], nbrs[2], nbrs[4], nbrs[6]
                if phase == 0:
                    if p2 * p4 * p6 or p4 * p6 * p8:
                        continue
                elif p2 * p4 * p8 or p2 * p6 * p8:
                    continue
                to_delete.append((x, y))
            if to_delete:
                ink.difference_update(to_delete)
                changed = True
        if not changed:
            return ink


# --------------------------------------------------------------------------
# Vectorize + junction analysis: arcs between skeleton nodes; adjacent
# branch pixels collapse into one junction point

def _neighbors(pixel: Pixel, ink: set[Pixel]) -> list[Pixel]:
    """8-neighbors, minus diagonal shortcuts: a diagonal step that is
    already connected through an orthogonal ink pixel does not count, so a
    clean L corner reads as degree 2, not as a junction."""
    x, y = pixel
    neighbors = []
    for dx, dy in _OFFSETS:
        q = (x + dx, y + dy)
        if q not in ink:
            continue
        if dx and dy and ((x + dx, y) in ink or (x, y + dy) in ink):
            continue
        neighbors.append(q)
    return neighbors


def _junction_clusters(ink: set[Pixel],
                       degree: dict[Pixel, int]
                       ) -> tuple[dict[Pixel, int], list[Point]]:
    """Group 8-adjacent branch pixels (degree >= 3); each cluster becomes
    one junction point (the centroid) shared by every incident arc."""
    cluster_of: dict[Pixel, int] = {}
    centroids: list[Point] = []
    for pixel in sorted(p for p in ink if degree[p] >= 3):
        if pixel in cluster_of:
            continue
        members = [pixel]
        cluster_of[pixel] = len(centroids)
        stack = [pixel]
        while stack:
            for nbr in _neighbors(stack.pop(), ink):
                if degree[nbr] >= 3 and nbr not in cluster_of:
                    cluster_of[nbr] = len(centroids)
                    members.append(nbr)
                    stack.append(nbr)
        centroids.append((sum(x for x, _ in members) / len(members),
                          sum(y for _, y in members) / len(members)))
    return cluster_of, centroids


def _trace_arcs(ink: set[Pixel], degree: dict[Pixel, int]) -> list[list[Pixel]]:
    arcs: list[list[Pixel]] = []
    visited: set[tuple[Pixel, Pixel]] = set()

    def walk(start: Pixel, step: Pixel) -> list[Pixel]:
        path = [start, step]
        visited.add((start, step))
        visited.add((step, start))
        prev, cur = start, step
        while degree[cur] == 2:
            nxt = next(n for n in _neighbors(cur, ink) if n != prev)
            visited.add((cur, nxt))
            visited.add((nxt, cur))
            path.append(nxt)
            prev, cur = cur, nxt
            if cur == start:
                break  # closed loop back to the origin
        return path

    for node in sorted(p for p in ink if degree[p] != 2):
        for nbr in sorted(_neighbors(node, ink)):
            if (node, nbr) not in visited:
                arcs.append(walk(node, nbr))
    # leftover degree-2 pixels are pure cycles
    covered = {pixel for stride in visited for pixel in stride}
    for pixel in sorted(p for p in ink if degree[p] == 2
                        and p not in covered):
        if all((pixel, n) in visited for n in _neighbors(pixel, ink)):
            continue
        arcs.append(walk(pixel, sorted(_neighbors(pixel, ink))[0]))
    return arcs


# --------------------------------------------------------------------------
# Polyline geometry

def _dist(p: Point, q: Point) -> float:
    return math.hypot(p[0] - q[0], p[1] - q[1])


def _perp_dist(p: Point, a: Point, b: Point) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    if dx == dy == 0:
        return _dist(p, a)
    t = ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    return _dist(p, (a[0] + t * dx, a[1] + t * dy))


def _rdp(points: list[Point], epsilon: float) -> list[Point]:
    if len(points) < 3:
        return list(points)
    a, b = points[0], points[-1]
    index, dmax = 0, -1.0
    for i in range(1, len(points) - 1):
        d = _perp_dist(points[i], a, b)
        if d > dmax:
            index, dmax = i, d
    if dmax > epsilon or a == b:  # closed loops always split
        return (_rdp(points[:index + 1], epsilon)[:-1]
                + _rdp(points[index:], epsilon))
    return [a, b]


def _length(points: list[Point]) -> float:
    return sum(_dist(p, q) for p, q in zip(points, points[1:]))


def _outward(points: list[Point], end: int) -> Point:
    """Unit tangent pointing out of the polyline at endpoint 0 or 1."""
    p, q = (points[1], points[0]) if end == 0 else (points[-2], points[-1])
    length = _dist(p, q)
    return ((q[0] - p[0]) / length, (q[1] - p[1]) / length)


def _in_box(point: Point, bbox: Bbox, pad: float) -> bool:
    x0, y0, x1, y1 = bbox
    return (x0 - pad <= point[0] <= x1 + pad
            and y0 - pad <= point[1] <= y1 + pad)


# --------------------------------------------------------------------------
# Runs and merging (flow-arrow bridging, dash-gap merging)

@dataclass
class _Run:
    points: list[Point]
    support_px: int                  # skeleton pixels the run was traced from
    gaps: int = 0                    # dash gaps merged into it
    bridged: list[str] = field(default_factory=list)  # symbol ids crossed

    @property
    def style(self) -> str:
        return "dashed" if self.gaps else "solid"


def _merge(a: _Run, a_end: int, b: _Run, b_end: int) -> _Run:
    a_pts = list(a.points) if a_end == 1 else list(reversed(a.points))
    b_pts = list(b.points) if b_end == 0 else list(reversed(b.points))
    return _Run(points=a_pts + b_pts,
                support_px=a.support_px + b.support_px,
                gaps=a.gaps + b.gaps,
                bridged=a.bridged + b.bridged)


def _aligned(a: _Run, a_end: int, b: _Run, b_end: int) -> bool:
    """The gap continues run a's direction and run b continues the gap's."""
    gap_vec = (b.points[0 if b_end == 0 else -1][0]
               - a.points[0 if a_end == 0 else -1][0],
               b.points[0 if b_end == 0 else -1][1]
               - a.points[0 if a_end == 0 else -1][1])
    gap_len = math.hypot(*gap_vec)
    if gap_len == 0:
        return True
    gap_dir = (gap_vec[0] / gap_len, gap_vec[1] / gap_len)
    out_a = _outward(a.points, a_end)
    in_b = _outward(b.points, b_end)  # points out of b, i.e. into the gap
    return (out_a[0] * gap_dir[0] + out_a[1] * gap_dir[1] >= _ALIGN_COS
            and -(in_b[0] * gap_dir[0] + in_b[1] * gap_dir[1]) >= _ALIGN_COS)


def _bridge_flow_arrows(runs: list[_Run],
                        transparent: list[SymbolDetection]) -> list[_Run]:
    """A flow arrow annotates a run instead of interrupting it: when exactly
    two run endpoints face each other across the arrow's box, re-join them.
    Anything else is left split — bridging never guesses."""
    for symbol in transparent:
        ends = [(run, end) for run in runs for end in (0, 1)
                if _in_box(run.points[0 if end == 0 else -1],
                           symbol.bbox, _BRIDGE_PAD)]
        if len(ends) != 2:
            continue
        (a, a_end), (b, b_end) = ends
        if a is b or not _aligned(a, a_end, b, b_end):
            continue
        merged = _merge(a, a_end, b, b_end)
        merged.bridged.append(symbol.id)
        runs = [r for r in runs if r is not a and r is not b] + [merged]
    return runs


def _crosses_box(p: Point, q: Point, boxes: list[Bbox]) -> bool:
    mid = ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
    return any(_in_box(mid, box, 0.0) for box in boxes)


def _dash_mergeable(run: _Run) -> bool:
    return run.gaps > 0 or _length(run.points) <= _DASH_FRAGMENT_MAX


def _merge_dashes(runs: list[_Run],
                  splitting_boxes: list[Bbox]) -> list[_Run]:
    """Short collinear fragments with small gaps are one dashed stroke.
    A gap is never merged across a masked symbol box — that split is the
    point of masking."""
    while True:
        candidates = []
        for i, a in enumerate(runs):
            for j, b in enumerate(runs):
                if j <= i:
                    continue
                dashable = _dash_mergeable(a) and _dash_mergeable(b)
                for a_end in (0, 1):
                    for b_end in (0, 1):
                        pa = a.points[0 if a_end == 0 else -1]
                        pb = b.points[0 if b_end == 0 else -1]
                        gap = _dist(pa, pb)
                        if (0 < gap <= _DASH_GAP_MAX
                                and (dashable or gap < _DASH_MIN_GAP)
                                and _aligned(a, a_end, b, b_end)
                                and not _crosses_box(pa, pb,
                                                     splitting_boxes)):
                            candidates.append((gap, i, a_end, j, b_end))
        if not candidates:
            return runs
        gap, i, a_end, j, b_end = min(candidates)
        merged = _merge(runs[i], a_end, runs[j], b_end)
        if gap >= _DASH_MIN_GAP:
            merged.gaps += 1
        runs = [r for k, r in enumerate(runs) if k not in (i, j)] + [merged]


# --------------------------------------------------------------------------
# The stage itself

def extract_line_network(sheet: Sheet,
                         symbols: list[SymbolDetection],
                         profile: ConventionProfile) -> list[LineDetection]:
    if sheet.raster is None:
        raise ValueError(
            f"{COMPONENT} reads pixels; Sheet {sheet.number} carries no "
            "raster")

    transparent = [s for s in symbols
                   if (entry := profile.legend.get(s.symbol_class))
                   is not None and entry.role == "FlowArrow"]
    transparent_ids = {s.id for s in transparent}
    splitting_boxes = [s.bbox for s in symbols
                       if s.id not in transparent_ids]

    ink = _ink_pixels(sheet.raster, sheet.width)
    for symbol in symbols:  # split runs at symbol boxes
        _mask_box(ink, symbol.bbox)
    skeleton = _thinned(ink)

    degree = {p: len(_neighbors(p, skeleton)) for p in skeleton}
    cluster_of, centroids = _junction_clusters(skeleton, degree)

    runs = []
    for arc in _trace_arcs(skeleton, degree):
        if (len(arc) == 2 and arc[0] in cluster_of and arc[1] in cluster_of
                and cluster_of[arc[0]] == cluster_of[arc[1]]):
            continue  # a step inside one junction cluster is not a run
        points = [(float(x), float(y)) for x, y in arc]
        if arc[0] in cluster_of:
            points[0] = centroids[cluster_of[arc[0]]]
        if arc[-1] in cluster_of:
            points[-1] = centroids[cluster_of[arc[-1]]]
        runs.append(_Run(points=_rdp(points, _RDP_EPSILON),
                         support_px=len(arc)))

    runs = _bridge_flow_arrows(runs, transparent)
    runs = _merge_dashes(runs, splitting_boxes)
    runs = [r for r in runs if r.support_px >= _MIN_RUN_PX]

    for run in runs:  # merging leaves collinear joints; canonical orientation
        run.points = _rdp(run.points, _RDP_EPSILON)
        if run.points[-1] < run.points[0]:
            run.points.reverse()
    runs.sort(key=lambda r: (r.points[0], r.points[-1]))

    detections = []
    for i, run in enumerate(runs):
        evidence = (f"traced {run.support_px} skeleton px "
                    f"{run.points[0]} .. {run.points[-1]}, "
                    f"style {run.style}")
        if run.gaps:
            evidence += f", {run.gaps} dash gaps merged"
        if run.bridged:
            evidence += f", bridged across {', '.join(sorted(run.bridged))}"
        detections.append(LineDetection(
            id=f"p{sheet.number}-line{i}",
            sheet=sheet.number,
            polyline=tuple(run.points),
            line_class=profile.line_styles.get(run.style, run.style),
            confidence=1.0,
            provenance=Provenance(component=COMPONENT, evidence=evidence),
        ))
    return detections
