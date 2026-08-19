"""Synthetic degradation transforms (ticket 14): blur, skew, noise, and
compression over the Sheet raster form, offline and stdlib-only.

The geometric alignment criterion is proven here on a drawn fixture with
exactly known geometry: a skewed box must still surround its skewed ink.
The factory emitting clean and degraded variants side by side is covered
in test_degrade_factory.py.
"""

import math
import random

import pytest

from pidgraph.degrade import (
    Blur,
    Compression,
    Noise,
    Raster,
    Skew,
    Variant,
    clip_bbox,
    degrade_bbox,
    degrade_polyline,
    degrade_raster,
    map_point,
    polyline_visible,
    transform_config,
    transform_from_config,
    variant_from_config,
)


def uniform(width: int, height: int, value: int = 255) -> Raster:
    return Raster(width, height, bytes([value]) * (width * height))


def with_ink(raster: Raster, dots: dict) -> Raster:
    pixels = bytearray(raster.pixels)
    for (x, y), value in dots.items():
        pixels[y * raster.width + x] = value
    return Raster(raster.width, raster.height, bytes(pixels))


def ink_pixels(raster: Raster, threshold: int = 128) -> list:
    return [(x, y)
            for y in range(raster.height)
            for x in range(raster.width)
            if raster.pixels[y * raster.width + x] < threshold]


def test_raster_refuses_pixels_that_do_not_fill_the_canvas():
    with pytest.raises(ValueError, match="3x2"):
        Raster(3, 2, b"\xff" * 5)


# --------------------------------------------------------------------------
# Blur: separable box blur, severity = radius in pixels

def test_blur_averages_ink_over_its_window_exactly():
    raster = with_ink(uniform(9, 9), {(4, 4): 0})
    blurred = Blur(radius=1).apply(raster, random.Random(0))
    expected = round((8 * 255) / 9)  # the 3x3 window holding the dot
    for y in range(9):
        for x in range(9):
            value = blurred.pixels[y * 9 + x]
            if 3 <= x <= 5 and 3 <= y <= 5:
                assert value == expected
            else:
                assert value == 255
    assert (blurred.width, blurred.height) == (9, 9)


def test_blur_radius_scales_the_spread():
    raster = with_ink(uniform(9, 9), {(4, 4): 0})
    touched = [sum(1 for v in Blur(radius=r).apply(
                   raster, random.Random(0)).pixels if v != 255)
               for r in (1, 2)]
    assert touched == [9, 25]


def test_blur_leaves_a_uniform_raster_unchanged():
    raster = uniform(7, 5, value=128)
    assert Blur(radius=2).apply(raster, random.Random(0)) == raster


def test_blur_refuses_a_non_positive_radius():
    with pytest.raises(ValueError, match="radius"):
        Blur(radius=0)


# --------------------------------------------------------------------------
# Noise: additive gaussian, severity = sigma in gray levels, seeded

def test_noise_is_reproducible_per_seed_and_differs_across_seeds():
    raster = uniform(16, 16, value=128)
    once = degrade_raster(raster, (Noise(sigma=8.0),), seed=7)
    again = degrade_raster(raster, (Noise(sigma=8.0),), seed=7)
    other = degrade_raster(raster, (Noise(sigma=8.0),), seed=8)
    assert once.pixels == again.pixels
    assert once.pixels != other.pixels
    assert once.pixels != raster.pixels


def test_noise_sigma_scales_the_deviation():
    raster = uniform(32, 32, value=128)

    def deviation(sigma):
        noisy = degrade_raster(raster, (Noise(sigma=sigma),), seed=1)
        return sum(abs(v - 128) for v in noisy.pixels) / len(noisy.pixels)

    assert deviation(40.0) > deviation(2.0) > 0.0


def test_noise_clamps_to_the_byte_range():
    paper = degrade_raster(uniform(16, 16, value=255),
                           (Noise(sigma=300.0),), seed=3)
    assert max(paper.pixels) == 255       # clamped, not wrapped
    assert min(paper.pixels) < 255        # and genuinely noisy


def test_noise_refuses_a_non_positive_sigma():
    with pytest.raises(ValueError, match="sigma"):
        Noise(sigma=0.0)


# --------------------------------------------------------------------------
# Skew: rotation about the raster center; the one geometric transform,
# so its point map is the contract labels align through

def test_skew_map_fixes_the_center_and_preserves_distance():
    x, y = Skew(angle_deg=90.0).map_point(50.0, 30.0, 100, 60)
    assert (x, y) == pytest.approx((50.0, 30.0))
    x, y = Skew(angle_deg=90.0).map_point(70.0, 30.0, 100, 60)
    assert (x, y) == pytest.approx((50.0, 50.0))


def test_skew_moves_ink_exactly_where_the_map_says():
    raster = with_ink(uniform(41, 41), {(30, 20): 0})
    skewed = Skew(angle_deg=90.0).apply(raster, random.Random(0))
    # pixel (30, 20) has center (30.5, 20.5); the 90-degree map about
    # the canvas center (20.5, 20.5) lands it on pixel (20, 30)
    assert skewed.pixels[30 * 41 + 20] < 20
    assert skewed.pixels[20 * 41 + 30] == 255
    assert (skewed.width, skewed.height) == (41, 41)


def test_skew_fills_uncovered_canvas_with_paper():
    skewed = Skew(angle_deg=45.0).apply(uniform(40, 40, value=0),
                                        random.Random(0))
    assert skewed.pixels[0] == 255                      # corner rotated out
    assert skewed.pixels[20 * 40 + 20] == 0             # center still ink


def test_skewed_box_still_surrounds_the_skewed_ink():
    """The alignment criterion, on a fixture with known geometry: a box
    drawn at exactly known pixels, skewed together with its label box,
    keeps every ink pixel inside the transformed box (within bilinear
    bleed) — and the transform genuinely moved the geometry."""
    width, height = 200, 120
    x0, y0, x1, y1 = 60, 40, 140, 80
    pixels = bytearray(b"\xff" * (width * height))
    for x in range(x0, x1 + 1):
        pixels[y0 * width + x] = 0
        pixels[y1 * width + x] = 0
    for y in range(y0, y1 + 1):
        pixels[y * width + x0] = 0
        pixels[y * width + x1] = 0
    raster = Raster(width, height, bytes(pixels))
    bbox = [x0, y0, x1 + 1, y1 + 1]      # surrounds the ink pixel centers

    transforms = (Skew(angle_deg=5.0),)
    skewed = degrade_raster(raster, transforms, seed=0)
    box = degrade_bbox(transforms, bbox, width, height)

    ink = ink_pixels(skewed)
    assert ink, "the skewed raster lost its ink"
    pad = 1.5  # bilinear resampling spreads ink under 1.5 px
    assert all(box[0] - pad <= x + 0.5 <= box[2] + pad
               and box[1] - pad <= y + 0.5 <= box[3] + pad
               for x, y in ink)
    # the skew genuinely moved geometry: the clean box no longer holds
    assert any(not (bbox[0] - pad <= x + 0.5 <= bbox[2] + pad
                    and bbox[1] - pad <= y + 0.5 <= bbox[3] + pad)
               for x, y in ink)
    assert max(abs(a - b) for a, b in zip(box, bbox)) > pad


def test_skew_refuses_a_non_finite_angle():
    with pytest.raises(ValueError, match="angle"):
        Skew(angle_deg=math.inf)


# --------------------------------------------------------------------------
# Compression: JPEG-style 8x8 block quantization, severity = quality

def test_compression_reconstructs_flat_blocks_exactly():
    raster = uniform(16, 16, value=200)
    assert Compression(quality=50).apply(raster, random.Random(0)) == raster


def gradient(width: int, height: int) -> Raster:
    return Raster(width, height,
                  bytes(min(255, x * 3 + y * 2)
                        for y in range(height) for x in range(width)))


def test_compression_quality_scales_the_artifacts():
    raster = gradient(24, 24)

    def error(quality):
        crushed = Compression(quality=quality).apply(raster,
                                                     random.Random(0))
        return sum(abs(a - b) for a, b in
                   zip(crushed.pixels, raster.pixels)) / len(raster.pixels)

    assert error(10) > error(90)
    assert error(90) < 6.0


def test_compression_handles_sizes_not_a_multiple_of_the_block():
    raster = gradient(13, 10)
    crushed = Compression(quality=30).apply(raster, random.Random(0))
    assert (crushed.width, crushed.height) == (13, 10)
    assert crushed == Compression(quality=30).apply(raster,
                                                    random.Random(9))


def test_compression_refuses_an_out_of_range_quality():
    with pytest.raises(ValueError, match="quality"):
        Compression(quality=0)
    with pytest.raises(ValueError, match="quality"):
        Compression(quality=101)


# --------------------------------------------------------------------------
# Composition and reproducibility across a transform sequence

def test_degrade_raster_applies_transforms_in_order():
    raster = with_ink(uniform(16, 16), {(8, 8): 0})
    blur_then_noise = degrade_raster(raster, (Blur(1), Noise(30.0)), seed=5)
    noise_then_blur = degrade_raster(raster, (Noise(30.0), Blur(1)), seed=5)
    assert blur_then_noise.pixels != noise_then_blur.pixels


def test_a_full_chain_is_reproducible_and_seed_sensitive():
    raster = with_ink(uniform(24, 24), {(10, 12): 0, (15, 6): 40})
    chain = (Blur(1), Skew(2.0), Noise(8.0), Compression(30))
    assert degrade_raster(raster, chain, seed=7) \
        == degrade_raster(raster, chain, seed=7)
    assert degrade_raster(raster, chain, seed=7).pixels \
        != degrade_raster(raster, chain, seed=8).pixels


def test_degrading_with_no_transforms_is_the_identity():
    raster = uniform(4, 4, value=9)
    assert degrade_raster(raster, (), seed=0) == raster


# --------------------------------------------------------------------------
# Label geometry follows the same point map as the raster

def test_degrade_bbox_encloses_the_mapped_corners():
    box = degrade_bbox((Skew(angle_deg=90.0),), [10.0, 20.0, 30.0, 40.0],
                       100, 100)
    assert box == pytest.approx([60.0, 10.0, 80.0, 30.0])


def test_degrade_polyline_maps_each_point():
    polyline = degrade_polyline((Skew(angle_deg=90.0),),
                                [[10.0, 20.0], [30.0, 40.0]], 100, 100)
    assert polyline[0] == pytest.approx([80.0, 10.0])
    assert polyline[1] == pytest.approx([60.0, 30.0])


def test_photometric_transforms_leave_geometry_unchanged():
    transforms = (Blur(2), Noise(20.0), Compression(10))
    assert degrade_bbox(transforms, [1.0, 2.0, 3.0, 4.0], 50, 50) \
        == [1.0, 2.0, 3.0, 4.0]
    assert map_point(transforms, 5.0, 6.0, 50, 50) == (5.0, 6.0)


def test_clip_bbox_intersects_the_canvas_or_drops():
    assert clip_bbox([10.0, 20.0, 30.0, 40.0], 100, 50) \
        == [10.0, 20.0, 30.0, 40.0]
    assert clip_bbox([-5.0, 10.0, 30.0, 60.0], 100, 50) \
        == [0.0, 10.0, 30.0, 50.0]
    assert clip_bbox([120.0, 10.0, 140.0, 40.0], 100, 50) is None
    assert clip_bbox([-10.0, -10.0, -1.0, -1.0], 100, 50) is None


def test_polyline_visibility_is_the_canvas_overlap():
    assert polyline_visible([[10.0, 10.0], [200.0, 10.0]], 100, 50)
    assert not polyline_visible([[110.0, 10.0], [200.0, 10.0]], 100, 50)
    assert not polyline_visible([[10.0, 60.0], [90.0, 80.0]], 100, 50)


# --------------------------------------------------------------------------
# Configuration: transforms and variants from plain JSON-shaped dicts

def test_transform_configs_round_trip():
    for transform in (Blur(2), Skew(-1.5), Noise(12.0),
                      Compression(40)):
        assert transform_from_config(transform_config(transform)) \
            == transform


def test_an_unknown_transform_kind_is_refused_naming_the_kinds():
    with pytest.raises(ValueError, match="ripple.*blur|blur.*ripple"):
        transform_from_config({"kind": "ripple"})


def test_an_unknown_transform_parameter_is_refused_by_name():
    with pytest.raises(ValueError, match="radios"):
        transform_from_config({"kind": "blur", "radios": 2})


def test_variant_refuses_names_that_are_not_path_safe():
    for name in ("", "sc/an", "a b", ".."):
        with pytest.raises(ValueError, match="name"):
            Variant(name, (Blur(1),))


def test_variant_name_refusal_states_the_actual_rule():
    # '_x' satisfies "[A-Za-z0-9._-]" but not the leading-character
    # rule, so the message must state the rule the regex enforces
    with pytest.raises(ValueError, match="letter or digit"):
        Variant("_x", (Blur(1),))


def test_variant_refuses_an_empty_transform_sequence():
    with pytest.raises(ValueError, match="transform"):
        Variant("scan", ())


def test_variant_refuses_a_non_integer_seed():
    with pytest.raises(ValueError, match="seed"):
        Variant("scan", (Blur(1),), seed="7")


def test_variant_from_config_parses_transforms_and_defaults_the_seed():
    variant = variant_from_config(
        {"name": "scan", "transforms": [{"kind": "blur", "radius": 2},
                                        {"kind": "noise", "sigma": 6.0}]})
    assert variant == Variant("scan", (Blur(2), Noise(6.0)), seed=0)
    seeded = variant_from_config(
        {"name": "scan", "seed": 5,
         "transforms": [{"kind": "skew", "angle_deg": 1.0}]})
    assert seeded.seed == 5


def test_variant_from_config_refuses_unknown_keys():
    with pytest.raises(ValueError, match="sneed"):
        variant_from_config({"name": "scan", "sneed": 1,
                             "transforms": [{"kind": "blur"}]})
