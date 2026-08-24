"""The border's two cut-points mean the same thing in both border modes.

The traffic light has a discrete mode (three colours) and a continuous one (a
gradient), and both are driven by ``red_below_frac`` / ``green_above_frac`` on a
model-agnostic 0..1 quality scale. Those are tuning knobs: they are meant to be
turned, and what they do has to be the same either way.

An earlier continuous mode spread its gradient over the whole 0..1 range and
used the two thresholds only to locate yellow, at their midpoint. Two
consequences, neither of which raises anything:

  - every pair with the same average produced an IDENTICAL border, so widening
    or narrowing the band did nothing at all;
  - at the discrete red cut-point the colour was already two-thirds of the way
    to yellow, and at the green cut-point only one-third of the way to green, so
    the two modes disagreed about what counted as red or green.

Pure Python -- no torch, no models.
"""

import pytest

from guidance.inference.traffic_light import (
    DEFAULT_GREEN_ABOVE_FRAC,
    DEFAULT_RED_BELOW_FRAC,
    fraction_to_continuous_rgb,
    fraction_to_discrete_rgb,
    normalize_score,
    score_to_border_rgb,
)

RED, GREEN = DEFAULT_RED_BELOW_FRAC, DEFAULT_GREEN_ABOVE_FRAC


def test_below_the_red_cut_point_the_gradient_is_solid_red():
    solid = fraction_to_continuous_rgb(0.0, RED, GREEN)
    for frac in (0.0, 0.1, 0.2, RED - 1e-9):
        assert fraction_to_continuous_rgb(frac, RED, GREEN) == solid


def test_at_or_above_the_green_cut_point_the_gradient_is_solid_green():
    solid = fraction_to_continuous_rgb(1.0, RED, GREEN)
    for frac in (GREEN, 0.8, 0.95, 1.0):
        assert fraction_to_continuous_rgb(frac, RED, GREEN) == solid


def test_the_modes_agree_on_what_is_red_and_what_is_green():
    """Not the same colours -- the gradient palette is softer -- but the same
    verdict at every boundary, which is what the cut-points are for."""
    for frac in (0.0, 0.2, RED - 1e-9):
        assert fraction_to_discrete_rgb(frac, RED, GREEN) == (255, 0, 0)
        assert fraction_to_continuous_rgb(frac, RED, GREEN) == \
            fraction_to_continuous_rgb(0.0, RED, GREEN)
    for frac in (GREEN, 0.9, 1.0):
        assert fraction_to_discrete_rgb(frac, RED, GREEN) == (0, 255, 0)
        assert fraction_to_continuous_rgb(frac, RED, GREEN) == \
            fraction_to_continuous_rgb(1.0, RED, GREEN)


def test_the_band_is_a_gradient_and_not_a_step():
    """Guards the two tests above from passing on a merely-discrete mapping."""
    inside = [fraction_to_continuous_rgb(f, RED, GREEN)
              for f in (0.36, 0.42, 0.5, 0.58, 0.64)]
    assert len(set(inside)) == len(inside), 'the band quantized into steps'
    # G-R, not the channels individually: red rises slightly on the red->yellow
    # leg (220 -> 230) because the anchors are a designed palette, not a ramp.
    # G-R is what increases monotonically along the whole gradient, and it is
    # what vigil's border_color bucketing reads.
    warmth = [int(c[1]) - int(c[0]) for c in inside]
    assert warmth == sorted(warmth), f'G-R is not monotonic across the band: {warmth}'
    assert warmth[-1] > warmth[0]


def test_pure_yellow_sits_at_the_band_midpoint():
    mid = (RED + GREEN) / 2.0
    r, g, b = fraction_to_continuous_rgb(mid, RED, GREEN)
    assert r > 200 and g > 200 and b < 100, f'midpoint is not yellow: {(r, g, b)}'


def test_widening_the_band_changes_the_border():
    """The regression this file exists for: the knobs must actually do something.

    Under the old mapping these two pairs had the same average and therefore
    produced byte-identical borders at every fraction.
    """
    narrow = [fraction_to_continuous_rgb(f, 0.45, 0.55) for f in (0.3, 0.5, 0.7)]
    wide = [fraction_to_continuous_rgb(f, 0.20, 0.80) for f in (0.3, 0.5, 0.7)]
    assert narrow != wide


def test_moving_the_band_moves_the_boundaries():
    strict = fraction_to_continuous_rgb(0.7, 0.6, 0.9)
    lax = fraction_to_continuous_rgb(0.7, 0.1, 0.4)
    assert lax == fraction_to_continuous_rgb(1.0, 0.1, 0.4), 'lax: expected green'
    assert strict != lax


@pytest.mark.parametrize('red_frac,green_frac', [(0.5, 0.5), (0.7, 0.3)])
def test_a_degenerate_band_does_not_divide_by_zero(red_frac, green_frac):
    """Zero or inverted width leaves no gradient to draw, and must not raise."""
    for frac in (0.0, 0.3, 0.5, 0.7, 1.0):
        assert len(fraction_to_continuous_rgb(frac, red_frac, green_frac)) == 3


def test_the_entry_point_normalizes_before_mapping():
    """score_to_border_rgb is the only path either border actually takes."""
    assert normalize_score(-3.0, -3.0, 3.0) == 0.0
    assert normalize_score(3.0, -3.0, 3.0) == 1.0
    assert score_to_border_rgb(-3.0, -3.0, 3.0, continuous=True) == \
        fraction_to_continuous_rgb(0.0, RED, GREEN)
    assert score_to_border_rgb(3.0, -3.0, 3.0, continuous=False) == (0, 255, 0)
