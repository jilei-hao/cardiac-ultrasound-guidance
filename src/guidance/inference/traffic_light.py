"""Shared, model-agnostic traffic-light colour mapping for the guidance border.

Both traffic-light sources emit a single continuous score:
  - the legacy ResNet18 pose classifier  (``inference/pose_classification``)
  - the USGR quality ScoreTransformer     (``inference/quality_score``)

This module turns ANY such score into a red/yellow/green border colour on a
model-agnostic PERCENTAGE scale, so the SAME discrete cut-points and the SAME
discrete<->continuous toggle apply to either model:

  1. Each model declares its raw score range ``[score_min, score_max]``. The
     score is normalized to a 0..1 "quality fraction" (0% = worst view,
     100% = best). This per-model range is the only model-specific tuning.
  2. Shared percentage thresholds then pick the discrete colour::

        fraction <  red_below_frac    -> red
        fraction >= green_above_frac  -> green
        else                          -> yellow

  3. Continuous mode interpolates red->yellow->green ACROSS THE SAME BAND the
     discrete cut-points define: solid red below ``red_below_frac``, solid green
     at or above ``green_above_frac``, and the gradient in between with pure
     yellow at the band's midpoint. The two modes therefore agree on what is red
     and what is green and differ only in how sharply the middle is drawn --
     and the two thresholds mean the same thing in both, which is what makes
     them tunable while the gradient is on.

The defaults ``(1/3, 2/3)`` reproduce the thresholds that used to be hard-coded
per model: with the quality range ``[-3, 3]`` they recover its ``-1``/``+1``
discrete bounds, and with the legacy range ``[-2, 1]`` they recover the legacy
classifier's ``-1`` (red) / ``0`` (green) bounds.
"""

# Shared discrete cut-points, as fractions of the 0..1 quality scale.
DEFAULT_RED_BELOW_FRAC = 1.0 / 3.0
DEFAULT_GREEN_ABOVE_FRAC = 2.0 / 3.0

# Discrete (quantized) traffic-light colours.
_RED = (255, 0, 0)
_YELLOW = (255, 255, 0)
_GREEN = (0, 255, 0)

# Continuous-gradient anchor colours -- softer than the saturated discrete
# trio (the same palette the USGR reference visualizer uses). red @ 0%,
# yellow @ band midpoint, green @ 100%.
_GRAD_RED = (220, 55, 45)
_GRAD_YELLOW = (230, 210, 45)
_GRAD_GREEN = (55, 210, 70)


def normalize_score(score: float, score_min: float, score_max: float) -> float:
    """Map a raw model score to a 0..1 quality fraction (clamped to [0, 1])."""
    span = score_max - score_min
    if span <= 0:
        return 0.5
    return min(1.0, max(0.0, (float(score) - score_min) / span))


def fraction_to_discrete_rgb(
    frac: float,
    red_below_frac: float = DEFAULT_RED_BELOW_FRAC,
    green_above_frac: float = DEFAULT_GREEN_ABOVE_FRAC,
) -> tuple:
    """Quantize a 0..1 quality fraction to a red/yellow/green (R, G, B)."""
    if frac < red_below_frac:
        return _RED
    if frac >= green_above_frac:
        return _GREEN
    return _YELLOW


def _lerp_rgb(lo: tuple, hi: tuple, u: float) -> tuple:
    return tuple(int(round(lo[i] * (1.0 - u) + hi[i] * u)) for i in range(3))


def fraction_to_continuous_rgb(
    frac: float,
    red_below_frac: float = DEFAULT_RED_BELOW_FRAC,
    green_above_frac: float = DEFAULT_GREEN_ABOVE_FRAC,
) -> tuple:
    """Interpolate red->yellow->green across the band the cut-points define.

    Outside the band the colour is SOLID: red below ``red_below_frac``, green at
    or above ``green_above_frac``, exactly where the discrete mode puts them.
    Only the middle is a gradient, with pure yellow at the band's midpoint.

    This is what makes the two thresholds tunable in continuous mode. An earlier
    version spread the gradient over the whole 0..1 range and used the cut-points
    only to locate yellow, so widening or narrowing the band did nothing -- every
    pair with the same average produced an identical border -- and at the
    discrete red boundary the colour was already two-thirds of the way to yellow.

    The fraction is clamped to ``[0, 1]``. Red is tested first, so a caller that
    passes an inverted pair gets the same answer the discrete mode gives.
    """
    t = min(1.0, max(0.0, frac))
    if t < red_below_frac:
        return _GRAD_RED
    if t >= green_above_frac:
        return _GRAD_GREEN
    # Only reachable when green_above_frac > red_below_frac: a band of zero or
    # negative width leaves no t that fails both tests above, so the midpoint
    # below cannot divide by zero.
    mid = (red_below_frac + green_above_frac) / 2.0
    if t <= mid:
        return _lerp_rgb(_GRAD_RED, _GRAD_YELLOW,
                         (t - red_below_frac) / (mid - red_below_frac))
    return _lerp_rgb(_GRAD_YELLOW, _GRAD_GREEN,
                     (t - mid) / (green_above_frac - mid))


def score_to_border_rgb(
    score: float,
    score_min: float,
    score_max: float,
    *,
    red_below_frac: float = DEFAULT_RED_BELOW_FRAC,
    green_above_frac: float = DEFAULT_GREEN_ABOVE_FRAC,
    continuous: bool = False,
) -> tuple:
    """Normalize a raw score then map it to the border (R, G, B).

    The single entry point both the legacy and quality borders go through.
    ``continuous=True`` returns a smooth gradient; otherwise the discrete
    three-colour traffic light.
    """
    frac = normalize_score(score, score_min, score_max)
    if continuous:
        return fraction_to_continuous_rgb(frac, red_below_frac, green_above_frac)
    return fraction_to_discrete_rgb(frac, red_below_frac, green_above_frac)
