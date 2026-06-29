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

  3. Continuous mode interpolates red->yellow->green across the fraction, with
     pure yellow at the midpoint of the ``[red_below_frac, green_above_frac]``
     band so the gradient stays aligned with the discrete cut-points.

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
    """Interpolate red->yellow->green smoothly across a 0..1 quality fraction.

    Pure yellow sits at the midpoint of the ``[red_below_frac, green_above_frac]``
    band so the gradient and the discrete cut-points agree on where 'marginal'
    is. The fraction is clamped to ``[0, 1]``.
    """
    t = min(1.0, max(0.0, frac))
    mid = min(1.0, max(0.0, (red_below_frac + green_above_frac) / 2.0))
    if t <= mid:
        u = (t / mid) if mid > 0.0 else 1.0
        return _lerp_rgb(_GRAD_RED, _GRAD_YELLOW, u)
    u = ((t - mid) / (1.0 - mid)) if mid < 1.0 else 0.0
    return _lerp_rgb(_GRAD_YELLOW, _GRAD_GREEN, u)


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
