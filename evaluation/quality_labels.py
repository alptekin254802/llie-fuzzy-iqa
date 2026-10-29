"""Optional display labels for the NR-free 0-100 score, not validated human categories."""
import math

QUALITY_BANDS = {
    "Poor": (0, 0, 20, 40),
    "Fair": (20, 40, 50, 65),
    "Good": (50, 65, 75, 90),
    "Excellent": (75, 90, 100, 100),
}


def quality_memberships(score):
    """Return the original trapezoidal display-band memberships."""
    score = float(score)
    if not math.isfinite(score) or not 0 <= score <= 100:
        raise ValueError("Display labels require a finite score in [0, 100].")
    values = {}
    for name, (a, b, c, d) in QUALITY_BANDS.items():
        left = 1.0 if a == b else (score - a) / (b - a)
        right = 1.0 if c == d else (d - score) / (d - c)
        values[name] = max(0.0, min(left, 1.0, right))
    return values


def quality_label(score):
    """Choose maximum membership; an exact tie selects the lower adjacent band.

    This helper does not change the numeric score. NR-mono's raw negative-LPIPS
    output must not be passed directly to this display mapping.
    """
    values = quality_memberships(score)
    return max(values, key=values.get)
