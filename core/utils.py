"""
core/utils.py
-------------
Small shared helpers used across the app.
"""


def bar_color(score: float) -> str:
    """Return a hex color based on a 0-1 confidence score."""
    if score >= 0.8:
        return "#1D9E75"
    elif score >= 0.5:
        return "#EF9F27"
    else:
        return "#E24B4A"