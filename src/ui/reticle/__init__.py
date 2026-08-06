"""2D reticle guidance display (vendored from USGR).

``reticle.py`` is a verbatim copy of the upstream USGR renderer; the CUSG-facing
wrapper is :class:`ReticleRenderer`. See the module docstrings for details.
"""

from .reticle import render_body_centric_transform
from .reticle_renderer import ReticleRenderer

__all__ = ["ReticleRenderer", "render_body_centric_transform"]
