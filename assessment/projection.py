"""Projection error E2: how a monocular view distorts the true angle.

Camera level, projection orthographic (a good approximation when the
player is distant relative to body scale). No recordings are used: the
true angle is prescribed by construction and its projection computed.

theta is the angle between the motion plane (where the joint actually
moves) and the image plane. It is not a single known value -- it comes
partly from camera placement and partly from the player's lean, unknown
before the serve -- so every quantity is evaluated over a sweep of theta.
"""

from typing import List

from serve_pipeline.config import PipelineConfig


def theta_values(config: PipelineConfig) -> List[float]:
    """The theta sweep in degrees, inclusive of both range ends.

    Enumerated from config.theta_range in steps of config.theta_step, so
    the same range feeds the projection curves and the later decidability
    criterion.
    """
    lo, hi = config.theta_range
    step = config.theta_step
    # Number of steps between the bounds; +1 to include the upper end.
    n = int(round((hi - lo) / step))
    return [lo + i * step for i in range(n + 1)]
