"""Projection error E2: how a monocular view distorts the true angle.

Camera level, projection orthographic (a good approximation when the
player is distant relative to body scale). No recordings are used: the
true angle is prescribed by construction and its projection computed.

theta is the angle between the motion plane (where the joint actually
moves) and the image plane. It is not a single known value -- it comes
partly from camera placement and partly from the player's lean, unknown
before the serve -- so every quantity is evaluated over a sweep of theta.
"""

import math
from typing import List, Tuple

from serve_pipeline.angles import vector_angle
from serve_pipeline.config import PipelineConfig


def _tilt_about_vertical(v: Tuple[float, float, float],
                         theta: float) -> Tuple[float, float, float]:
    """Rotate a 3D direction by theta (deg) about the vertical y axis.

    The vertical is the line where the motion plane meets the image plane,
    so rotating about it by theta swings a point out of the image plane by
    exactly the motion-plane-to-image-plane angle.
    """
    t = math.radians(theta)
    x, y, z = v
    return (x * math.cos(t) + z * math.sin(t),
            y,
            -x * math.sin(t) + z * math.cos(t))


def numeric_projected_angle(a_true: float, theta: float) -> float:
    """Projected enclosed angle of a two-segment joint in degrees.

    Knee, elbow and shoulder are two segments meeting at a joint; unlike
    the trunk there is no closed form, so the projection is evaluated
    numerically. The two segments are placed symmetrically about the
    vertical (the plane-intersection axis), each at half the enclosed
    angle, then the whole joint plane is tilted by theta and each segment
    projected orthographically. The enclosed angle is re-read with the
    same atan2 convention as the pipeline (vector_angle).

    Simplification: both segments share one out-of-plane tilt (a coplanar
    joint). In reality each can tilt independently; capturing that needs a
    two-parameter sweep and is left as a documented limitation.
    """
    h = math.radians(a_true / 2.0)
    # Vertex at the origin, arms symmetric about the +y bisector, initially
    # in the image plane (z = 0).
    arm_left = (-math.sin(h), math.cos(h), 0.0)
    arm_right = (math.sin(h), math.cos(h), 0.0)
    left = project_orthographic(_tilt_about_vertical(arm_left, theta))
    right = project_orthographic(_tilt_about_vertical(arm_right, theta))
    return vector_angle(left, right)


def trunk_projected_angle(a_true: float, theta: float) -> float:
    """Projected trunk inclination in degrees (closed form).

    Trunk inclination is a single line (the trunk axis) read against the
    fixed image vertical, so the projection has a closed form: tilting the
    lean plane by theta foreshortens only the horizontal component, giving
    tan(a_proj) = tan(a_true) * cos(theta). Inputs and output in degrees.
    """
    a = math.radians(a_true)
    t = math.radians(theta)
    return math.degrees(math.atan(math.tan(a) * math.cos(t)))


def project_orthographic(v: Tuple[float, float, float]) -> Tuple[float, float]:
    """Orthographic image of a 3D direction: keep x and y, drop depth z.

    A level camera and parallel projection. This ignores perspective, so
    the projection error it reports is a LOWER BOUND: a real lens adds
    foreshortening on top, more so the closer or more off-centre the
    player. The far-player assumption makes the gap small but non-zero.
    """
    return v[0], v[1]


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
