"""Projection error E2: how a monocular view distorts the true angle.

Camera level, projection orthographic (good when the player is distant relative
to body scale). No recordings: the true angle is prescribed by construction and
its projection computed.

theta is the angle between the motion plane (where the joint moves) and the
image plane. It is not a single known value (partly camera placement, partly
the player's lean, unknown before the serve), so every quantity is swept over
theta.
"""

import math
from dataclasses import dataclass
from typing import List, Tuple

from serve_pipeline.angles import vector_angle
from serve_pipeline.config import PipelineConfig
from serve_pipeline.rules import RULES


def _tilt_about_vertical(v: Tuple[float, float, float],
                         theta: float) -> Tuple[float, float, float]:
    """Rotate a 3D direction by theta (deg) about the vertical y axis.

    The vertical is where the motion plane meets the image plane, so rotating
    about it by theta swings a point out of the image plane by exactly that
    angle.
    """
    t = math.radians(theta)
    x, y, z = v
    return (x * math.cos(t) + z * math.sin(t),
            y,
            -x * math.sin(t) + z * math.cos(t))


def numeric_projected_angle(a_true: float, theta: float) -> float:
    """Projected enclosed angle of a two-segment joint in degrees.

    Knee, elbow and shoulder are two segments meeting at a joint; unlike the
    trunk there is no closed form. The segments are placed symmetrically about
    the vertical at half the enclosed angle, the joint plane is tilted by
    theta, each segment projected orthographically, and the angle re-read with
    vector_angle.

    Simplification: both segments share one out-of-plane tilt (a coplanar
    joint). Independent tilts would need a two-parameter sweep; left as a
    documented limit.
    """
    h = math.radians(a_true / 2.0)
    arm_left = (-math.sin(h), math.cos(h), 0.0)
    arm_right = (math.sin(h), math.cos(h), 0.0)
    left = project_orthographic(_tilt_about_vertical(arm_left, theta))
    right = project_orthographic(_tilt_about_vertical(arm_right, theta))
    return vector_angle(left, right)


def trunk_projected_angle(a_true: float, theta: float) -> float:
    """Projected trunk inclination in degrees (closed form).

    A single line (trunk axis) against the image vertical, so tilting the lean
    plane by theta foreshortens only the horizontal component: tan(a_proj) =
    tan(a_true) * cos(theta).
    """
    a = math.radians(a_true)
    t = math.radians(theta)
    return math.degrees(math.atan(math.tan(a) * math.cos(t)))


def project_orthographic(v: Tuple[float, float, float]) -> Tuple[float, float]:
    """Ignoring perspective makes the reported error a lower bound: a real
    lens adds foreshortening, more so the closer or more off-centre the
    player. The far-player assumption keeps the gap small.
    """
    return v[0], v[1]


def theta_values(config: PipelineConfig) -> List[float]:
    lo, hi = config.theta_range
    step = config.theta_step
    n = int(round((hi - lo) / step))
    return [lo + i * step for i in range(n + 1)]


@dataclass
class ProjectionCurve:
    """Per-criterion projected angle across the theta sweep.

    a_true is the prescribed true angle (each rule's reference mean);
    projected[i] is how it appears at thetas[i]. kind is "closed_form" (trunk)
    or "numeric".
    """

    criterion: str
    kind: str
    a_true: float
    thetas: List[float]
    projected: List[float]


def projection_curves(config: PipelineConfig) -> List[ProjectionCurve]:
    thetas = theta_values(config)
    curves: List[ProjectionCurve] = []
    for rule in RULES:
        closed = rule.id == "trunk_inclination"
        model = trunk_projected_angle if closed else numeric_projected_angle
        projected = [model(rule.mean, th) for th in thetas]
        curves.append(ProjectionCurve(
            criterion=rule.id,
            kind="closed_form" if closed else "numeric",
            a_true=rule.mean, thetas=thetas, projected=projected))
    return curves
