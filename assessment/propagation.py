"""Landmark noise E1 on top of the projection E2 (Monte Carlo).

Perturb each landmark by an isotropic Gaussian of SD sigma pixels, read the
perturbed landmarks into the angle, and estimate the spread (SD, degrees) by
Monte Carlo.

Per criterion: a fixed pixel error subtends a larger angle across a shorter
segment, so the short arm segments (elbow, shoulder) react more strongly than
the longer trunk and leg segments. That is why segment lengths enter here.

Simplification: the noise is isotropic and independent between frames, whereas
real landmark error is temporally correlated and larger in the fast, blurred
serve phases; the figures are indicative, not exact.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from serve_pipeline.config import PipelineConfig

from assessment.projection import (_tilt_about_vertical, project_orthographic,
                                    theta_values)
from serve_pipeline.angles import vector_angle
from serve_pipeline.rules import RULES

Point3 = Tuple[float, float, float]
Point2 = Tuple[float, float]


def two_segment_points(a_true: float, len1: float, len2: float,
                       chain: bool) -> List[Point3]:
    """Three landmark points for a two-segment criterion at true angle a_true.

    Placed in the image plane (z = 0), symmetric about the +y vertical (where
    the motion plane meets the image plane), so tilting by theta reuses the
    projection convention. Returned as [first, vertex, last], the order the angle
    readers expect.

    chain=True is the knee/elbow turning angle: first sits back along the incoming
    segment, so the turning angle at the vertex equals a_true. chain=False is the
    shoulder V angle: both outer points emanate from the vertex.
    """
    h = math.radians(a_true / 2.0)
    # Two segment directions symmetric about the +y bisector, a_true apart.
    d1 = (-math.sin(h), math.cos(h), 0.0)
    d2 = (math.sin(h), math.cos(h), 0.0)
    vertex = (0.0, 0.0, 0.0)
    # Chain: the incoming segment points into the vertex, so first lies on the
    # far side (-d1). V: the arm points out from the vertex.
    sign = -1.0 if chain else 1.0
    first = (sign * len1 * d1[0], sign * len1 * d1[1], 0.0)
    last = (len2 * d2[0], len2 * d2[1], 0.0)
    return [first, vertex, last]


def trunk_points(a_true: float, length: float) -> List[Point3]:
    """Two landmark points for trunk inclination at true lean a_true.

    Mid-hip at the origin, mid-shoulder one trunk length away, leaning by a_true
    from the +y vertical (the fixed reference, no landmark, no noise). Returned
    as [mid_hip, mid_shoulder].

    Conservative simplification: each mid-point is one perturbed landmark. A real
    mid-point averages two landmarks (quieter by ~1/sqrt(2)), so treating it as
    one overstates the trunk spread, keeping the estimate on the safe side.
    """
    a = math.radians(a_true)
    mid_hip = (0.0, 0.0, 0.0)
    mid_shoulder = (length * math.sin(a), length * math.cos(a), 0.0)
    return [mid_hip, mid_shoulder]

# Representative stature in pixels: a synthetic stand-in, absolute value
# arbitrary and logged with every output. Only the ratios between the segment
# lengths below drive the per-criterion ordering, so the stature cancels out.
REP_STATURE_PX = 600.0

# Segment lengths as fractions of stature (Winter body-segment proportions). The
# arm segments are shortest, which is why elbow and shoulder come out most
# noise-sensitive.
_UPPER_ARM = 0.186
_FOREARM = 0.146
_THIGH = 0.245
_SHANK = 0.246
_TRUNK = 0.288

# Per criterion, the pixel length of each segment whose endpoints carry a
# landmark: two for the joints, one for the trunk (its second reference is the
# fixed image vertical). Elbow and shoulder read short arm segments, so scatter most.
SEGMENT_LENGTHS_PX = {
    "trunk_inclination": (_TRUNK * REP_STATURE_PX,),
    "front_knee_flexion": (_THIGH * REP_STATURE_PX, _SHANK * REP_STATURE_PX),
    "elbow_flexion": (_UPPER_ARM * REP_STATURE_PX, _FOREARM * REP_STATURE_PX),
    "shoulder_elevation": (_UPPER_ARM * REP_STATURE_PX, _TRUNK * REP_STATURE_PX),
}

# How each criterion's angle is formed, fixing how points are built and read:
# "trunk" is one segment vs the vertical, "chain" a turning angle (knee, elbow),
# "vertex" the interior V angle at the shoulder.
CRITERION_KIND = {
    "trunk_inclination": "trunk",
    "front_knee_flexion": "chain",
    "elbow_flexion": "chain",
    "shoulder_elevation": "vertex",
}


def landmark_points(criterion: str, a_true: float) -> List[Point3]:
    """The true 3D landmark points for a criterion at true angle a_true.

    Dispatches on CRITERION_KIND to the matching builder with the
    criterion's representative segment lengths.
    """
    kind = CRITERION_KIND[criterion]
    lengths = SEGMENT_LENGTHS_PX[criterion]
    if kind == "trunk":
        return trunk_points(a_true, lengths[0])
    return two_segment_points(a_true, lengths[0], lengths[1],
                              chain=(kind == "chain"))
