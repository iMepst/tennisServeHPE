import numpy as np
import pytest

from assessment.propagation import (add_noise, angular_spread,
                                    landmark_points, noise_propagation,
                                    project_points, read_angle)
from serve_pipeline.config import PipelineConfig
from serve_pipeline.rules import RULES

_MEAN = {r.id: r.mean for r in RULES}
_SIGMA = 3.0


def _sd(criterion, theta, sigma, config):
    return angular_spread(criterion, _MEAN[criterion], theta, sigma,
                          config)


def test_spread_grows_with_sigma():
    config = PipelineConfig()
    low = _sd("elbow_flexion", 0.0, 1.0, config)
    high = _sd("elbow_flexion", 0.0, 5.0, config)
    assert high > low


def test_arm_segments_scatter_more_than_trunk_and_leg():
    config = PipelineConfig()
    trunk = _sd("trunk_inclination", 0.0, _SIGMA, config)
    knee = _sd("front_knee_flexion", 0.0, _SIGMA, config)
    elbow = _sd("elbow_flexion", 0.0, _SIGMA, config)
    shoulder = _sd("shoulder_elevation", 0.0, _SIGMA, config)
    assert elbow > trunk
    assert elbow > knee
    assert shoulder > trunk


def test_mean_is_unbiased_at_theta_zero():
    config = PipelineConfig()
    rng = np.random.default_rng(config.seed)
    points = project_points(
        landmark_points("elbow_flexion", _MEAN["elbow_flexion"]), 0.0)
    draws = [read_angle("elbow_flexion", add_noise(points, _SIGMA, rng))
             for _ in range(config.mc_samples)]
    assert np.mean(draws) == pytest.approx(_MEAN["elbow_flexion"], abs=0.2)


def test_deterministic_under_fixed_seed():
    config = PipelineConfig()
    first = noise_propagation(config, _SIGMA)
    second = noise_propagation(config, _SIGMA)
    for a, b in zip(first, second):
        assert a.sd_deg == b.sd_deg
