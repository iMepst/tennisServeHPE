import pytest

from assessment.propagation import angular_spread, noise_propagation
from serve_pipeline.config import PipelineConfig
from serve_pipeline.rules import RULES

_MEAN = {r.id: r.mean for r in RULES}


def _sd(criterion, theta, sigma, config):
    return angular_spread(criterion, _MEAN[criterion], theta, sigma,
                          config).sd_deg


def test_spread_grows_with_sigma():
    config = PipelineConfig()
    low = _sd("elbow_flexion", 0.0, 1.0, config)
    high = _sd("elbow_flexion", 0.0, 5.0, config)
    assert high > low


def test_arm_segments_scatter_more_than_trunk_and_leg():
    # A fixed pixel error subtends a larger angle on the shorter arm
    # segments than on the longer trunk and leg segments.
    config = PipelineConfig()
    trunk = _sd("trunk_inclination", 0.0, config.sigma, config)
    knee = _sd("front_knee_flexion", 0.0, config.sigma, config)
    elbow = _sd("elbow_flexion", 0.0, config.sigma, config)
    shoulder = _sd("shoulder_elevation", 0.0, config.sigma, config)
    assert elbow > trunk
    assert elbow > knee
    assert shoulder > trunk


def test_mean_is_unbiased_at_theta_zero():
    # Without projection, the noisy readings centre on the true angle.
    config = PipelineConfig()
    spread = angular_spread("elbow_flexion", _MEAN["elbow_flexion"], 0.0,
                            config.sigma, config)
    assert spread.mean_deg == pytest.approx(_MEAN["elbow_flexion"], abs=0.2)


def test_deterministic_under_fixed_seed():
    config = PipelineConfig()
    first = noise_propagation(config)
    second = noise_propagation(config)
    for a, b in zip(first, second):
        assert a.sd_deg == b.sd_deg
