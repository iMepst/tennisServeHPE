import pytest

from assessment.decidability import (assess_series, band_half_width,
                                     decidability)
from serve_pipeline.config import PipelineConfig
from serve_pipeline.rules import RULES

_RULE = {r.id: r for r in RULES}


def test_half_width_is_one_reference_sd():
    rule = _RULE["trunk_inclination"]
    assert band_half_width(rule) == pytest.approx(rule.sd)


def test_ratio_is_induced_sd_over_half_width():
    for d in decidability(PipelineConfig()):
        for sd, ratio in zip(d.induced_sd, d.ratio):
            assert ratio == pytest.approx(sd / d.half_width)
