import csv
import json
import os

from assessment.annotation import EventError, _event_type_error
from assessment.decidability import Decidability
from assessment.report import (_DECIDABILITY_HEADER, _NOISE_HEADER,
                               _PROJECTION_HEADER, build_assessment_report,
                               decidability_rows, event_error_dict,
                               _unreliable_onset)
from assessment.run_measured import SigmaPoint
from serve_pipeline.config import PipelineConfig
from serve_pipeline.rules import RULES


def _fast_config() -> PipelineConfig:
    """A coarse, cheap config: the report logic is what is under test, not the
    Monte-Carlo precision, so shrink the samples and the two sweeps."""
    config = PipelineConfig()
    config.mc_samples = 200
    config.theta_step = 15.0            # thetas 0, 15, 30, 45
    config.sigma_sweep = (2.0, 4.0)
    return config


def _read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def _header(path):
    with open(path) as f:
        return next(csv.reader(f))


def _read_json(path):
    with open(path) as f:
        return json.load(f)

